"""Trainer: warm-up, early stopping, validation metrics, oracle switch, tw_nndf diagnostics, run_one().

Protocol (frozen by construction, see docstrings):
  * train graph: labels are ``y_obs`` / ``pnu`` at the label cut; ``y_true`` is dropped by the loader.
    ORACLE run: the train loader keeps ``y_true`` and ``NaiveBCELoss`` receives ``y_true`` as the target.
  * validation (``y_true``, full labels) selects the best epoch and the FPR=1% threshold; the test set is only
    scored once, after the best epoch is restored. Test labels never select anything.
  * scores are LOGITS.
Nothing here is a reproduction of a paper; see the open points in the round-6 notes.
"""
from __future__ import annotations

import copy
import math
import time
import gc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn as nn

from src.losses.calibration import (
    MatureTemperatureTracker, SlidingWindowPrior, sliding_window_prior, unlabeled_prior, young_counts,
)
from src.losses.maturity_mask import MaturityMaskLoss
from src.losses.tw_nndf import HybridTWnnDFLoss, TimeWindowNNDFLoss, estimate_class_prior
from src.losses.tw_nndf import HybridTWnnDFLoss
from src.data.dataset import load_phase_datasets, read_meta, seed_labels
from src.eval import metrics as M
from src.graph.builder import PNU_CODES, TXN
from src.losses.tw_nndf import HybridTWnnDFLoss, TimeWindowNNDFLoss
from src.losses import (
    NaiveBCELoss, TimeWindowNNDFLoss, build_loss, compute_age_days, estimate_class_prior,
    lam_from_median_delay,
)
from src.losses import DelayIPWLoss
from src.models.classifier import FraudClassifier, entity_vocab_sizes_from_meta
from src.train.utils import (
    PREREG_STATUS, EarlyStopper, Stopwatch, get_device, make_run_id, peak_memory_mb, reset_peak_memory,
    set_seed, warmup_factor,
)

LOSS_NAMES = ("naive_bce", "maturity_mask", "delay_ipw", "tw_nndf")
MONITORS = ("pr_auc", "roc_auc", "recall_at_fpr")
NNDF_KEYS = ("pos_risk", "neg_risk_raw", "clamped", "ascent", "bce")
PRIOR_MODES = ("mature_mean", "sliding_window", "sliding_unlabeled", "sliding_dynamic")
AMOUNT_KEYS = ("amount_aware", "amount_quantile", "amount_min_age_days")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    loss: str = "naive_bce"
    oracle: bool = False                 # True: NaiveBCE on y_true (needs expose_train_true_labels)
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-5
    warmup_epochs: int = 1
    patience: int = 5
    min_delta: float = 0.0
    monitor: str = "pr_auc"
    grad_clip: float | None = 1.0
    seed: int = 42
    device: str = "auto"
    fpr: float = 0.01
    review_cost: float = 0.0
    prior: float | None = None           # tw_nndf: None -> estimated on the mature region (observed labels only)
    loss_params: dict[str, Any] = field(default_factory=dict)
    max_train_batches: int | None = None  # debug / profiling only
    save_dir: str | None = None

    def __post_init__(self) -> None:
        if self.loss not in LOSS_NAMES:
            raise ValueError(f"loss must be one of {LOSS_NAMES}.")
        if self.monitor not in MONITORS:
            raise ValueError(f"monitor must be one of {MONITORS}.")
        if self.epochs < 1 or self.warmup_epochs < 0 or self.warmup_epochs >= self.epochs:
            raise ValueError("need epochs >= 1 and 0 <= warmup_epochs < epochs.")

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any], **overrides: Any) -> "TrainConfig":
        """Read ``train:``, ``eval:`` and ``loss:`` of base.yaml; keyword overrides win."""
        t, e, lc = dict(cfg.get("train", {})), dict(cfg.get("eval", {})), dict(cfg.get("loss", {}))
        name = overrides.pop("loss", None) or lc.get("name", "naive_bce")
        clip = t.get("grad_clip", 1.0)
        kw: dict[str, Any] = dict(
            loss=name,
            loss_params=dict(lc.get(name, {}) or {}),
            epochs=int(t.get("epochs", 30)),
            lr=float(t.get("lr", 1e-3)),
            weight_decay=float(t.get("weight_decay", 1e-5)),
            warmup_epochs=int(t.get("warmup_epochs", 1)),
            patience=int(t.get("patience", 5)),
            min_delta=float(t.get("min_delta", 0.0)),
            monitor=str(t.get("monitor", "pr_auc")),
            grad_clip=None if clip is None else float(clip),
            seed=int(cfg.get("seed", 42)),
            fpr=float(e.get("fpr", 0.01)),
            review_cost=float(e.get("review_cost", 0.0)),
        )
        kw.update(overrides)
        return cls(**kw)


# --------------------------------------------------------------------------- #
# Loss construction (+ prior / delay diagnostics for the log)
# --------------------------------------------------------------------------- #

def build_training_loss(cfg: TrainConfig, train_ds: Any, meta: Mapping[str, Any]) -> tuple[nn.Module, dict[str, Any]]:
    """Return ``(loss module, info dict)``. ``info`` is stored with every run for the prior analysis.

    Only ``train_ds`` is given: no validation / test statistic can reach a loss through this function. The p95
    amount, mean amount and every prior come from the TRAIN graph (train-view rows) only.
    """
    if cfg.oracle:
        return NaiveBCELoss(), {"loss": "naive_bce", "oracle": True}
    full = train_ds.data[TXN]
    label_cut = float(train_ds.data.label_cut)
    window = float(meta["config"]["delay"]["maturity_window_days"])
    in_u = (full.pnu == PNU_CODES["U"]) & full.in_train_view
    age_all = compute_age_days(full.time, label_cut)
    amt_view = full.amount[full.in_train_view].float()  # TRAIN graph, train-view rows only
    info: dict[str, Any] = {
        "loss": cfg.loss, "oracle": False, "maturity_window_days": window,
        "n_P": int((full.pnu == PNU_CODES["P"]).sum()), "n_N": int((full.pnu == PNU_CODES["N"]).sum()),
        "n_U": int(in_u.sum()),
        # DIAGNOSTIC ONLY (reads y_true to document the prior mismatch; never passed to any loss)
        "diag_true_fraud_rate_in_U": float(full.y_true[in_u].float().mean()) if bool(in_u.any()) else None,
    }
    params = dict(cfg.loss_params)

    def amount_rule(aw: Mapping[str, Any]) -> tuple[float | None, float]:
        """Window + Amount-Aware rule shared by maturity_mask and tw_nndf: ``(p95_train or None, min_age)``."""
        if not bool(aw.get("amount_aware", False)):
            return None, 0.0
        q = float(aw.get("amount_quantile", 0.95))
        if not 0.0 < q < 1.0:
            raise ValueError("amount_quantile must be in (0, 1).")
        if amt_view.numel() == 0:
            raise ValueError("No train-view rows: cannot compute the amount quantile.")
        thr = float(torch.quantile(amt_view, q))  # TRAIN amounts only
        mn = aw.get("amount_min_age_days")
        min_age = window / 2.0 if mn is None else float(mn)  # default W/2 = design choice (Threats to Validity)
        trusted = in_u & (full.amount >= thr) & (age_all >= min_age)
        info.update(
            amount_aware=True, amount_quantile=q, amount_threshold=thr, amount_min_age_days=min_age,
            n_U_trusted=int(trusted.sum()),
            # DIAGNOSTIC ONLY (reads y_true: share of hidden fraud among the U rows treated as negatives)
            diag_true_fraud_rate_in_trusted_U=(
                float(full.y_true[trusted].float().mean()) if bool(trusted.any()) else None),
        )
        return thr, min_age

    if cfg.loss == "maturity_mask":
        thr, min_age = amount_rule({k: params.pop(k) for k in AMOUNT_KEYS if k in params})
        if thr is not None:
            params.update(amount_threshold=thr, amount_min_age_days=min_age)
        return MaturityMaskLoss(**params), info

    if cfg.loss == "tw_nndf":
        prior_mode = str(params.pop("prior_mode", "mature_mean"))
        span_days = float(params.pop("prior_span_days", 14.0))
        n_windows = int(params.pop("prior_n_windows", 3))
        shrink = float(params.pop("prior_shrink", 200.0))
        variant = str(params.pop("variant", "pu_only"))
        lambda_pu = float(params.pop("lambda_pu", 1.0))
        calib_cfg = dict(params.pop("calibration", None) or {})
        aw = {k: params.pop(k) for k in AMOUNT_KEYS if k in params}
        if prior_mode not in PRIOR_MODES:
            raise ValueError(f"prior_mode must be one of {PRIOR_MODES}.")
        if variant not in ("pu_only", "hybrid"):
            raise ValueError("loss.tw_nndf.variant must be 'pu_only' or 'hybrid'.")
        if "unlabeled_statuses" in params:
            params["unlabeled_statuses"] = tuple(params["unlabeled_statuses"])

        est = estimate_class_prior(full.y_obs, age_all, window, full.in_train_view)  # observed labels only
        prior_auto = est
        info.update(prior_estimated=est)
        if prior_mode != "mature_mean":
            sw = sliding_window_prior(full.y_obs, age_all, window, in_train_view=full.in_train_view,
                                      span_days=span_days, n_windows=n_windows)
            prior_auto = sw.prior  # sliding_dynamic uses it as the fallback scalar
            info.update(prior_sliding=sw.prior, prior_window_rates=list(sw.window_rates),
                        prior_window_counts=list(sw.window_counts))
            if prior_mode == "sliding_unlabeled":
                if tuple(params.get("unlabeled_statuses", ("U",))) != ("U",):
                    raise ValueError("prior_mode='sliding_unlabeled' needs unlabeled_statuses == ['U'].")
                n_p, n_u = young_counts(full.pnu, age_all, window, full.in_train_view)
                prior_auto = unlabeled_prior(sw.prior, n_p, n_u)
                info.update(n_P_young=n_p, n_U_young=n_u)
        prior = cfg.prior if cfg.prior is not None else prior_auto
        source = "config" if cfg.prior is not None else ("estimated" if prior_mode == "mature_mean" else prior_mode)
        info.update(prior=prior, prior_mode=prior_mode, prior_source=source, variant=variant)
        if bool(params.get("amount_weighted", False)):
            params["amount_ref"] = float(amt_view.mean())  # mean TRAIN amount
            info.update(amount_weighted=True, amount_ref=params["amount_ref"])

        if variant == "pu_only":  # V1 / V1.5 behaviour: hybrid-only keys were popped above and are ignored
            return TimeWindowNNDFLoss(prior=prior, **params), info

        thr, min_age = amount_rule(aw)
        sp = None
        if prior_mode == "sliding_dynamic" and cfg.prior is None:  # --prior override pins a fixed pi
            sp = SlidingWindowPrior.fit(full.y_obs, age_all, window, full.in_train_view,
                                        span_days=span_days, shrink=shrink)
            info.update(prior_dynamic_min=float(sp.table.min()), prior_dynamic_max=float(sp.table.max()))
        tracker = MatureTemperatureTracker(**calib_cfg) if bool(calib_cfg.pop("enabled", True)) else None
        info.update(lambda_pu=lambda_pu, calibration=tracker is not None)
        return HybridTWnnDFLoss(prior=prior, lambda_pu=lambda_pu, amount_threshold=thr,
                                amount_min_age_days=min_age, calibrator=tracker, sliding_prior=sp, **params), info

    if cfg.loss == "delay_ipw":
        lam = params.pop("lam", None)
        info["lam_defaulted"] = lam is None
        lam = lam_from_median_delay(window / 2.0) if lam is None else float(lam)  # arbitrary default: sweep it
        info["lam"] = lam
        return DelayIPWLoss(lam=lam, **params), info

    return build_loss(cfg.loss, **params), info


def compute_batch_loss(
    model: nn.Module, loss_fn: nn.Module, batch: Any, label_cut: float, oracle: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward + loss on the seed rows of one batch (shared with scripts/profile_epoch.py)."""
    labels = seed_labels(batch)
    if oracle:
        if "y_true" not in labels:
            raise KeyError("Oracle run needs y_true in the train batches (expose_train_true_labels=True).")
        y = labels["y_true"]
    else:
        y = labels["y_obs"]
    age = compute_age_days(labels["time"], label_cut)
    kwargs: dict[str, Any] = {"age_days": age, "in_train_view": labels["in_train_view"]}
    if getattr(loss_fn, "needs_amount", False):  # amount-aware masking / cost-sensitive nnDF only
        if "amount" not in labels:
            raise KeyError("This loss needs `amount` in the batches.")
        kwargs["amount"] = labels["amount"]
    logits = model(batch)
    loss = loss_fn(logits, y, labels["pnu"], **kwargs)
    return loss, logits


# --------------------------------------------------------------------------- #
# One step / prediction
# --------------------------------------------------------------------------- #
def compute_batch_loss(
    model: nn.Module, loss_fn: nn.Module, batch: Any, label_cut: float, oracle: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward + loss on the seed rows of one batch (shared with scripts/profile_epoch.py)."""
    labels = seed_labels(batch)
    if oracle:
        if "y_true" not in labels:
            raise KeyError("Oracle run needs y_true in the train batches (expose_train_true_labels=True).")
        y = labels["y_true"]
    else:
        y = labels["y_obs"]
    age = compute_age_days(labels["time"], label_cut)
    logits = model(batch)
    extra = {"amount": labels["amount"]} if getattr(loss_fn, "needs_amount", False) else {}
    loss = loss_fn(logits, y, labels["pnu"], age_days=age, in_train_view=labels["in_train_view"], **extra)
    return loss, logits


@torch.no_grad()
def predict(model: nn.Module, ds: Any, device: torch.device, max_batches: int | None = None) -> dict[str, np.ndarray]:
    """Logits and evaluation fields for every seed of ``ds`` (needs ``y_true`` and ``amount`` in the batches)."""
    was_training = model.training
    model.eval()
    scores, ys, amounts = [], [], []
    try:
        for i, batch in enumerate(ds):
            if max_batches is not None and i >= max_batches:
                break
            batch = batch.to(device)
            labels = seed_labels(batch)
            if "y_true" not in labels or "amount" not in labels:
                raise KeyError("Evaluation needs y_true and amount in the batches (expose_true_labels=True).")
            scores.append(model(batch).detach().float().cpu().numpy())
            ys.append(labels["y_true"].cpu().numpy())
            amounts.append(labels["amount"].cpu().numpy().astype(np.float64))
    finally:
        model.train(was_training)
    return {"scores": np.concatenate(scores), "y_true": np.concatenate(ys), "amount": np.concatenate(amounts)}


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(
        self, model: nn.Module, loss_fn: nn.Module, train_ds: Any, val_ds: Any, cfg: TrainConfig,
        *, log: Callable[[str], None] = print,
    ) -> None:
        self.cfg = cfg
        self.device = get_device(cfg.device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.train_ds, self.val_ds = train_ds, val_ds
        self.label_cut = float(train_ds.data.label_cut)
        self.log = log
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        n = len(train_ds)
        self.steps_per_epoch = n if cfg.max_train_batches is None else min(n, cfg.max_train_batches)
        self.warmup_steps = cfg.warmup_epochs * self.steps_per_epoch
        self.global_step = 0
        self._is_nndf = isinstance(loss_fn, TimeWindowNNDFLoss)

    # ---- one epoch ---------------------------------------------------------- #
    def train_epoch(self) -> dict[str, float]:
        cfg, dev = self.cfg, self.device
        self.model.train()
        sw = Stopwatch(dev)
        sw.start()
        loss_sum = torch.zeros((), device=dev)
        risk_sum = {k: torch.zeros((), device=dev) for k in NNDF_KEYS}
        n_used = n_empty = 0
        for step, batch in enumerate(self.train_ds):
            if cfg.max_train_batches is not None and step >= cfg.max_train_batches:
                break
            batch = batch.to(dev)
            lr = cfg.lr * warmup_factor(self.global_step, self.warmup_steps)
            for g in self.opt.param_groups:
                g["lr"] = lr
            self.global_step += 1

            loss, _ = compute_batch_loss(self.model, self.loss_fn, batch, self.label_cut, cfg.oracle)
            self.opt.zero_grad(set_to_none=True)
            if self._is_nndf and self.loss_fn.last_risks is None:  # no P or no U in this batch
                n_empty += 1
                continue
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.opt.step()
            loss_sum += loss.detach()
            n_used += 1
            if self._is_nndf:  # tensors accumulated on device: no GPU sync per step
                for k in NNDF_KEYS:
                    risk_sum[k] += self.loss_fn.last_risks.get(k, loss_sum.new_zeros(())).to(dev)  # pu_only has no "bce"
        elapsed = sw.stop()

        denom = max(n_used, 1)
        mean_loss = float(loss_sum) / denom
        if not math.isfinite(mean_loss):
            raise FloatingPointError("Training loss is not finite: training diverged.")
        out = {"train_loss": mean_loss, "n_steps": float(n_used), "n_empty_batches": float(n_empty),
               "lr_end": float(self.opt.param_groups[0]["lr"]), "train_sec": elapsed}
        if self._is_nndf:
            out.update({"nndf_pos_risk": float(risk_sum["pos_risk"]) / denom,
                        "nndf_neg_risk_raw": float(risk_sum["neg_risk_raw"]) / denom,
                        "nndf_clamped_rate": float(risk_sum["clamped"]) / denom,
                        "nndf_bce": float(risk_sum["bce"]) / denom,
                        "nndf_ascent_rate": float(risk_sum["ascent"]) / denom})
        return out

    # ---- evaluation ---------------------------------------------------------- #
    def evaluate(self, ds: Any) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        out = predict(self.model, ds, self.device)
        return M.summarize_scores(out["y_true"], out["scores"], fpr=self.cfg.fpr), out

    def evaluate_test(self, test_ds: Any, threshold: float) -> dict[str, float]:
        """Score the test set ONCE with the threshold fitted on validation."""
        m, out = self.evaluate(test_ds)
        op = M.operating_point(out["y_true"], out["scores"], out["amount"], threshold,
                               review_cost=self.cfg.review_cost)
        return {**m, **{f"op_{k}": v for k, v in op.items()}}

    # ---- full training run ---------------------------------------------------- #
    def fit(self) -> dict[str, Any]:
        cfg = self.cfg
        stopper = EarlyStopper(cfg.patience, cfg.min_delta, start_epoch=cfg.warmup_epochs + 1)
        best_state: dict[str, torch.Tensor] | None = None
        history: list[dict[str, float]] = []
        stopped_early = False
        reset_peak_memory(self.device)
        t0 = time.perf_counter()

        for epoch in range(1, cfg.epochs + 1):
            tr = self.train_epoch()
            end = getattr(self.loss_fn, "end_epoch", None)      # HybridTWnnDFLoss only: refit T on TRAIN rows
            if end is not None:
                t_new = end()
                if t_new is not None:
                    tr["calib_T"] = float(t_new)                 # shown as T= in the epoch log
                    cal = getattr(self.loss_fn, "calibrator", None)
                    if cal is not None and cal.last_raw_temperature is not None:
                        tr["calib_T_raw"] = float(cal.last_raw_temperature)   # pre-clamp value
            va, _ = self.evaluate(self.val_ds)
            improved = stopper.update(va[cfg.monitor], epoch)
            if improved:
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                if cfg.save_dir:
                    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
                    torch.save({"epoch": epoch, "state_dict": best_state, "cfg": asdict(cfg)},
                               Path(cfg.save_dir) / "best.pt")
            history.append({"epoch": float(epoch), **tr, "val_pr_auc": va["pr_auc"],
                            "val_roc_auc": va["roc_auc"], "val_recall_at_fpr": va["recall_at_fpr"]})
            extra = (f" | nndf neg_raw={tr['nndf_neg_risk_raw']:+.4f} clamped={tr['nndf_clamped_rate']:.2f}"
                     f" ascent={tr['nndf_ascent_rate']:.2f} bce={tr['nndf_bce']:.4f} T={tr.get('calib_T', float('nan')):.3f}"
                     f" empty={int(tr['n_empty_batches'])}") if self._is_nndf else ""
            self.log(f"epoch {epoch:>2} loss={tr['train_loss']:.4f} val_prauc={va['pr_auc']:.4f} "
                     f"val_auc={va['roc_auc']:.4f} val_rec@fpr={va['recall_at_fpr']:.4f} "
                     f"{'*' if improved else ' '} ({tr['train_sec']:.0f}s){extra}")
            if stopper.should_stop:
                stopped_early = True
                break

        if best_state is None:
            raise RuntimeError("No finite validation score: training diverged.")
        self.model.load_state_dict(best_state)
        val, val_out = self.evaluate(self.val_ds)
        thr = val["threshold_at_fpr"]
        val_op = M.operating_point(val_out["y_true"], val_out["scores"], val_out["amount"], thr,
                                   review_cost=cfg.review_cost)
        return {
            "best_epoch": stopper.best_epoch, "epochs_run": len(history), "stopped_early": stopped_early,
            "history": history, "val": val, "val_op": val_op, "val_threshold": thr,
            "train_seconds": time.perf_counter() - t0, "peak_vram_mb": peak_memory_mb(self.device),
        }


# --------------------------------------------------------------------------- #
# One benchmark run
# --------------------------------------------------------------------------- #
def run_one(
    cfg: Mapping[str, Any],
    processed_dir: str | Path,
    *,
    loss: str | None = None,
    oracle: bool = False,
    backbone: str | None = None,
    use_temporal: bool | None = None,
    lp: bool | None = None,
    seed: int | None = None,
    prior: float | None = None,          # tw_nndf prior override (prior-sensitivity group only)
    device: str = "auto",
    save_dir: str | None = None,
    max_train_batches: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train one configuration end to end; return ``{"row": flat dict for the results CSV, "details": ...}``."""
    pdir = Path(processed_dir)
    meta = read_meta(pdir)
    tcfg = TrainConfig.from_cfg(cfg, loss=loss, oracle=oracle, device=device, save_dir=save_dir,
                                 max_train_batches=max_train_batches, prior=prior,
                                **({} if seed is None else {"seed": seed}))
    set_seed(tcfg.seed)

    features_cfg = copy.deepcopy(dict(cfg.get("features", {})))
    if lp is not None:
        features_cfg["lp_features"] = {**(features_cfg.get("lp_features") or {}), "enabled": bool(lp)}
    sampler_cfg = dict(cfg["sampler"])
    dsets = load_phase_datasets(pdir, sampler_cfg, phases=("train", "val"), features_cfg=features_cfg,
                                expose_train_true_labels=oracle)
    train_ds, val_ds = dsets["train"], dsets["val"]

    model_cfg = dict(cfg["model"])
    if use_temporal is not None:
        model_cfg["use_temporal"] = use_temporal
    backbone = backbone or str(model_cfg.get("backbone", "sage"))
    model = FraudClassifier.from_config(model_cfg, in_dim=train_ds.model_in_dim,  # 457, or 461 with LP features
                                        entity_vocab_sizes=entity_vocab_sizes_from_meta(meta), conv_type=backbone)
    loss_fn, info = build_training_loss(tcfg, train_ds, meta)

    lp_on = train_ds.lp_features is not None
    gru_on = bool(model_cfg.get("use_temporal", True))
    label = "oracle" if oracle else tcfg.loss
    run_id = make_run_id(pdir.name, label, backbone, gru_on, lp_on, tcfg.seed)
    log(f"[{run_id}] in_dim={train_ds.model_in_dim} params={model.num_parameters():,} info={info}")

    trainer = Trainer(model, loss_fn, train_ds, val_ds, tcfg, log=log)
    fit = trainer.fit()
    trainer.train_ds = trainer.val_ds = None   # Trainer holds references: drop them or `del` frees nothing
    del dsets, train_ds, val_ds
    gc.collect()
    test_ds = load_phase_datasets(pdir, sampler_cfg, phases=("test",), features_cfg=features_cfg)["test"]
    test = trainer.evaluate_test(test_ds, fit["val_threshold"])

    row = {
        "run_id": run_id, "delay": pdir.name, "loss": label, "oracle": oracle, "backbone": backbone,
        "use_temporal": gru_on, "lp_features": lp_on, "seed": tcfg.seed, "in_dim": model.backbone.txn_in[0].in_features,
        "n_params": model.num_parameters(), "prereg_status": PREREG_STATUS,
        "best_epoch": fit["best_epoch"], "epochs_run": fit["epochs_run"],
        "val_pr_auc": fit["val"]["pr_auc"], "val_roc_auc": fit["val"]["roc_auc"],
        "test_pr_auc": test["pr_auc"], "test_roc_auc": test["roc_auc"],
        "test_recall_at_fpr": test["recall_at_fpr"],            # own-curve recall at FPR <= fpr
        "test_recall_at_valthr": test["op_recall"], "test_fpr_at_valthr": test["op_fpr"],
        "test_efl": test["op_efl"], "test_efl_ratio": test["op_efl_ratio"],
        "train_seconds": fit["train_seconds"],
        "peak_vram_alloc_mb": None if fit["peak_vram_mb"] is None else fit["peak_vram_mb"]["allocated"],
    }
    return {"row": row, "details": {"info": info, "fit": fit, "test": test, "cfg": asdict(tcfg)}}


__all__ = ["TrainConfig", "build_training_loss", "compute_batch_loss", "predict", "Trainer", "run_one"]