#!/usr/bin/env python
"""Pre-flight checks, one part at a time. Nothing here trains a model.

    python scripts/preflight_checks.py --part 1      # gradient scale bce vs pu, amount weights (dummy data)
    python scripts/preflight_checks.py --part 2      # T tracker timing / saturation, prior drift table (dummy data)
    python scripts/preflight_checks.py --part 2 --processed_dir data/processed/delay_high   # + REAL priors (train graph)
    python scripts/preflight_checks.py --part 3 [--runs_dir results/runs]   # model selection, pos_weight scan
    python scripts/preflight_checks.py --part 4 --processed_dir data/processed/delay_high   # causality (needs pyg-lib)
    python scripts/preflight_checks.py --part 5 --base configs/base.yaml --hybrid configs/base_hybrid.yaml

Tags: PASS / WARN (look at it) / FAIL (fix before the full run) / SKIP (input missing).
Thresholds marked "heuristic" are my own choice (KHÔNG CHẮC), not standards. Exit code 1 if any FAIL.
The dummy world has NO hidden positives inside U: it isolates the loss mechanics, not real data behaviour.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from src.graph.builder import PNU_CODES

P, N, U = PNU_CODES["P"], PNU_CODES["N"], PNU_CODES["U"]
MISSING = object()


class Report:
    def __init__(self) -> None:
        self.count = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}

    def add(self, tag: str, msg: str) -> None:
        self.count[tag] += 1
        print(f"  [{tag}] {msg}")


def _f(x) -> float:
    return float("nan") if x is None else float(x)


# --------------------------------------------------------------------------- #
# Dummy data
# --------------------------------------------------------------------------- #
def make_batch(n: int = 4096, p_rate: float = 0.035, u_frac: float = 0.25, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    n_p, n_u = max(int(n * p_rate), 8), int(n * u_frac)
    n_n = n - n_p - n_u
    pnu = torch.cat([torch.full((n_p,), P), torch.full((n_n,), N), torch.full((n_u,), U)]).long()
    y = (pnu == P).float()
    age = torch.where(pnu == U, torch.full((n,), 5.0), torch.full((n,), 40.0))
    view = torch.ones(n, dtype=torch.bool)
    amt = torch.exp(torch.randn(n, generator=g) * 1.2 + 3.5)
    amt[torch.randperm(n, generator=g)[:5]] = 1e5          # extreme outliers
    return pnu, y, age, view, amt


def make_logits(pnu: torch.Tensor, kind: str, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(pnu.numel(), generator=g) * 0.05
    return torch.where(pnu == P, z + 3.0, z - 3.0) if kind == "separated" else z


# --------------------------------------------------------------------------- #
# PART 1: gradient scale of bce vs lambda_pu * pu, amount weights
# --------------------------------------------------------------------------- #
def part1(rep: Report, args) -> None:
    print("PART 1: gradient scale of bce vs lambda_pu * pu; amount weights")
    try:
        from src.losses.tw_nndf import HybridTWnnDFLoss as H
    except ImportError as exc:
        rep.add("FAIL", f"HybridTWnnDFLoss not importable: {exc}")
        return
    pnu, y, age, view, amt = make_batch()

    def grad(fn, z):
        z = z.clone().requires_grad_(True)
        loss = fn(z, y, pnu, age_days=age, in_train_view=view, amount=amt)
        loss.backward()
        return z.grad.detach(), fn.last_risks

    for prior in args.priors:
        for kind in ("init", "separated"):
            z = make_logits(pnu, kind)
            g_b, _ = grad(H(prior, mode=args.mode, lambda_pu=0.0), z)
            g_t, r = grad(H(prior, mode=args.mode, lambda_pu=1.0), z)
            g_pu = g_t - g_b
            nb, npu = float(g_b.norm()), float(g_pu.norm())
            ratio = args.lambda_pu * npu / max(nb, 1e-30)
            u_live = float((g_pu[pnu == U].abs() > 0).float().mean())
            tag, note = "PASS", ""
            if npu <= 1e-3 * nb:
                tag, note = "WARN", " | PU term ~inert (heuristic: < 1e-3 x BCE)"
            elif ratio > 3.0:
                tag, note = "WARN", " | PU dominates (heuristic: lambda*PU > 3 x BCE)"
            rep.add(tag, f"prior={prior:.4f} logits={kind:<9} |g_bce|={nb:.3e} |g_pu|={npu:.3e} "
                         f"lambda*pu/bce={ratio:.2f} ascent={float(r['ascent']):.0f} clamped={float(r['clamped']):.0f} "
                         f"U_rows_with_PU_grad={u_live:.0%}{note}")

    try:
        from src.train.trainer import _amount_ref_mean_one as ref_fn
    except ImportError:
        ref_fn = None
        rep.add("WARN", "trainer._amount_ref_mean_one missing: amount_ref = plain mean (capped weights average < 1)")
    refs = {"mean": float(amt.mean())}
    if ref_fn is not None:
        refs["capped_mean=1"] = ref_fn(amt, args.max_weight)
    for name, ref in refs.items():
        w = (amt / ref).clamp(max=args.max_weight)
        tag = "PASS" if abs(float(w.mean()) - 1.0) < 0.05 else "WARN"
        capped = float((amt / ref > args.max_weight).float().mean())
        rep.add(tag, f"amount_ref={name} ({ref:.1f}): mean w={float(w.mean()):.3f} max w={float(w.max()):.1f} "
                     f"capped rows={capped:.1%}")
    ref = refs.get("capped_mean=1", refs["mean"])
    z = make_logits(pnu, "init")
    g_w, _ = grad(H(args.priors[0], mode=args.mode, lambda_pu=1.0, amount_weighted=True, amount_ref=ref,
                    max_weight=args.max_weight), z)
    g_u, _ = grad(H(args.priors[0], mode=args.mode, lambda_pu=1.0), z)
    ratio = float(g_w.abs().max()) / max(float(g_u.abs().max()), 1e-30)
    rep.add("PASS" if ratio <= 2 * args.max_weight else "WARN",
            f"max |row grad| weighted/unweighted = {ratio:.1f} (cap {args.max_weight:g}; heuristic bound 2x cap)")


# --------------------------------------------------------------------------- #
# PART 2: T tracker (timing, saturation) and prior
# --------------------------------------------------------------------------- #
def part2(rep: Report, args) -> None:
    print("PART 2: temperature tracker and prior")
    try:
        from src.losses.calibration import MatureTemperatureTracker, unlabeled_prior
        from src.losses.tw_nndf import HybridTWnnDFLoss as H, TimeWindowNNDFLoss as PU
    except ImportError as exc:
        rep.add("FAIL", f"import failed: {exc}")
        return
    prior = args.priors[0]

    # 2a: when does T change?
    torch.manual_seed(0)
    pnu, y, age, view, amt = make_batch(n=2048)
    x = torch.randn(pnu.numel(), 8)
    x[pnu == P] += 1.0
    lin = torch.nn.Linear(8, 1)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    tracker = MatureTemperatureTracker()
    loss_fn = H(prior, mode=args.mode, lambda_pu=args.lambda_pu, calibrator=tracker)
    for epoch in range(1, 4):
        seen = set()
        for _ in range(5):
            seen.add(round(tracker.temperature, 9))
            opt.zero_grad()
            loss_fn(lin(x).squeeze(1), y, pnu, age_days=age, in_train_view=view, amount=amt).backward()
            opt.step()
        t_old, t_new = tracker.temperature, loss_fn.end_epoch()
        at_bound = min(abs(t_new - tracker.t_min), abs(t_new - tracker.t_max)) < 1e-6
        tag = "WARN" if (len(seen) != 1 or at_bound) else "PASS"
        rep.add(tag, f"epoch {epoch}: T constant inside the epoch={len(seen) == 1} ({t_old:.3f}) -> after refit {t_new:.3f}"
                     f"{' | T AT ITS BOUND' if at_bound else ''}")
    g = torch.Generator().manual_seed(0)
    z0 = torch.randn(20000, generator=g) * 2.0
    yy = torch.bernoulli(torch.sigmoid(z0), generator=g)
    for scale in (0.3, 10.0):
        tr = MatureTemperatureTracker()
        tr.collect(scale * z0, yy)
        t = tr.refit()
        hit = min(abs(t - tr.t_min), abs(t - tr.t_max)) < 1e-6
        rep.add("WARN" if hit else "PASS", f"logits x{scale:g} (true T={1 / scale:.2f}): fitted T={t:.3f}"
                                           f"{' (clamped: calibration cannot fully correct this)' if hit else ''}")

    # 2b: does a small / large T kill the gradient of the PU term?
    z_sep = make_logits(pnu, "separated")

    def u_grad(scale: float, T: float):
        z = (z_sep * scale).clone().requires_grad_(True)
        PU(prior, mode=args.mode)(z / T, y, pnu, age_days=age, in_train_view=view).backward()
        return z.grad.detach()

    g0 = u_grad(1.0, 1.0)
    base_norm = max(float(g0.norm()), 1e-30)
    base_u = max(float(g0[pnu == U].abs().mean()), 1e-30)
    print(f"  (baseline |g|={float(g0.norm()):.3e} at logit scale 1, T=1; 0 means the PU gradient is clamped to zero)")
    for scale in (1.0, 2.5):
        for T in (0.25, 1.0, 4.0):
            gg = u_grad(scale, T)
            sat = float((gg[pnu == U].abs() < 1e-3 * base_u).float().mean())
            tag = "WARN" if sat > 0.5 else "PASS"
            rep.add(tag, f"logit scale {scale:g}, T={T:g}: |g|/|g_base|={float(gg.norm()) / base_norm:.2f} "
                         f"U rows with grad < 1e-3 x baseline={sat:.0%}{' | vanishing (heuristic > 50%)' if sat > 0.5 else ''}")

    # 2c: when does the unlabeled prior degenerate?
    pi, n_young = 0.035, 100_000
    for p_young in (1000, 2000, 3000, 3500, 4000):
        n_u = n_young - p_young
        raw = (pi * n_young - p_young) / n_u
        used = unlabeled_prior(pi, p_young, n_u)
        rep.add("WARN" if raw <= 1e-4 else "PASS",
                f"pi={pi} n_young={n_young} observed P in young={p_young}: raw pi_U={raw:.4f} -> used {used:.4f}"
                f"{' (CLIPPED: nnPU degenerates to U = negative)' if raw <= 1e-4 else ''}")

    # 2d: REAL priors from the train graph
    if not args.processed_dir:
        rep.add("SKIP", "real priors: pass --processed_dir data/processed/<delay>")
        return
    from src.data.dataset import load_phase_graph, read_meta
    from src.train.trainer import TrainConfig, build_training_loss
    pdir = Path(args.processed_dir)
    meta = read_meta(pdir)
    ds = SimpleNamespace(data=load_phase_graph(pdir / "train.pt", validate=False))
    for mode in ("mature_mean", "sliding_window", "sliding_unlabeled"):
        cfg = TrainConfig(loss="tw_nndf", loss_params={"mode": "nnpu", "unlabeled_statuses": ["U"], "prior_mode": mode})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, info = build_training_loss(cfg, ds, meta)
        pr = float(info["prior"])
        true_u = info.get("diag_true_fraud_rate_in_U")  # diagnostic only (reads y_true)
        extreme = pr < 1e-3 or pr > 0.1 or bool(caught)
        rep.add("WARN" if extreme else "PASS",
                f"{mode:<18} prior={pr:.4f} true rate in U (diag)={_f(true_u):.4f} "
                f"raw_pi_U={_f(info.get('prior_unlabeled_raw')):.4f} warnings={len(caught)}")


# --------------------------------------------------------------------------- #
# PART 3: model selection, pos_weight
# --------------------------------------------------------------------------- #
def part3(rep: Report, args) -> None:
    print("PART 3: model selection and class weighting")
    for path in (args.base, args.hybrid):
        p = Path(path)
        if not p.exists():
            rep.add("SKIP", f"{path} not found")
            continue
        monitor = yaml.safe_load(p.read_text(encoding="utf-8"))["train"].get("monitor")
        rep.add("PASS" if monitor == "pr_auc" else "WARN", f"{path}: train.monitor = {monitor} (best epoch is picked on validation {monitor})")

    hits = [str(f.relative_to(REPO_ROOT)) for f in list((REPO_ROOT / "src" / "losses").glob("*.py"))
            + [REPO_ROOT / "src" / "train" / "trainer.py"] if f.exists() and "pos_weight" in f.read_text(encoding="utf-8")]
    rep.add("FAIL" if hits else "PASS", "pos_weight used in: " + (", ".join(hits) if hits else "no loss / trainer (same footing for all methods)"))

    from src.train.utils import EarlyStopper
    curve = [0.35, 0.31, 0.29, 0.28, 0.28, 0.27, 0.27]       # shape seen at V1 (illustrative, not data)
    st = EarlyStopper(args.patience, 0.0, start_epoch=args.warmup + 1)
    stopped = None
    for e, v in enumerate(curve, 1):
        st.update(v, e)
        if st.should_stop:
            stopped = e
            break
    rep.add("WARN" if st.best_epoch <= args.warmup + 1 else "PASS",
            f"illustrative decreasing curve: best_epoch={st.best_epoch}, stops at epoch {stopped} -> the restored model is "
            f"the epoch-{st.best_epoch} model (warmup={args.warmup}, patience={args.patience})")

    if not args.runs_dir:
        rep.add("SKIP", "run JSONs: pass --runs_dir results/runs")
        return
    files = sorted(glob.glob(str(Path(args.runs_dir) / "*.json")))
    if not files:
        rep.add("SKIP", f"no JSON in {args.runs_dir}")
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        fit = d.get("fit") or {}
        hist = fit.get("history") or []
        if not hist:
            continue
        best = int(fit["best_epoch"])
        row = next((h for h in hist if int(h["epoch"]) == best), {})
        rep.add("WARN" if best <= args.warmup + 1 else "PASS",
                f"{Path(f).stem}: best_epoch={best}/{fit.get('epochs_run')} "
                f"val_pr_auc best={_f(row.get('val_pr_auc')):.4f} last={_f(hist[-1].get('val_pr_auc')):.4f}")


# --------------------------------------------------------------------------- #
# PART 4: causality on val / test (real graphs, needs pyg-lib)
# --------------------------------------------------------------------------- #
def part4(rep: Report, args) -> None:
    print("PART 4: causal sampling and label invariance on validation / test")
    if not args.processed_dir:
        rep.add("SKIP", "pass --processed_dir data/processed/<delay>")
        return
    import numpy as np
    from src.data.dataset import load_phase_datasets, load_phase_graph, read_meta
    from src.graph.builder import TXN
    from src.graph.sampler import CausalityError, build_neighbor_loader, has_pyg_lib, verify_batch_causality
    from src.losses.canary import model_leakage_hook
    from src.models.classifier import FraudClassifier, entity_vocab_sizes_from_meta
    if not has_pyg_lib():
        rep.add("SKIP", "pyg-lib missing: temporal sampling impossible (the code refuses to fall back)")
        return
    pdir = Path(args.processed_dir)
    cfg = yaml.safe_load(Path(args.base).read_text(encoding="utf-8"))
    dsets = load_phase_datasets(pdir, dict(cfg["sampler"]), phases=("val", "test"))

    for phase, ds in dsets.items():                        # 4a: no node / edge later than its seed
        try:
            n = 0
            for batch in ds:
                verify_batch_causality(batch, strict_time=ds.strict_time)
                n += 1
                if n >= args.n_batches:
                    break
            rep.add("PASS", f"{phase}: {n} batches, no node / edge later than its seed (strict_time={ds.strict_time})")
        except CausalityError as exc:
            rep.add("FAIL", f"{phase}: {exc}")

    va = load_phase_graph(pdir / "val.pt", validate=False)  # 4b: delete-the-future, val.pt vs test.pt
    te = load_phase_graph(pdir / "test.pt", validate=False)
    rng = np.random.default_rng(0)
    cand = torch.nonzero(va[TXN].seed_mask).view(-1)[-5000:]
    seeds = cand[torch.from_numpy(rng.choice(len(cand), 50, replace=False))].sort().values

    def history(data):
        m = torch.zeros(data[TXN].num_nodes, dtype=torch.bool)
        m[seeds] = True
        loader = build_neighbor_loader(data, [15, 10], batch_size=len(seeds), seed_mask=m, temporal_strategy="last")
        b = next(iter(loader))
        verify_batch_causality(b)
        st, bs = b[TXN], int(b[TXN].batch_size)
        return st.tid[:bs], [sorted(st.tid[st.batch == i].tolist()) for i in range(bs)]

    s_va, h_va = history(va)
    s_te, h_te = history(te)
    same = torch.equal(s_va, s_te) and h_va == h_te
    rep.add("PASS" if same else "FAIL", "history of val seeds is identical when the TEST rows are added to the graph"
                                         if same else "history of a val seed CHANGED when later rows were added: the future leaks")

    ds = dsets["val"]                                      # 4c: logits do not depend on label fields
    meta = read_meta(pdir)
    model = FraudClassifier.from_config(dict(cfg["model"]), in_dim=ds.model_in_dim,
                                        entity_vocab_sizes=entity_vocab_sizes_from_meta(meta),
                                        conv_type=str(cfg["model"].get("backbone", "sage")))
    try:
        model_leakage_hook(model, next(iter(ds)))
        rep.add("PASS", "model output is invariant to y_obs / y_true / pnu of a real validation batch")
    except AssertionError as exc:
        rep.add("FAIL", f"label leaks into the forward pass: {exc}")


# --------------------------------------------------------------------------- #
# PART 5: configs
# --------------------------------------------------------------------------- #
EXPECT_COMMON = {
    ("train", "lr"): 0.001, ("train", "weight_decay"): 1e-5, ("train", "epochs"): 30, ("train", "patience"): 5,
    ("train", "warmup_epochs"): 1, ("train", "monitor"): "pr_auc", ("train", "grad_clip"): 1.0,
    ("sampler", "batch_size"): 1024, ("sampler", "strict_time"): True, ("sampler", "temporal_strategy"): "uniform",
    ("eval", "fpr"): 0.01, ("eval", "review_cost"): 0.0, ("model", "use_temporal"): True, ("model", "hidden_dim"): 128,
}
EXPECT_HYBRID = {
    ("loss", "tw_nndf", "variant"): "hybrid", ("loss", "tw_nndf", "mode"): "nnpu", ("loss", "tw_nndf", "lambda_pu"): 0.3,
    ("loss", "tw_nndf", "amount_weighted"): True, ("loss", "tw_nndf", "max_weight"): 10.0,
    ("loss", "tw_nndf", "amount_aware"): True, ("loss", "tw_nndf", "amount_min_age_days"): None,
    ("loss", "tw_nndf", "calibration", "enabled"): True,
}
INFO_KEYS = [("loss", "tw_nndf", "prior_mode"), ("loss", "tw_nndf", "beta"), ("loss", "tw_nndf", "gamma"),
             ("loss", "maturity_mask", "amount_aware"), ("loss", "delay_ipw", "lam")]


def dig(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return MISSING
        d = d[k]
    return d


def part5(rep: Report, args) -> None:
    print("PART 5: configs (leftovers of debug / smoke runs)")
    cfgs = {}
    for name, path in (("base", args.base), ("hybrid", args.hybrid)):
        p = Path(path)
        if p.exists():
            cfgs[name] = yaml.safe_load(p.read_text(encoding="utf-8"))
        else:
            rep.add("SKIP", f"{path} not found")
    for name, cfg in cfgs.items():
        expect = dict(EXPECT_COMMON, **(EXPECT_HYBRID if name == "hybrid" else {}))
        bad = []
        for path, want in expect.items():
            got = dig(cfg, path)
            if got is MISSING or got != want:
                bad.append(f"{'.'.join(path)}={'<missing>' if got is MISSING else got!r} (expected {want!r})")
        rep.add("WARN" if bad else "PASS", f"{name}: " + ("; ".join(bad) if bad else f"{len(expect)} keys at their full-run values"))
        info = ", ".join(f"{'.'.join(k[1:])}={'-' if dig(cfg, k) is MISSING else dig(cfg, k)}" for k in INFO_KEYS)
        print(f"         {name} (info): {info}")
    if len(cfgs) == 2:
        diff = sorted(k for k in set(cfgs["base"]) | set(cfgs["hybrid"])
                      if k != "loss" and cfgs["base"].get(k) != cfgs["hybrid"].get(k))
        rep.add("WARN" if diff else "PASS", "base vs hybrid differ outside `loss` in: " + ", ".join(diff) if diff
                else "base and hybrid differ only in the `loss` block")

    bpath = Path(args.benchmark)
    if not bpath.exists():
        rep.add("SKIP", f"{args.benchmark} not found")
        return
    bench = yaml.safe_load(bpath.read_text(encoding="utf-8"))["benchmark"]
    spec = importlib.util.spec_from_file_location("rb_preflight", REPO_ROOT / "scripts" / "run_benchmark.py")
    rb = importlib.util.module_from_spec(spec)
    sys.modules["rb_preflight"] = rb                        # dataclasses need the module registered
    spec.loader.exec_module(rb)
    plan = rb.build_plan(bench)
    main_runs = [s for s in plan if not s.oracle]
    first_ok = all(not s.oracle for s in plan[:len(main_runs)])
    rep.add("PASS" if (len(plan), len(main_runs)) == (90, 72) and first_ok else "FAIL",
            f"plan = {len(plan)} runs ({len(main_runs)} method + {len(plan) - len(main_runs)} oracle); "
            f"method runs first: {first_ok}")
    rep.add("PASS" if sorted(bench["seeds"]) == [42, 43, 44] else "WARN", f"seeds = {bench['seeds']}")
    mh = bench.get("max_hours")
    rep.add("WARN" if (mh is not None and mh > 12) else "PASS",
            f"benchmark.max_hours = {mh} (per invocation; pass --max_hours <session cap> on the CLI)")
    for name in bench["delays"]:
        dpath = REPO_ROOT / "configs" / f"{name}.yaml"
        if not dpath.exists():
            rep.add("FAIL", f"{dpath.name} missing")
            continue
        d = yaml.safe_load(dpath.read_text(encoding="utf-8"))["delay"]
        s, k, cap = d["params"]["scale_days"], d["params"]["shape"], d["params"]["cap_days"]
        f_cap = 1.0 - math.exp(-((cap / s) ** k))
        median = s * (-math.log(1.0 - 0.5 * f_cap)) ** (1.0 / k)
        lam = (bench.get("lam_by_delay") or {}).get(name)
        lam_ok = lam is not None and abs(lam - math.log(2.0) / median) <= 0.02 * math.log(2.0) / median
        w_ok = d["maturity_window_days"] >= cap
        rep.add("PASS" if lam_ok and w_ok else "FAIL",
                f"{name}: lam={lam} (ln2/median={math.log(2.0) / median:.4f}) W={d['maturity_window_days']} >= cap={cap}: {w_ok}")


# --------------------------------------------------------------------------- #
PARTS = {"1": part1, "2": part2, "3": part3, "4": part4, "5": part5}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pre-flight checks, one part at a time.")
    ap.add_argument("--part", nargs="+", default=["all"], choices=["all", *PARTS])
    ap.add_argument("--processed_dir", default=None, help="e.g. data/processed/delay_high (parts 2 and 4)")
    ap.add_argument("--runs_dir", default=None, help="folder of run JSONs, e.g. results/runs (part 3)")
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--hybrid", default="configs/base_hybrid.yaml")
    ap.add_argument("--benchmark", default="configs/benchmark.yaml")
    ap.add_argument("--priors", type=float, nargs="+", default=[0.0338, 0.0139],
                    help="dummy-data priors: 0.0338 = V1 estimate, 0.0139 = true rate in U (delay_high)")
    ap.add_argument("--mode", choices=["clamp", "nnpu"], default="nnpu")
    ap.add_argument("--lambda_pu", type=float, default=0.3)
    ap.add_argument("--max_weight", type=float, default=10.0)
    ap.add_argument("--n_batches", type=int, default=10)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args(argv)

    rep = Report()
    for p in (list(PARTS) if "all" in args.part else args.part):
        try:
            PARTS[p](rep, args)
        except Exception as exc:  # noqa: BLE001 - one broken part must not hide the others
            rep.add("FAIL", f"part {p} crashed: {type(exc).__name__}: {exc}")
        print()
    print("SUMMARY: " + " | ".join(f"{k}={v}" for k, v in rep.count.items()))
    return 1 if rep.count["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())