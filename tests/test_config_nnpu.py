# =====================================================================
# FILE: tests/test_config_nnpu.py
# V1.5: base.yaml and base_nnpu.yaml now carry the SAME hybrid `loss:` block, so the PU-core tests build
#       TimeWindowNNDFLoss from the PU-core keys only (unlabeled_statuses, mode, beta, gamma) and run on both files.
#       Run: pytest -q tests/test_config_nnpu.py tests/test_tw_nndf_ascent.py
# =====================================================================
from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.graph.builder import PNU_CODES  # noqa: E402
from src.losses import TimeWindowNNDFLoss  # noqa: E402
from src.train.trainer import TrainConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P, U = PNU_CODES["P"], PNU_CODES["U"]
CONFIGS = ["base.yaml", "base_nnpu.yaml"]
_PU_CORE_KEYS = ("unlabeled_statuses", "mode", "beta", "gamma")   # constructor keys of the PU core


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def _loss_from_yaml(name: str) -> TimeWindowNNDFLoss:
    params = {k: v for k, v in TrainConfig.from_cfg(_load(name), loss="tw_nndf").loss_params.items()
              if k in _PU_CORE_KEYS}
    if "unlabeled_statuses" in params:
        params["unlabeled_statuses"] = tuple(params["unlabeled_statuses"])
    return TimeWindowNNDFLoss(prior=0.0338, **params)


@pytest.mark.parametrize("name", CONFIGS)
def test_yaml_mode_reaches_the_loss(name):
    fn = _loss_from_yaml(name)
    assert fn.mode == "nnpu" and fn.beta == 0.0 and fn.gamma == 1.0


@pytest.mark.parametrize("name", CONFIGS)
def test_ascent_branch_fires_with_config_loss(name):
    fn = _loss_from_yaml(name)
    n_p, n_u = 50, 500
    preds = torch.cat([torch.full((n_p,), 4.0), torch.full((n_u,), -4.0)]).requires_grad_(True)
    pnu = torch.cat([torch.full((n_p,), P), torch.full((n_u,), U)]).long()
    loss = fn(preds, (pnu == P).long(), pnu, age_days=torch.zeros_like(preds))
    loss.backward()
    assert float(fn.last_risks["ascent"]) == 1.0
    assert float(preds.grad[n_p:].abs().max()) > 0.0       # U rows are NOT dead (clamp mode would give 0)


def test_run_benchmark_has_no_duplicate_definitions():
    src = (ROOT / "scripts" / "run_benchmark.py").read_text(encoding="utf-8")
    assert src.count("def apply_prior_override(") == 1


def test_base_nnpu_is_base_plus_loss_overlay():
    a, b = _load("base.yaml"), _load("base_nnpu.yaml")
    a.pop("loss")
    b_loss = b.pop("loss")
    assert a == b, "base_nnpu.yaml drifted from base.yaml outside `loss:`"
    t = b_loss["tw_nndf"]
    assert t["variant"] == "hybrid" and t["mode"] == "nnpu" and t["lambda_pu"] == 0.3


@pytest.mark.parametrize("name", CONFIGS)
def test_yaml_reaches_the_trainer_config(name):
    p = TrainConfig.from_cfg(_load(name), loss="tw_nndf").loss_params
    assert p["lambda_pu"] == 0.3 and p["calibration"]["enabled"] is True