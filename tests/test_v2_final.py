"""V2 final regression tests: validation-only thresholding and variant-aware analysis."""
from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")
pytest.importorskip("torch_geometric")

from scripts.run_benchmark import build_plan  # noqa: E402
from src.eval import analyzer as A  # noqa: E402
import src.train.trainer as trainer_module  # noqa: E402
from src.train.trainer import Trainer  # noqa: E402


def test_test_recall_uses_validation_threshold_and_never_fits_test_threshold(monkeypatch):
    """A test-set-derived threshold would return a different recall on this deliberately chosen example."""
    trainer = Trainer.__new__(Trainer)
    trainer.model = object()
    trainer.device = torch.device("cpu")
    trainer.cfg = SimpleNamespace(review_cost=0.0, fpr=0.01)

    test_output = {
        "y_true": [1, 0, 1, 0],
        "scores": [0.6, 0.7, 0.4, 0.1],
        "amount": [100.0, 1.0, 50.0, 1.0],
    }

    def fake_predict(model, dataset, device):
        return test_output

    def forbidden_test_threshold(*args, **kwargs):
        pytest.fail("Test evaluation must not call summarize_scores / fit a threshold on test labels.")

    monkeypatch.setattr(trainer_module, "predict", fake_predict)
    monkeypatch.setattr(trainer_module.M, "summarize_scores", forbidden_test_threshold)

    validation_threshold = 0.5
    result = trainer.evaluate_test(object(), validation_threshold)

    # Applying the validation threshold flags scores 0.6 and 0.7: one of two frauds.
    assert result["val_threshold_applied"] == pytest.approx(validation_threshold)
    assert result["op_threshold"] == pytest.approx(validation_threshold)
    assert result["recall_at_fpr"] == pytest.approx(0.5)
    assert result["op_recall"] == pytest.approx(0.5)
    assert result["op_fpr"] == pytest.approx(0.5)
    assert "threshold_at_fpr" not in result


def _row(run_id, loss, variant, seed, pr_auc, efl):
    return {
        "run_id": run_id,
        "status": "ok",
        "delay": "delay_high",
        "loss": loss,
        "variant": variant or "",
        "backbone": "sage",
        "use_temporal": "False",
        "lp_features": "False",
        "seed": str(seed),
        "test_pr_auc": str(pr_auc),
        "test_roc_auc": "0.9",
        "test_recall_at_fpr": "0.25",
        "test_efl_ratio": str(efl),
        "prereg_status": "unfrozen",
        "cfg_hash": "v2-test",
        "prior_override": "",
    }


def test_analyzer_keeps_v2_variants_separate(tmp_path):
    rows = []
    for i, seed in enumerate((42, 43, 44)):
        rows.append(_row(f"a0_{seed}", "naive_bce", None, seed, 0.30 + i * 0.01, 0.70 - i * 0.01))
        rows.append(_row(f"light_{seed}", "tw_nndf", "v2_light", seed, 0.40 + i * 0.01, 0.60 - i * 0.01))
        rows.append(_row(f"imb_{seed}", "tw_nndf", "v2_imbalance", seed, 0.45 + i * 0.01, 0.55 - i * 0.01))

    parsed, warnings = A.clean_rows(rows)
    values = A.collect_values(parsed)
    variant_cells = [c for c in values if c.loss == "tw_nndf"]
    assert len(variant_cells) == 2
    assert {c.variant for c in variant_cells} == {"v2_light", "v2_imbalance"}
    assert all(len(values[c]["test_pr_auc"]) == 3 for c in variant_cells)

    result = A.run_analysis(rows, tmp_path, plots=False)
    with (Path(tmp_path) / "summary.csv").open(newline="", encoding="utf-8") as f:
        summary = list(csv.DictReader(f))
    tw_rows = [r for r in summary if r["loss"] == "tw_nndf" and r["metric"] == "test_pr_auc"]
    assert len(tw_rows) == 2
    assert {r["variant"] for r in tw_rows} == {"v2_light", "v2_imbalance"}
    diag_variants = {r["variant"] for r in result["nndf"]}
    assert diag_variants == {"v2_light", "v2_imbalance"}


def test_v2_probe_expands_to_exactly_15_non_temporal_runs():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "benchmark_v2_probe.yaml").read_text(encoding="utf-8"))
    plan = build_plan(config["benchmark"])

    assert len(plan) == 15
    assert len({spec.run_id for spec in plan}) == 15
    assert all(spec.delay == "delay_high" for spec in plan)
    assert all(spec.backbone == "sage" for spec in plan)
    assert all(spec.use_temporal is False for spec in plan)

    baselines = [spec for spec in plan if spec.variant is None]
    variants = [spec for spec in plan if spec.variant is not None]
    assert len(baselines) == 6
    assert len(variants) == 9
    assert {spec.loss for spec in baselines} == {"naive_bce", "maturity_mask"}
    assert {spec.variant for spec in variants} == {"v2_light", "v2_imbalance", "v2_conf"}
    assert {spec.seed for spec in plan} == {42, 43, 44}