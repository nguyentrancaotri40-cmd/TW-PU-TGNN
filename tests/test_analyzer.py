"""Round-7 tests: cleaning, verdicts, Recovery Ratio gate, LaTeX markers, end-to-end files (synthetic rows)."""
from __future__ import annotations

import math

import pytest

from src.eval import analyzer as A


def mk(delay, loss, seed, pr, efl, lp=False, status="ok"):
    return {"run_id": f"{delay}_{loss}_{'lp' if lp else 'nolp'}_s{seed}", "status": status, "delay": delay, "loss": loss,
            "backbone": "sage", "use_temporal": "True", "lp_features": str(lp), "seed": str(seed),
            "test_pr_auc": str(pr), "test_efl_ratio": str(efl), "test_roc_auc": "0.9", "test_recall_at_fpr": "0.5",
            "prereg_status": "unfrozen", "cfg_hash": "abc"}


def block(delay, loss, prs, efls, **kw):
    return [mk(delay, loss, 42 + i, p, e, **kw) for i, (p, e) in enumerate(zip(prs, efls))]


@pytest.fixture()
def rows():
    r = []
    r += block("delay_low", "naive_bce", [0.30, 0.31, 0.32], [0.60, 0.61, 0.62])
    r += block("delay_low", "oracle", [0.50, 0.51, 0.52], [0.40, 0.41, 0.42])
    r += block("delay_low", "tw_nndf", [0.40, 0.41, 0.42], [0.50, 0.51, 0.52])
    r += block("delay_low", "maturity_mask", [0.31, 0.32, 0.31], [0.60, 0.61, 0.60])
    r += block("delay_high", "naive_bce", [0.30, 0.31, 0.32], [0.60, 0.61, 0.62])
    r += block("delay_high", "oracle", [0.31, 0.32, 0.33], [0.59, 0.60, 0.61])   # weak
    r += block("delay_high", "tw_nndf", [0.40, 0.41, 0.42], [0.50, 0.51, 0.52])
    r += [mk("delay_low", "tw_nndf", 99, 0.0, 0.0, status="failed")]        # must be ignored
    return r


DL_TW = A.Cell("delay_low", "tw_nndf", "sage", True, False)
DL_MM = A.Cell("delay_low", "maturity_mask", "sage", True, False)


def test_clean_rows_drops_failed_and_dedups(rows):
    dup = mk("delay_low", "naive_bce", 42, 0.99, 0.10)                        # same run_id as an existing row
    parsed, warns = A.clean_rows(rows + [dup])
    assert len(parsed) == 21                                                  # 12 (delay_low) + 9 (delay_high), no failed
    assert any("status != ok" in w for w in warns)
    p = [q for q in parsed if q["run_id"] == "delay_low_naive_bce_nolp_s42"][0]
    assert p["test_pr_auc"] == pytest.approx(0.99)                            # the LAST row wins


def test_verdicts_against_a0_ci(rows):
    values = A.collect_values(A.clean_rows(rows)[0])
    comp = A.compare_to_naive(values)
    assert comp[(DL_TW, "test_pr_auc")]["verdict"] == "improves"
    assert comp[(DL_TW, "test_efl_ratio")]["verdict"] == "improves"           # lower is better: 0.51 < CI low 0.585
    assert comp[(DL_MM, "test_pr_auc")]["verdict"] == "within_CI"
    assert comp[(DL_MM, "test_efl_ratio")]["verdict"] == "within_CI"


def test_recovery_reported_and_not_reported(rows):
    rec = A.recovery_table(A.collect_values(A.clean_rows(rows)[0]))
    get = lambda d, l, m: [r for r in rec if (r["delay"], r["loss"], r["metric"]) == (d, l, m)][0]
    ok = get("delay_low", "tw_nndf", "test_pr_auc")
    assert ok["reported"] and ok["value"] == pytest.approx((0.41 - 0.31) / (0.51 - 0.31))
    weak = get("delay_high", "tw_nndf", "test_pr_auc")                        # oracle mean 0.32 is inside A0's CI
    assert (not weak["reported"]) and math.isnan(weak["value"]) and "CI" in weak["reason"]


def test_recovery_lower_is_better_for_efl(rows):
    rec = A.recovery_table(A.collect_values(A.clean_rows(rows)[0]))
    r = [x for x in rec if (x["delay"], x["loss"], x["metric"]) == ("delay_low", "tw_nndf", "test_efl_ratio")][0]
    assert r["reported"] and r["value"] == pytest.approx((0.61 - 0.51) / (0.61 - 0.41))


def test_latex_tables_contain_markers(rows):
    values = A.collect_values(A.clean_rows(rows)[0])
    summ, comp = A.summarise(values), A.compare_to_naive(values)
    tex = A.main_table_tex("delay_low", summ, comp, "unfrozen")
    assert r"\dagger" in tex and "TW-nnDF" in tex and "unfrozen" in tex and r"\textbf{" in tex
    assert r"\begin{tabular}" in tex and tex.count(r"\toprule") == 1
    rtex = A.recovery_table_tex(A.recovery_table(values), "unfrozen")
    assert "n/r" in rtex and "0.50" in rtex


def test_end_to_end_writes_files(rows, tmp_path):
    res = A.run_analysis(rows, tmp_path, plots=False)
    assert res["prereg"].startswith("unfrozen") and res["n_runs"] == 21
    for name in ("summary.csv", "recovery.csv", "nndf_diagnostics.csv", "table_recovery.tex",
                 "table_main_delay_low.tex", "table_main_delay_high.tex"):
        assert (tmp_path / name).exists()
    with pytest.raises(ValueError):
        A.run_analysis([mk("d", "naive_bce", 1, 0.1, 0.1, status="failed")], tmp_path, plots=False)