# tests/test_headroom.py
import pytest
pytest.importorskip("scipy")
from src.eval.headroom import has_headroom, headroom_table, main, welch_diff_ci


def test_welch_known_answer_equal_halves():
    # equal SEs, n=3 -> Welch df = 4, t_{.975,4} = 2.7764; half_diff = sqrt(2)*0.1*2.7764/4.3027
    r = welch_diff_ci(0.5, 0.1, 3, 0.3, 0.1, 3)
    assert r["diff"] == pytest.approx(-0.2) and r["df"] == pytest.approx(4.0, abs=1e-6)
    assert (r["hi"] - r["lo"]) / 2 == pytest.approx(0.09126, abs=5e-4)


def test_identical_groups_include_zero_and_sign_flips():
    assert not welch_diff_ci(0.4, 0.05, 3, 0.4, 0.05, 3)["excludes_zero"]
    a, b = welch_diff_ci(0.3, 0.02, 3, 0.5, 0.02, 3), welch_diff_ci(0.5, 0.02, 3, 0.3, 0.02, 3)
    assert a["excludes_zero"] and a["diff"] == pytest.approx(-b["diff"])


def test_real_v1_numbers_qualitative(tmp_path):
    rows = [  # copied from summary.csv (test_pr_auc)
        ("delay_high", "naive_bce", 0.340249667, 0.135926589), ("delay_high", "oracle", 0.411797595, 0.031264520),
        ("delay_low", "naive_bce", 0.411928949, 0.039570596), ("delay_low", "tw_nndf", 0.219110809, 0.080324570),
    ]
    p = tmp_path / "summary.csv"
    p.write_text("delay,method,metric,n,mean,half\n" +
                 "\n".join(f"{d},{m},test_pr_auc,3,{mu},{h}" for d, m, mu, h in rows) + "\n")
    from src.eval.headroom import load_summary
    t = {(r["delay"], r["method"]): r for r in headroom_table(load_summary(str(p)))}
    assert not t[("delay_high", "oracle")]["excludes_zero"] and not has_headroom(t[("delay_high", "oracle")])
    assert t[("delay_low", "tw_nndf")]["excludes_zero"] and t[("delay_low", "tw_nndf")]["hi"] < 0
    assert main(["--summary", str(p)]) == 0