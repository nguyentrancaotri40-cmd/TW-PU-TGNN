"""Round-7 tests: benchmark plan, resume, failure policy, config guard, time budget (fake run_fn, no training)."""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_benchmark", ROOT / "scripts" / "run_benchmark.py")
RB = importlib.util.module_from_spec(_spec)
sys.modules["run_benchmark"] = RB
_spec.loader.exec_module(RB)

LOSSES4 = ["naive_bce", "maturity_mask", "delay_ipw", "tw_nndf"]
BENCH = {
    "seeds": [1, 2, 3], "delays": ["delay_low", "delay_high"], "backbone": "sage",
    "lam_by_delay": {"delay_low": 0.2, "delay_high": 0.05},
    "groups": {
        "oracle": {"oracle": True},
        "main": {"losses": LOSSES4, "lp": False, "use_temporal": True},
        "lp_naive": {"enabled": False, "losses": ["naive_bce"], "lp": True, "use_temporal": True},
    },
}
BASE_CFG = {"sampler": {}, "model": {}, "train": {"epochs": 3}, "eval": {}, "loss": {"delay_ipw": {"lam": 0.23}}, "features": {}}


def make_run_fn(calls, fail_on=None, clock=None, cost=0.0):
    def run_fn(cfg, pdir, **kw):
        calls.append((pdir.name, kw["loss"], kw["oracle"], kw["seed"], cfg["loss"].get("delay_ipw")))
        if fail_on is not None and (pdir.name, kw["loss"], kw["seed"]) == fail_on:
            raise RuntimeError("boom")
        if clock is not None:
            clock["t"] += cost
        label = "oracle" if kw["oracle"] else kw["loss"]
        return {"row": {"run_id": "x", "delay": pdir.name, "loss": label, "oracle": kw["oracle"],
                        "backbone": kw["backbone"], "use_temporal": kw["use_temporal"], "lp_features": kw["lp"],
                        "seed": kw["seed"], "best_epoch": 1, "test_pr_auc": 0.5},
                "details": {"info": {"lam": 0.1}, "fit": {"history": [{"epoch": 1.0}]}, "test": {}, "cfg": {}}}
    return run_fn


def paths(tmp: Path) -> dict:
    return dict(processed_root=tmp / "proc", results_path=tmp / "r.csv", details_dir=tmp / "runs")


def test_plan_counts_and_order():
    plan = RB.build_plan(BENCH)
    assert len(plan) == 6 + 24 and len({s.run_id for s in plan}) == len(plan)
    assert all(s.oracle for s in plan[:6])                                   # oracle first
    assert [(s.delay, s.loss, s.seed) for s in plan[6:9]] == [("delay_low", "naive_bce", i) for i in (1, 2, 3)]
    assert plan[0].run_id == "delay_low_oracle_sage_gru_nolp_s1"
    b2 = copy.deepcopy(BENCH)
    b2["groups"]["lp_naive"]["enabled"] = True
    assert len(RB.build_plan(b2)) == 36


def test_duplicate_run_id_raises():
    b = copy.deepcopy(BENCH)
    b["groups"]["again"] = {"losses": ["naive_bce"], "lp": False, "use_temporal": True}
    with pytest.raises(ValueError, match="duplicate"):
        RB.build_plan(b)


def test_prior_group():
    b = copy.deepcopy(BENCH)
    b["groups"] = {"p": {"losses": ["tw_nndf"], "priors_by_delay": {"delay_low": [0.0139]}}}
    plan = RB.build_plan(b)
    assert len(plan) == 3 and all(s.delay == "delay_low" and s.run_id.endswith("_pi0.0139") for s in plan)
    b["groups"] = {"p": {"losses": ["naive_bce"], "priors_by_delay": {"delay_low": [0.0139]}}}
    with pytest.raises(ValueError):
        RB.build_plan(b)


def test_missing_lam_fails_before_any_run(tmp_path):
    b = copy.deepcopy(BENCH)
    b["lam_by_delay"]["delay_low"] = None
    calls: list = []
    with pytest.raises(KeyError, match="lam_by_delay"):
        RB.execute(RB.build_plan(b), BASE_CFG, b, run_fn=make_run_fn(calls), **paths(tmp_path))
    assert calls == []
    cfg = RB.resolve_cfg(BASE_CFG, BENCH, [s for s in RB.build_plan(BENCH) if s.loss == "delay_ipw"][0], False)
    assert cfg["loss"]["delay_ipw"] == {"lam": 0.2}                          # base.yaml's 0.23 is ignored


def test_resume_skips_done_and_failed_policy(tmp_path):
    plan = RB.build_plan(BENCH)
    p = paths(tmp_path)
    out = RB.execute(plan, BASE_CFG, BENCH, run_fn=make_run_fn([], fail_on=("delay_low", "maturity_mask", 1)), **p)
    assert (out["n_ok"], out["n_failed"]) == (len(plan) - 1, 1)
    rows = RB.read_rows(p["results_path"])
    assert sum(r["status"] == "ok" for r in rows) == len(plan) - 1 and rows[0]["prereg_status"] in ("unfrozen", "")

    calls2: list = []
    out2 = RB.execute(plan, BASE_CFG, BENCH, run_fn=make_run_fn(calls2), **p)       # nothing to do, failure not retried
    assert calls2 == [] and out2["n_skipped_failed"] == 1

    calls3: list = []
    RB.execute(plan, BASE_CFG, BENCH, run_fn=make_run_fn(calls3), retry_failed=True, **p)
    assert len(calls3) == 1 and calls3[0][1] == "maturity_mask"
    ok_ids = {r["run_id"] for r in RB.read_rows(p["results_path"]) if r["status"] == "ok"}
    assert ok_ids == {s.run_id for s in plan}


def test_config_hash_guard(tmp_path):
    plan = RB.build_plan(BENCH)[:2]
    p = paths(tmp_path)
    RB.execute(plan, BASE_CFG, BENCH, run_fn=make_run_fn([]), **p)
    changed = copy.deepcopy(BASE_CFG)
    changed["train"]["epochs"] = 5
    with pytest.raises(RuntimeError, match="config hash"):
        RB.execute(RB.build_plan(BENCH)[:4], changed, BENCH, run_fn=make_run_fn([]), **p)
    RB.execute(RB.build_plan(BENCH)[:4], changed, BENCH, run_fn=make_run_fn([]), allow_config_change=True, **p)


def test_time_budget_stops_before_a_run_that_does_not_fit(tmp_path):
    clock = {"t": 0.0}
    calls: list = []
    plan = RB.build_plan(BENCH)
    out = RB.execute(plan, BASE_CFG, BENCH, run_fn=make_run_fn(calls, clock=clock, cost=3600.0),
                     clock=lambda: clock["t"], max_hours=2.5, **paths(tmp_path))
    assert len(calls) == 2 and out["stopped_for_budget"] and out["n_remaining"] == len(plan) - 2


def test_csv_header_guard(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        RB.read_rows(bad)
    with pytest.raises(RuntimeError):
        RB.append_row(bad, {"run_id": "x"})