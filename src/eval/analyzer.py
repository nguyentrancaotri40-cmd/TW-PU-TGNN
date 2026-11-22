"""Aggregate benchmark_results.csv: 95% CI over seeds, verdict vs A0, Recovery Ratio, LaTeX tables and figures.

Rules implemented (pre-registered, see README):
* a method counts as an improvement only if its seed-mean lies OUTSIDE the 95% CI of Naive A0 on the favourable
  side, for PR-AUC (higher is better) or EFL ratio (lower is better);
* Recovery Ratio = (S_method - S_naive) / (S_oracle - S_naive) (sign-flipped for EFL), reported only if the oracle is
  outside A0's CI on the favourable side; otherwise "n/r" and the reason is kept;
* CI = Student-t over seeds (src.eval.metrics.ci95). With 3 seeds t = 4.303, so CIs are wide: "within CI" is common.
Only rows with status == "ok" are used; a duplicate run_id keeps the LAST row.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.eval import metrics as M

NAN = float("nan")


class Cell(NamedTuple):
    delay: str
    loss: str          # naive_bce | maturity_mask | delay_ipw | tw_nndf | oracle
    backbone: str
    use_temporal: bool
    lp: bool


METRICS: dict[str, tuple[str, bool]] = {   # column -> (header, higher_is_better)
    "test_pr_auc": ("PR-AUC", True),
    "test_roc_auc": ("ROC-AUC", True),
    "test_recall_at_fpr": ("Rec@FPR", True),
    "test_efl_ratio": ("EFL ratio", False),
}
PRIMARY = ("test_pr_auc", "test_efl_ratio")
DIAG_COLS = ("test_fpr_at_valthr", "test_recall_at_valthr", "val_pr_auc", "best_epoch", "epochs_run",
             "train_seconds", "peak_vram_alloc_mb")
NNDF_COLS = ("nndf_prior", "nndf_prior_estimated", "diag_true_fraud_rate_in_U", "nndf_clamped_best",
             "nndf_clamped_mean", "nndf_neg_risk_raw_best", "nndf_pos_risk_best", "n_empty_batches_total")
LABELS = {"naive_bce": "A0 Naive BCE", "maturity_mask": "Maturity Masking", "delay_ipw": "Delay-IPW",
          "tw_nndf": "TW-nnDF", "oracle": r"Oracle ($y_{\mathrm{true}}$)"}
_ORDER = {"naive_bce": 0, "maturity_mask": 1, "delay_ipw": 2, "tw_nndf": 3, "oracle": 9}


# --------------------------------------------------------------------------- #
# Loading and cleaning
# --------------------------------------------------------------------------- #
def _f(x: Any) -> float:
    if x is None or x == "":
        return NAN
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def _b(x: Any) -> bool:
    return x if isinstance(x, bool) else str(x).strip().lower() in {"true", "1", "yes"}


def load_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clean_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep status == ok, dedup by run_id (last wins), parse types. Returns (parsed rows, warnings)."""
    warns: list[str] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    n_bad = 0
    for r in rows:
        if str(r.get("status", "")).strip() != "ok":
            n_bad += 1
            continue
        by_id[str(r["run_id"])] = r
    if n_bad:
        warns.append(f"{n_bad} row(s) with status != ok were ignored")
    hashes = {str(r.get("cfg_hash", "")) for r in by_id.values()} - {""}
    if len(hashes) > 1:
        warns.append(f"rows come from {len(hashes)} different config hashes {sorted(hashes)}: protocols are mixed")
    num_cols = list(METRICS) + list(DIAG_COLS) + list(NNDF_COLS)
    parsed: list[dict[str, Any]] = []
    for r in by_id.values():
        p: dict[str, Any] = {
            "run_id": str(r["run_id"]), "delay": str(r["delay"]), "loss": str(r["loss"]),
            "backbone": str(r["backbone"]), "use_temporal": _b(r["use_temporal"]), "lp": _b(r["lp_features"]),
            "seed": int(float(r["seed"])), "prereg_status": str(r.get("prereg_status", "")),
            "prior_override": r.get("prior_override", ""),
        }
        p.update({c: _f(r.get(c)) for c in num_cols})
        parsed.append(p)
    parsed.sort(key=lambda p: (p["delay"], p["loss"], p["seed"]))
    return parsed, warns


def collect_values(parsed: Sequence[Mapping[str, Any]]) -> dict[Cell, dict[str, list[float]]]:
    """Per cell, per metric: the finite values over seeds (sorted by seed). Prior-override runs are excluded."""
    out: dict[Cell, dict[str, list[float]]] = {}
    for p in sorted(parsed, key=lambda q: q["seed"]):
        if str(p.get("prior_override", "")) not in ("", "None"):
            continue
        cell = Cell(p["delay"], p["loss"], p["backbone"], p["use_temporal"], p["lp"])
        d = out.setdefault(cell, {m: [] for m in METRICS})
        for m in METRICS:
            if math.isfinite(p[m]):
                d[m].append(float(p[m]))
    return out


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _kind(c: Cell) -> str:
    return "oracle" if c.loss == "oracle" else "a0" if (c.loss == "naive_bce" and not c.lp) else "method"


def _a0(c: Cell) -> Cell:
    return Cell(c.delay, "naive_bce", c.backbone, c.use_temporal, False)


def _oracle(c: Cell) -> Cell:
    return Cell(c.delay, "oracle", c.backbone, c.use_temporal, False)


def _sort_key(c: Cell) -> tuple:
    return (c.delay, c.backbone, _ORDER.get(c.loss, 5), not c.use_temporal, c.lp)


def summarise(values: Mapping[Cell, Mapping[str, Sequence[float]]]) -> dict[tuple[Cell, str], dict[str, float]]:
    out: dict[tuple[Cell, str], dict[str, float]] = {}
    for cell, d in values.items():
        for m, v in d.items():
            if not v:
                continue
            if len(v) >= 2:
                mean, lo, hi = M.ci95(v)
            else:
                mean, lo, hi = float(v[0]), NAN, NAN
            out[(cell, m)] = {"n": len(v), "mean": mean, "ci_low": lo, "ci_high": hi, "half": (hi - mean) if len(v) >= 2 else NAN}
    return out


def compare_to_naive(values: Mapping[Cell, Mapping[str, Sequence[float]]]) -> dict[tuple[Cell, str], dict[str, Any]]:
    """Verdict of every method cell against the 95% CI of its A0 (same delay / backbone / temporal flag)."""
    out: dict[tuple[Cell, str], dict[str, Any]] = {}
    for cell, d in values.items():
        if _kind(cell) != "method":
            continue
        a0 = values.get(_a0(cell))
        for m in PRIMARY:
            higher = METRICS[m][1]
            v = d.get(m, [])
            if not v:
                continue
            if a0 is None or len(a0.get(m, [])) < 2:
                out[(cell, m)] = {"verdict": "n/a", "position": "n/a", "naive_mean": NAN, "naive_ci_low": NAN, "naive_ci_high": NAN}
                continue
            mean = float(np.mean(v))
            nm, lo, hi = M.ci95(a0[m])
            pos = M.ci_position(mean, a0[m])
            verdict = "within_CI" if pos == "inside" else ("improves" if (pos == "above") == higher else "worse")
            out[(cell, m)] = {"verdict": verdict, "position": pos, "naive_mean": nm, "naive_ci_low": lo, "naive_ci_high": hi}
    return out


def recovery_table(values: Mapping[Cell, Mapping[str, Sequence[float]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in sorted(values, key=_sort_key):
        if _kind(cell) != "method":
            continue
        a0, orc = values.get(_a0(cell)), values.get(_oracle(cell))
        for m in PRIMARY:
            higher = METRICS[m][1]
            rec: dict[str, Any] = {"delay": cell.delay, "loss": cell.loss, "backbone": cell.backbone,
                                   "use_temporal": cell.use_temporal, "lp": cell.lp, "metric": m, "reported": False,
                                   "value": NAN, "reason": "", "method_mean": NAN, "naive_mean": NAN,
                                   "oracle_mean": NAN, "naive_ci_low": NAN, "naive_ci_high": NAN,
                                   "n_method": len(values[cell].get(m, [])),
                                   "n_naive": 0 if a0 is None else len(a0.get(m, [])),
                                   "n_oracle": 0 if orc is None else len(orc.get(m, []))}
            mv = values[cell].get(m, [])
            if not mv:
                rec["reason"] = "no values"
            elif a0 is None or len(a0.get(m, [])) < 2:
                rec["reason"] = "A0 has fewer than 2 seeds (no CI)"
            elif orc is None or not orc.get(m):
                rec["reason"] = "no oracle run for this delay"
            else:
                r = M.recovery_ratio(mv, a0[m], orc[m], higher)
                rec.update(reported=r.reported, value=NAN if r.value is None else float(r.value), reason=r.reason,
                           method_mean=r.method_mean, naive_mean=r.naive_mean, oracle_mean=r.oracle_mean,
                           naive_ci_low=r.naive_ci[0], naive_ci_high=r.naive_ci[1])
            rows.append(rec)
    return rows


def nndf_diagnostics(parsed: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Seed-mean of the tw_nndf clamp / prior diagnostics per (delay, backbone, temporal, lp)."""
    groups: dict[tuple, list[Mapping[str, Any]]] = {}
    for p in parsed:
        if p["loss"] == "tw_nndf" and str(p.get("prior_override", "")) in ("", "None"):
            groups.setdefault((p["delay"], p["backbone"], p["use_temporal"], p["lp"]), []).append(p)
    out = []
    for (delay, bb, tmp, lp), ps in sorted(groups.items()):
        row: dict[str, Any] = {"delay": delay, "backbone": bb, "use_temporal": tmp, "lp": lp, "n_seeds": len(ps)}
        for c in NNDF_COLS:
            vals = [q[c] for q in ps if math.isfinite(q[c])]
            row[c] = float(np.mean(vals)) if vals else NAN
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
def _esc(s: str) -> str:
    for a, b in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def method_label(c: Cell, latex: bool = True) -> str:
    base = LABELS.get(c.loss, _esc(c.loss)) if latex else c.loss
    if c.lp:
        base += " + LP feat."
    if not c.use_temporal:
        base += " (no GRU)"
    if c.backbone != "sage":
        base += f" [{c.backbone}]"
    return base


def _fmt(s: Mapping[str, float]) -> str:
    if int(s["n"]) >= 2:
        return f"{s['mean']:.3f} $\\pm$ {s['half']:.3f}"
    return f"{s['mean']:.3f} (n={int(s['n'])})"


def main_table_tex(delay: str, summ: Mapping[tuple[Cell, str], Mapping[str, float]],
                   comp: Mapping[tuple[Cell, str], Mapping[str, Any]], prereg: str) -> str:
    cells = sorted({c for (c, _m) in summ if c.delay == delay}, key=_sort_key)
    metrics = list(METRICS)
    best: dict[str, Cell] = {}
    for m in metrics:
        cand = [(summ[(c, m)]["mean"], c) for c in cells
                if c.loss != "oracle" and (c, m) in summ and math.isfinite(summ[(c, m)]["mean"])]
        if cand:
            best[m] = (max if METRICS[m][1] else min)(cand, key=lambda t: t[0])[1]
    head = " & ".join(["Method"] + [METRICS[m][0] + (r" $\uparrow$" if METRICS[m][1] else r" $\downarrow$") for m in metrics])
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Test results, " + _esc(delay) + r". Mean $\pm$ 95\% CI half-width (Student-$t$ over seeds). "
        r"$^{\dagger}$: seed-mean outside the 95\% CI of A0 on the favourable side (PR-AUC, EFL ratio). "
        r"Bold: best non-oracle mean. Pre-registration status: " + _esc(prereg) + r".}",
        r"\label{tab:main_" + delay + "}",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}", r"\toprule", head + r" \\", r"\midrule",
    ]
    seen_oracle = False
    for c in cells:
        if c.loss == "oracle" and not seen_oracle:
            lines.append(r"\midrule")
            seen_oracle = True
        cols = []
        for m in metrics:
            s = summ.get((c, m))
            if s is None:
                cols.append("--")
                continue
            txt = _fmt(s)
            if m in PRIMARY and comp.get((c, m), {}).get("verdict") == "improves":
                txt += r"$^{\dagger}$"
            if best.get(m) == c:
                txt = r"\textbf{" + txt + "}"
            cols.append(txt)
        lines.append(method_label(c) + " & " + " & ".join(cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def recovery_table_tex(rec: Sequence[Mapping[str, Any]], prereg: str) -> str:
    keyed: dict[tuple, dict[str, Mapping[str, Any]]] = {}
    for r in rec:
        keyed.setdefault((r["delay"], r["backbone"], r["loss"], r["use_temporal"], r["lp"]), {})[r["metric"]] = r

    def cell_txt(r: Mapping[str, Any] | None) -> str:
        return "--" if r is None else (f"{r['value']:.2f}" if r["reported"] else "n/r")

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Recovery Ratio $=(S_{\mathrm{method}}-S_{\mathrm{A0}})/(S_{\mathrm{oracle}}-S_{\mathrm{A0}})$, "
        r"sign-flipped for EFL. n/r: the oracle--A0 gap lies inside the 95\% CI of A0, ratio not reported. "
        r"Values are not clipped (can be $<0$ or $>1$). Pre-registration status: " + _esc(prereg) + r".}",
        r"\label{tab:recovery}", r"\begin{tabular}{llcc}", r"\toprule",
        r"Delay & Method & RR PR-AUC & RR EFL \\", r"\midrule",
    ]
    for (delay, bb, loss, tmp, lp), d in sorted(keyed.items(), key=lambda kv: _sort_key(Cell(*kv[0]))):
        c = Cell(delay, loss, bb, tmp, lp)
        lines.append(f"{_esc(delay)} & {method_label(c)} & {cell_txt(d.get('test_pr_auc'))} & {cell_txt(d.get('test_efl_ratio'))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Figures (matplotlib is optional)
# --------------------------------------------------------------------------- #
def plot_metric(summ: Mapping[tuple[Cell, str], Mapping[str, float]], metric: str, out_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    delays = sorted({c.delay for (c, m) in summ if m == metric})
    if not delays:
        return False
    fig, axes = plt.subplots(1, len(delays), figsize=(4.2 * len(delays), 3.4), squeeze=False)
    for ax, delay in zip(axes[0], delays):
        cells = sorted({c for (c, m) in summ if m == metric and c.delay == delay}, key=_sort_key, reverse=True)
        for y, c in enumerate(cells):
            s = summ[(c, metric)]
            color = {"a0": "#555555", "oracle": "#009E73", "method": "#0072B2"}[_kind(c)]   # Okabe-Ito
            half = s["half"] if math.isfinite(s["half"]) else 0.0
            ax.errorbar(s["mean"], y, xerr=half, fmt="o", color=color, capsize=3)
        a0 = [s for (c, m), s in summ.items() if m == metric and c.delay == delay and _kind(c) == "a0"]
        if a0 and math.isfinite(a0[0]["ci_low"]):
            ax.axvspan(a0[0]["ci_low"], a0[0]["ci_high"], color="#999999", alpha=0.2)
        ax.set_yticks(range(len(cells)))
        ax.set_yticklabels([method_label(c, latex=False) for c in cells], fontsize=8)
        ax.set_title(delay, fontsize=10)
        ax.set_xlabel(METRICS[metric][0] + (" (higher is better)" if METRICS[metric][1] else " (lower is better)"))
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(columns) if columns else (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (isinstance(v, float) and not math.isfinite(v)) else v) for k, v in r.items()})


def run_analysis(rows: Sequence[Mapping[str, Any]], out_dir: str | Path, *, plots: bool = True) -> dict[str, Any]:
    out = Path(out_dir)
    parsed, warns = clean_rows(rows)
    if not parsed:
        raise ValueError("No rows with status == ok.")
    prereg_values = sorted({p["prereg_status"] for p in parsed} - {""})
    prereg = "frozen" if prereg_values == ["frozen"] else "unfrozen (exploratory)"
    values = collect_values(parsed)
    if any(len(v) < 2 for d in values.values() for v in d.values() if v):
        warns.append("some cells have fewer than 2 seeds: no CI, no verdict")
    summ, comp, rec, diag = summarise(values), compare_to_naive(values), recovery_table(values), nndf_diagnostics(parsed)

    summary_rows = [{"delay": c.delay, "method": method_label(c, latex=False), "loss": c.loss, "backbone": c.backbone,
                     "use_temporal": c.use_temporal, "lp": c.lp, "metric": m, **s,
                     "verdict": comp.get((c, m), {}).get("verdict", "")} for (c, m), s in sorted(summ.items(), key=lambda kv: (_sort_key(kv[0][0]), kv[0][1]))]
    _write_csv(out / "summary.csv", summary_rows)
    _write_csv(out / "recovery.csv", rec)
    _write_csv(out / "nndf_diagnostics.csv", diag)
    files = ["summary.csv", "recovery.csv", "nndf_diagnostics.csv"]
    for delay in sorted({c.delay for (c, _m) in summ}):
        name = f"table_main_{delay}.tex"
        (out / name).write_text(main_table_tex(delay, summ, comp, prereg), encoding="utf-8")
        files.append(name)
    (out / "table_recovery.tex").write_text(recovery_table_tex(rec, prereg), encoding="utf-8")
    files.append("table_recovery.tex")
    figs = []
    if plots:
        for m in PRIMARY:
            if plot_metric(summ, m, out / f"fig_{m}.pdf"):
                figs.append(f"fig_{m}.pdf")
        if figs:
            (out / "figures.tex").write_text("\n".join(
                r"\begin{figure}[t]\centering\includegraphics[width=\linewidth]{" + f + r"}\end{figure}" for f in figs) + "\n", encoding="utf-8")
        else:
            warns.append("no figure written (matplotlib missing or no data)")
    return {"warnings": warns, "prereg": prereg, "files": files + figs, "n_runs": len(parsed),
            "summary": summ, "comparison": comp, "recovery": rec, "nndf": diag}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Analyse results/benchmark_results.csv")
    ap.add_argument("--results", default=str(REPO_ROOT / "results" / "benchmark_results.csv"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "analysis"))
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args(argv)
    try:
        res = run_analysis(load_rows(args.results), args.out, plots=not args.no_plots)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[analyzer] ERROR: {exc}")
        return 1
    print(f"[analyzer] pre-registration: {res['prereg'].upper()} | {res['n_runs']} ok runs")
    for w in res["warnings"]:
        print(f"[analyzer] WARNING: {w}")
    n_rep = sum(1 for r in res["recovery"] if r["reported"])
    print(f"[analyzer] Recovery Ratio reported for {n_rep}/{len(res['recovery'])} (method, metric) pairs")
    for r in res["recovery"]:
        if not r["reported"]:
            print(f"   n/r {r['delay']:<12} {r['loss']:<14} {r['metric']:<14} {r['reason']}")
    print(f"[analyzer] wrote {len(res['files'])} files to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())