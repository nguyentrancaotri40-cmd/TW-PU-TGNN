# =====================================================================
# FILE: scripts/make_report.py
# TRẠNG THÁI: ĐÃ SỬA (đường dẫn mặc định: hết lặp "results/results" khi chạy local;
#                     trên Kaggle tìm cả <repo>/results và /kaggle/working/results, loại trùng)
# =====================================================================
#!/usr/bin/env python
"""Final report: merge benchmark CSVs, 300-dpi PNG figures (+PDF), LaTeX tables printed to stdout.

    python scripts/make_report.py                       # defaults: <repo>/results (+ /kaggle/working/results on Kaggle)
    python scripts/make_report.py --results results/benchmark_results_nnpu.csv --fig_dir figures

Pass --results explicitly when V1 (clamp) and V1.5 (nnpu) CSVs both exist: run_id of tw_nndf is the same in both,
and the analyzer keeps the LAST row per run_id, so a blind glob would let V1.5 silently replace V1.

All statistics come from src.eval.analyzer (same CI / verdict / Recovery-Ratio rules as the pre-registration).
Colour follows the METHOD (Okabe-Ito, colour-blind safe); no-GRU variants are hatched; individual seeds are shown.
"""
from __future__ import annotations

import argparse
import glob
import math
import sys
import zipfile
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Missing package '{exc.name}'. Install it with: pip install {exc.name}")

from src.eval import analyzer as A

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "savefig.dpi": 300})
COLORS = {"naive_bce": "#555555", "maturity_mask": "#0072B2", "delay_ipw": "#E69F00",
          "tw_nndf": "#D55E00", "oracle": "#009E73", "lp": "#CC79A7"}          # Okabe-Ito
MARKERS = {"naive_bce": "s", "maturity_mask": "o", "delay_ipw": "^", "tw_nndf": "D", "oracle": "*", "lp": "v"}
SHORT = {"naive_bce": "A0", "maturity_mask": "Maturity\nMask", "delay_ipw": "Delay-\nIPW", "tw_nndf": "TW-\nnnDF",
         "oracle": "Oracle"}
SHADES = ["#BBBBBB", "#666666", "#222222"]
KAGGLE_ROOT = Path("/kaggle/working")


def style(c: A.Cell) -> tuple[str, str, str, str]:
    """(label, colour, hatch, marker): colour follows the method, never its rank."""
    key = "lp" if (c.lp and c.loss == "naive_bce") else c.loss
    label = "A0\n+LP" if key == "lp" else SHORT.get(c.loss, c.loss)
    if not c.use_temporal:
        label += "\n(no GRU)"
    return label, COLORS.get(key, "#999999"), "" if c.use_temporal else "//", MARKERS.get(key, "o")


def _delays(names: set[str]) -> list[str]:
    return sorted(names, key=lambda d: (0 if "low" in d else 1 if "high" in d else 2, d))


def _half(s: dict[str, float]) -> float:
    return s["half"] if math.isfinite(s["half"]) else 0.0


def _save(fig: Any, path: Path, prereg: str) -> None:
    fig.text(0.995, 0.002, f"pre-registration: {prereg}", ha="right", va="bottom", fontsize=6.5, color="#666666")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def figure_bars(values: dict, summ: dict, comp: dict, metric: str, path: Path, prereg: str) -> None:
    header, higher = A.METRICS[metric]
    delays = _delays({c.delay for c in values})
    n_seeds = sorted({len(v[metric]) for v in values.values() if v.get(metric)})
    fig, axes = plt.subplots(1, len(delays), figsize=(4.6 * len(delays), 4.2), squeeze=False, sharey=True)
    for ax, delay in zip(axes[0], delays):
        cells = sorted([c for c, d in values.items() if c.delay == delay and d.get(metric)], key=A._sort_key)
        for x, c in enumerate(cells):
            label, color, hatch, _ = style(c)
            s = summ[(c, metric)]
            ax.bar(x, s["mean"], yerr=_half(s), color=color, alpha=0.85, hatch=hatch, edgecolor="black",
                   linewidth=0.6, capsize=3, zorder=2)
            ax.scatter([x] * len(values[c][metric]), values[c][metric], s=9, color="black", zorder=4)
            if comp.get((c, metric), {}).get("verdict") == "improves":
                ax.annotate("†", (x, s["mean"] + _half(s)), ha="center", va="bottom", fontsize=12)
        a0 = [c for c in cells if A._kind(c) == "a0" and c.use_temporal]
        if a0 and math.isfinite(summ[(a0[0], metric)]["ci_low"]):
            s0 = summ[(a0[0], metric)]
            ax.axhspan(s0["ci_low"], s0["ci_high"], color="#999999", alpha=0.2, zorder=0)
            ax.axhline(s0["mean"], color="#555555", ls="--", lw=0.8, zorder=1)
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([style(c)[0] for c in cells], fontsize=7.5)
        ax.set_title(delay.replace("_", " "))
        ax.margins(y=0.15)
        ax.grid(axis="y", alpha=0.3, zorder=0)
    axes[0][0].set_ylabel(f"Test {header} ({'higher' if higher else 'lower'} is better)")
    fig.suptitle(f"Test {header} by method (mean ± 95% CI, Student-t, n = {'/'.join(map(str, n_seeds))} seeds)", y=1.0)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.legend(handles=[Patch(facecolor="#999999", alpha=0.3, label="A0 95% CI"),
                        Line2D([], [], marker="o", color="black", ls="", ms=3, label="individual seeds"),
                        Line2D([], [], marker="$†$", color="black", ls="", ms=7, label="outside A0 CI, favourable side")],
               loc="lower center", ncol=3, frameon=False)
    _save(fig, path, prereg)


def figure_lines(values: dict, summ: dict, path: Path, prereg: str) -> None:
    delays = _delays({c.delay for c in values})
    xs = {d: i for i, d in enumerate(delays)}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for ax, metric in zip(axes, A.PRIMARY):
        header, higher = A.METRICS[metric]
        fams: dict[tuple, list[A.Cell]] = {}
        for c in values:
            fams.setdefault((c.loss, c.lp, c.use_temporal, c.backbone), []).append(c)
        for cells in fams.values():
            pts = sorted((xs[c.delay], summ[(c, metric)]) for c in cells if (c, metric) in summ)
            if not pts:
                continue
            label, color, _, marker = style(cells[0])
            ax.errorbar([p[0] for p in pts], [p[1]["mean"] for p in pts], yerr=[_half(p[1]) for p in pts],
                        color=color, marker=marker, ms=6, lw=1.4, capsize=3,
                        ls="-" if cells[0].use_temporal else "--", label=label.replace("\n", " "))
        ax.set_xticks(range(len(delays)))
        ax.set_xticklabels([d.replace("_", " ") for d in delays])
        ax.set_xlim(-0.3, len(delays) - 0.7)
        ax.set_xlabel("label-delay regime (lines only connect the regimes; no interpolation is implied)", fontsize=7.5)
        ax.set_ylabel(f"Test {header} ({'higher' if higher else 'lower'} is better)")
        ax.set_title(header)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Effect of the delay regime (mean ± 95% CI)", y=1.0)
    fig.tight_layout()
    _save(fig, path, prereg)


def figure_recovery(rec: Sequence[dict], path: Path, prereg: str) -> bool:
    if not rec:
        return False
    delays = _delays({r["delay"] for r in rec})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for ax, metric in zip(axes, A.PRIMARY):
        rows = [r for r in rec if r["metric"] == metric]
        fams = sorted({(r["loss"], r["lp"], r["use_temporal"], r["backbone"]) for r in rows},
                      key=lambda f: A._sort_key(A.Cell("", f[0], f[3], f[2], f[1])))
        width = 0.8 / max(len(delays), 1)
        for j, d in enumerate(delays):
            for i, f in enumerate(fams):
                r = next((q for q in rows if q["delay"] == d and (q["loss"], q["lp"], q["use_temporal"], q["backbone"]) == f), None)
                if r is None:
                    continue
                x = i + (j - (len(delays) - 1) / 2) * width
                if r["reported"]:
                    ax.bar(x, r["value"], width * 0.92, color=SHADES[j % 3], edgecolor="black", linewidth=0.6)
                else:
                    ax.text(x, 0.02, "n/r", ha="center", va="bottom", rotation=90, fontsize=7, color="#444444")
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(1, color="#009E73", ls="--", lw=0.9)
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels([style(A.Cell("", f[0], f[3], f[2], f[1]))[0] for f in fams], fontsize=7.5)
        ax.set_ylabel("Recovery Ratio (0 = A0, 1 = oracle)")
        ax.set_title(A.METRICS[metric][0])
        ax.grid(axis="y", alpha=0.3)
        ax.legend(handles=[Patch(facecolor=SHADES[j % 3], edgecolor="black", label=d.replace("_", " ")) for j, d in enumerate(delays)],
                  frameon=False, fontsize=7)
    fig.suptitle("Recovery Ratio (n/r: oracle-A0 gap inside A0's 95% CI; not clipped)", y=1.0)
    fig.tight_layout()
    _save(fig, path, prereg)
    return True


def bundle(dirs: Sequence[Path], out_zip: Path) -> None:
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for root in dirs:
            for f in root.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(root.parent))


def default_result_patterns() -> list[str]:
    """Where benchmark CSVs normally live: <repo>/results, plus /kaggle/working/results when it exists."""
    pats = [str(REPO_ROOT / "results" / "benchmark_results*.csv")]
    if KAGGLE_ROOT.exists():
        pats.append(str(KAGGLE_ROOT / "results" / "benchmark_results*.csv"))
    return pats


def expand_files(patterns: Sequence[str]) -> list[str]:
    """Glob every pattern, drop duplicates that resolve to the same file (repo may BE /kaggle/working)."""
    seen: dict[Path, str] = {}
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            seen.setdefault(Path(p).resolve(), p)
    return sorted(seen.values())


def main(argv: Sequence[str] | None = None) -> int:
    out_base = KAGGLE_ROOT if KAGGLE_ROOT.exists() else REPO_ROOT / "results"
    ap = argparse.ArgumentParser(description="Figures and LaTeX tables from benchmark CSVs.")
    ap.add_argument("--results", nargs="+", default=default_result_patterns())
    ap.add_argument("--fig_dir", default=str(out_base / "figures"))
    ap.add_argument("--out_dir", default=str(out_base / "analysis"))
    args = ap.parse_args(argv)

    files = expand_files(args.results)
    if not files:
        print(f"[report] ERROR: no CSV matches {args.results}")
        return 1
    df = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files], ignore_index=True)
    rows = df.to_dict("records")
    fig_dir, out_dir = Path(args.fig_dir), Path(args.out_dir)
    try:
        res = A.run_analysis(rows, out_dir, plots=False)       # CSVs + .tex tables (same rules as the pre-registration)
    except ValueError as exc:
        print(f"[report] ERROR: {exc}")
        return 1
    values = A.collect_values(A.clean_rows(rows)[0])
    summ, comp, rec, prereg = res["summary"], res["comparison"], res["recovery"], res["prereg"]

    print(f"[report] {len(files)} file(s), {res['n_runs']} ok runs | pre-registration: {prereg.upper()}")
    for f in files:
        print(f"[report] using {f}")
    for w in res["warnings"]:
        print(f"[report] WARNING: {w}")
    figure_bars(values, summ, comp, "test_pr_auc", fig_dir / "fig1_pr_auc_bars.png", prereg)
    figure_bars(values, summ, comp, "test_efl_ratio", fig_dir / "fig2_efl_bars.png", prereg)
    figure_lines(values, summ, fig_dir / "fig3_delay_lines.png", prereg)
    figure_recovery(rec, fig_dir / "fig4_recovery.png", prereg)

    tab = pd.DataFrame([{"delay": c.delay, "method": A.method_label(c, latex=False).replace("\n", " "), "metric": m,
                         "n": int(s["n"]), "mean": round(s["mean"], 4), "ci_half": round(s["half"], 4),
                         "verdict": comp.get((c, m), {}).get("verdict", "")}
                        for (c, m), s in summ.items() if m in A.PRIMARY])
    print("\n" + tab.sort_values(["delay", "metric", "method"]).to_string(index=False))
    print("\n" + "=" * 30 + " LaTeX " + "=" * 30)
    for f in sorted(out_dir.glob("table_*.tex")):
        print(f"% ---- {f.name} ----")
        print(f.read_text(encoding="utf-8"))
    bundle([fig_dir, out_dir], out_base / "report_bundle.zip")
    print(f"[report] figures -> {fig_dir} | tables/CSV -> {out_dir} | zip -> {out_base / 'report_bundle.zip'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())