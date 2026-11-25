# DelayGraphBench / TW-PU-TGNN

Continuous-time heterogeneous transaction graphs for fraud detection under **label delay** on IEEE-CIS.
The V2 protocol compares Naive BCE, Maturity-Masked BCE, Delay-IPW and TW-nnDF. Hetero-GraphSAGE is the
main backbone; Hetero-GAT is the secondary backbone check. The V2 architecture does **not** use GRU, TGN,
EvolveGCN or domain adaptation.

> Status: the benchmark run matrix and thresholds are **NOT pre-registered/frozen yet**
> (`src.train.utils.PREREG_STATUS = "unfrozen"`). Every result row and every LaTeX caption carries this status.
> Nothing here is a reproduction of a published system; "ATLAS-style" means lag-safe LP features inspired by ATLAS.

## Layout

`configs/` | `src/data/` | `src/graph/` | `src/features/` | `src/models/` | `src/losses/` |
`src/train/` | `src/eval/` | `scripts/` | `tests/` | `results/`

## Setup

Install PyTorch, a matching PyG build (including the sampler dependencies required by your environment),
pandas, PyYAML, and pytest. Use versions that match your torch/CUDA build.

## V2 protocol

* Continuous timestamps are retained at transaction-level resolution.
* P = reported fraud, N = mature unreported, U = immature unreported at the training label cut.
* The train graph hides `y_true`; only the explicitly labelled oracle run may expose it.
* Hetero-GraphSAGE is the main model and Hetero-GAT is the comparator. `model.use_temporal: false` is required:
  no recurrent GRU branch is active.
* Temporal causality is enforced by the graph sampler (`strict_time: true`): sampled neighbours must be earlier
  than the seed transaction.
* TW-nnDF combines maturity-masked BCE with cost-sensitive nnPU. Amount thresholds and amount references are
  fitted on the training view only. The unlabeled prior is estimated from observed training labels and bounded
  only by the numerical `PRIOR_EPS`; no floor based on hidden fraud prevalence is permitted.
* Validation selects the best epoch and fits the FPR operating threshold. Test PR-AUC/ROC-AUC are ranking metrics;
  test Recall/FPR and EFL are computed by applying that **validation-fitted threshold**. A threshold is never fitted
  on test labels.
* `negative_risk_rate` records batches with negative raw nnPU negative risk. `nnpu_ascent_rate` records batches
  using the nnPU ascent branch. `zero_pu_rate` records batches with no valid P/U contribution. These are distinct
  diagnostics; a negative raw risk is not synonymous with a clamped objective in nnPU mode.
* Scores are logits. EFL = missed fraud amount + `review_cost` × FP count (`review_cost = 0` by default; freeze
  the business cost before interpreting EFL as a deployment metric).
* A method counts as an improvement only if its seed-mean (PR-AUC or EFL ratio) is outside the 95% CI of A0.
* Recovery Ratio = `(S_method - S_A0) / (S_oracle - S_A0)`, reported only if the oracle-A0 gap exceeds A0's 95% CI.
* Main benchmark: 4 methods × 2 backbones × 3 delay regimes × 3 seeds = 72 method runs, followed by 18 oracle runs.
  The V2 probe config is a separate 15-run matrix: 2 baselines × 3 seeds plus 3 TW-nnDF variants × 3 seeds.
* 3 seeds (42, 43, 44), Student-t CI (t = 4.303 for n = 3). GPU runs are not bit-reproducible.

## Known limits

* The prior estimated from mature observed training labels can differ from the true hidden-fraud rate in U.
  The true rate is a post-hoc diagnostic only; it must never determine the prior floor or training objective.
* Cost-sensitive weighting assumes the amount distribution used for weighting is informative for the hidden-positive
  population. Amount-dependent reporting delays can violate this assumption and must be treated as a threat to validity.
* Confidence-filtering U changes the selected-U distribution and can bias the prior; `v2_conf` is a heuristic ablation,
  not an unbiased nnPU estimator.
* LP features are counted on sampled neighbours (fan-out limited); only seed nodes carry them.
* Test results of different delay settings are not comparable in absolute value unless the test set is identical.

## Steps (commands are for you to run)

1. Data: `python scripts/prepare_data.py --delay_config configs/delay_high.yaml` (and other delay regimes)
2. Tests: `pytest -q`
3. Profile: `python scripts/profile_epoch.py --processed_dir data/processed/delay_high --device cuda`
4. Plan: `python scripts/run_benchmark.py --dry_run --est_epoch_minutes <measured>`
5. Smoke: `python scripts/run_benchmark.py --smoke --seeds 42 --delays delay_high`
6. Benchmark: `python scripts/run_benchmark.py --max_hours 10` (re-run the same command to resume)
7. Analysis: `python -m src.eval.analyzer --results results/benchmark_results.csv --out results/analysis`

Fill `lam_by_delay` in `configs/benchmark.yaml` before the main benchmark (lambda = ln2 / median delay in days,
per delay).

## Resuming across Kaggle sessions

`results/benchmark_results.csv` is append-only (one fsynced row per finished run). Before a session ends,
keep that file (download it, or save it as a Kaggle dataset). In the next session, pass it with `--seed_results <path>`;
it is copied only if `results/benchmark_results.csv` does not exist yet. Check your own weekly GPU quota and session
limit; they are not assumed here.

If you change anything in train/model/sampler/eval/loss/features or `lam_by_delay`, the runner refuses to continue
on the old CSV (config hash); use a new `--results` file.

## Outputs (`results/analysis`)

`summary.csv`, `recovery.csv`, `nndf_diagnostics.csv`, `table_main_<delay>.tex`, `table_recovery.tex`,
`fig_test_pr_auc.pdf`, `fig_test_efl_ratio.pdf`, `figures.tex`.