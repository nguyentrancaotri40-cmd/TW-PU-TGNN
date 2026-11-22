# TW-PU-TGNN / DelayGraphBench

Temporal heterogeneous GNN for transaction fraud under **delayed (immature) labels** on IEEE-CIS.
It compares feature-level ATLAS-style label-propagation features against loss-level methods
(Maturity Masking, Delay-IPW, TW-nnDF), with a naive baseline **A0** and an **oracle**.

> Status: the benchmark run matrix and thresholds are **NOT pre-registered/frozen yet**
> (`src.train.utils.PREREG_STATUS = "unfrozen"`). Every result row and every LaTeX caption carries this status.
> Nothing here is a reproduction of a published system; "ATLAS-style" means lag-safe LP features inspired by ATLAS.

## Layout
configs/ (base.yaml, delay_low.yaml, delay_high.yaml, benchmark.yaml) | src/data, src/graph, src/features,
src/models, src/losses, src/train, src/eval | scripts/ | tests/ | results/

## Setup
pip install torch torch_geometric pyg-lib (pyg-lib is REQUIRED: the temporal sampler needs it), pandas, pyyaml, pytest.
Use versions that match your torch/CUDA build.

## Protocol (what is fixed in code)
* Labels at cut `t_train`: P = reported fraud, N = mature unreported, U = immature unreported.
* Train graph hides `y_true`; the oracle run is the only one that keeps it (`expose_train_true_labels`).
* Validation uses full `y_true` to pick the best epoch and the FPR=1% threshold; test is scored once.
* Scores are logits. EFL = missed fraud amount + review_cost x #FP (review_cost = 0 by default; EFL is our definition).
* A method counts as an improvement only if its seed-mean (PR-AUC or EFL) is outside the 95% CI of A0.
* Recovery Ratio = (S_method - S_A0) / (S_oracle - S_A0), reported only if the oracle-A0 gap exceeds A0's 95% CI.
* 3 seeds (42, 43, 44), Student-t CI (t = 4.303 for n = 3). GPU runs are not bit-reproducible.

## Known limits
* The prior estimated for tw_nndf (~3.5%) differs from the true fraud rate in U (1.39% delay_high, 1.82% delay_low):
  the clamp rate is logged per run (`nndf_clamped_*`) and summarised in `results/analysis/nndf_diagnostics.csv`.
* LP features are counted on sampled neighbours (fan-out limited); only seed nodes carry them.
* Test results of different delay settings are not comparable in absolute value unless the test set is identical.

## Steps (commands are for you to run)
1. Data:      python scripts/prepare_data.py --delay_config configs/delay_high.yaml  (and delay_low)
2. Tests:     pytest -q
3. Profile:   python scripts/profile_epoch.py --processed_dir data/processed/delay_high --device cuda
4. Plan:      python scripts/run_benchmark.py --dry_run --est_epoch_minutes <measured>
5. Smoke:     python scripts/run_benchmark.py --smoke --seeds 42 --delays delay_high
6. Benchmark: python scripts/run_benchmark.py --max_hours 10          # re-run the same command to resume
7. Analysis:  python -m src.eval.analyzer --results results/benchmark_results.csv --out results/analysis

Fill `lam_by_delay` in configs/benchmark.yaml before step 6 (lambda = ln2 / median delay in days, per delay).

## Resuming across Kaggle sessions
`results/benchmark_results.csv` is append-only (one fsynced row per finished run). Before a session ends,
keep that file (download it, or save it as a Kaggle dataset). In the next session, pass it with
`--seed_results <path>`; it is copied only if `results/benchmark_results.csv` does not exist yet.
Check your own weekly GPU quota and session limit; they are not assumed here.
If you change anything in train/model/sampler/eval/loss/features or lam_by_delay, the runner refuses to
continue on the old CSV (config hash); use a new --results file.

## Outputs (results/analysis)
summary.csv, recovery.csv, nndf_diagnostics.csv, table_main_<delay>.tex, table_recovery.tex,
fig_test_pr_auc.pdf, fig_test_efl_ratio.pdf, figures.tex.