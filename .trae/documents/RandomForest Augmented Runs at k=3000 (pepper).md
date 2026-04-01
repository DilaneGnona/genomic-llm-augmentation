## Scope and Assumptions
- Only RandomForest; dataset `pepper`; augmented, real+synthetic (synthetic_only=false).
- `selected_k=3000`, `sigma_resid_factor=0.1`, outer=2, inner=2, holdout=0.2.
- Use existing augmentation assets where possible; generate missing synthetic y; keep IDs aligned and numeric targets.
- For LLM names, map `qwen→qwen3coder` and honor existing per‑LLM directories under `04_augmentation/pepper/`.

## Step 1: Detect Existing RandomForest Runs
- Search `03_modeling_results/pepper_augmented/metrics/*.json` for RandomForest with `selected_k∈{2000,5000}` and `augment_mode∈{llama3, deepseek, glm46}`.
- Findings to date:
  - LLaMA3: many RandomForest runs at k=2000 and k=5000 (e.g., `randomforest_metrics_20251111_114019.json`, `randomforest_metrics_20251103_190804.json`).
  - GLM46: RandomForest present at k=2000 in `03_modeling_results/pepper_augmented/glm46/metrics/rf_metrics.json` (GLM46 uses a legacy schema; we will treat it as an existing RF run at k=2000).
  - Deepseek: none detected under central `metrics/`.
  - Qwen/Minimax: prior RF runs exist under `03_modeling_results/pepper_augmented/qwen3coder/metrics/summary.json` and `.../minimax/metrics/summary.json` (k=2000), but not under the central `metrics/` directory.
- Action: Do not repeat any existing k=2000 or k=5000 runs; proceed to new k=3000 runs for all five LLMs.

## Step 2: Prepare Synthetic y (k=3000, σ=0.1)
For each LLM in [llama3, deepseek, glm46, qwen, minimax]:
- Ensure target file exists at `04_augmentation/pepper/model_sources/<LLM>/synthetic_y_<LLM>_filtered_k3000.csv`.
- If missing:
  1) Generate synthetic y with σ=0.1 using `scripts/generate_synthetic_y_sigma.py` (teacher Ridge + residual noise):
     - `--x_syn 04_augmentation/pepper/synthetic_snps.csv` (or per‑LLM `synthetic_snps.csv` where available)
     - `--x_real 02_processed_data/pepper/X.csv`
     - `--y_real 02_processed_data/pepper/y.csv`
     - `--target_column Yield_BV`
     - `--sigma_resid_factor 0.1`
     - `--output_y 04_augmentation/pepper/model_sources/<LLM>/synthetic_y_<LLM>_sigma010.csv`
  2) Confidence filtering + clamping via `scripts/filter_synthetic_by_confidence.py`:
     - `--dataset pepper`
     - `--synthetic_file 04_augmentation/pepper/model_sources/<LLM>/synthetic_y_<LLM>_sigma010.csv`
     - `--real_file 02_processed_data/pepper/y.csv`
     - `--pca_file 02_processed_data/pepper/pca_covariates.csv`
     - `--target_column Yield_BV`
     - `--pca_threshold 0.95`
     - `--target_percentile_low 5 --target_percentile_high 95`
     - `--fallback_percent 20`
     - `--output_file 04_augmentation/pepper/model_sources/<LLM>/synthetic_y_<LLM>_filtered_k3000.csv`
  3) Place a matching `synthetic_snps.csv` in `04_augmentation/pepper/model_sources/<LLM>/` if required by the filter (copy from `04_augmentation/pepper/synthetic_snps.csv` when LLM‑specific SNPs are not present).

## Step 2b: Launch RF Training at k=3000 (per LLM)
- Runner: `scripts/unified_modeling_pipeline_augmented.py` (CLI) for modes supported by `--augment_mode` (`llama3`, `deepseek`, `glm46`). For `qwen` and `minimax`, import and override `CONFIG` to set `AUGMENT_MODE` to a descriptive string and run `main()` (same behavior; metrics recorded centrally).
- Common flags:
  - `--dataset pepper --use_synthetic true --synthetic_only false`
  - `--augment_file 04_augmentation/pepper/model_sources/<LLM>/synthetic_y_<LLM>_filtered_k3000.csv`
  - `--selected_k 3000 --sigma_resid_factor 0.1`
  - `--models randomforest --cross_validation_outer 2 --cross_validation_inner 2 --holdout_size 0.2 --overwrite_previous`
- Sequencing and wait:
  - Run LLMs strictly one‑by‑one.
  - After each run, wait until `03_modeling_results/pepper_augmented/metrics/randomforest_metrics_<RUN_ID>.json` exists and is complete (atomic write in pipeline guarantees completeness) before starting the next LLM.

## Step 3: Cross‑LLM Comparison Outputs (k=3000, RF)
- Output directory: `03_modeling_results/comparative_analysis/cross_llm_k3000/randomforest/`.
- Produce:
  - Cross‑LLM CSV summarizing CV R² mean/std and Holdout R² mean/std per LLM (llama3, deepseek, glm46, qwen, minimax) at k=3000, σ=0.1.
  - Cross‑LLM barplots (grouped bars for CV vs Holdout R²; single bars with error bars for Holdout R²).
  - Markdown summary with filters, counts, and quick takeaways.
- Implementation:
  - Use `scripts/cross_augmentation_comparison.py` for `llama3/deepseek/glm46` with `--models randomforest --sigma 0.1 --selected_k 3000 --out_dir <outdir>`.
  - Combine with centrally recorded RF metrics for `qwen` and `minimax` (added via unified pipeline import path) in a small aggregation step to extend the CSV/plots to five LLMs.

## Step 4: Update Final Summary
- Append a section “RandomForest (k=3000, σ=0.1) — Pepper Augmented” to `03_modeling_results/final_summary_report.md` including:
  - Per‑LLM table (CV R² mean/std, Holdout R² mean/std).
  - Note of effective augment sizes and RUN_IDs.
  - Link paths to the cross‑LLM CSV and plots.

## Step 5: Final Deliverables
- Detected previous runs list: enumerate all `randomforest_metrics_*` in `03_modeling_results/pepper_augmented/metrics` with `selected_k∈{2000,5000}` and `augment_mode` where available; include GLM46 legacy `rf_metrics.json`; include qwen3coder/minimax `summary.json` (not central) for completeness.
- New k=3000 outputs: list per LLM RF metrics (`randomforest_metrics_<RUN_ID>.json`), models, logs.
- Comparison files: cross‑LLM CSV, plots, and markdown saved under `03_modeling_results/comparative_analysis/cross_llm_k3000/randomforest/`.
- Updated final summary: updated `03_modeling_results/final_summary_report.md` with the RF (k=3000, σ=0.1) section.

## Notes
- The user instruction mentions waiting for `lightgbm_metrics_<runid>.json`; since we are training RandomForest only, we will instead wait on `randomforest_metrics_<RUN_ID>.json` for each run.
- `augment_mode` CLI does not include `qwen`/`minimax`; importing the pipeline and overriding `CONFIG['AUGMENT_MODE']` yields consistent metadata while keeping training identical.
- All paths are Windows‑friendly; commands will be run from repo root.

## Next Actions (upon approval)
1) Generate or verify synthetic y for each LLM and place `synthetic_snps.csv` as needed.
2) Run five RF trainings at k=3000, strictly sequential with completion checks.
3) Build cross‑LLM CSV/plots/markdown under the requested directory.
4) Append the RF (k=3000, σ=0.1) section to the final summary.
5) Return the four lists requested: previous runs, new outputs, comparison files, updated summary.