## Sequencing Rules (Adhered)

* Run LLMs strictly one-by-one; start the next only after the previous `randomforest_metrics_<RUN_ID>.json` is fully written.

* Parameters: `dataset=pepper`, `use_synthetic=true`, `synthetic_only=false`, `selected_k=3000`, `sigma_resid_factor=0.1`, `cross_validation_outer=2`, `cross_validation_inner=2`, `holdout_size=0.2`, `overwrite_previous=true`.

## Outputs Expected

* Per-LLM RandomForest metrics JSONs at `k=3000`.

* Cross-LLM comparison artifacts (CSV, barplots, markdown) for RandomForest.

* Updated final summary report with the new RandomForest (k=3000, σ=0.1) section.

## Current Progress

* Existing k=2000/5000 RF runs detected; no repeats.

* Synthetic y (k=3000, σ=0.1) prepared for llama3, deepseek, glm46, qwen, minimax.

* RF k=3000 launched for llama3 → deepseek → glm46; awaiting metrics completion sequentially.

## Next Steps (upon approval)

* Launch and complete qwen, then minimax RF k=3000 runs, each after confirming the previous metrics file.

* Build cross-LLM CSV/barplots/markdown under `03_modeling_results/comparative_analysis/cross_llm_k3000/randomforest/`.

* Append the RandomForest (k=3000, σ=0.1) section to `03_modeling_results/final_summary_report.md`.

* Deliver: list of previous runs, list of new k=3000 outputs, comparison files, updated summary.

