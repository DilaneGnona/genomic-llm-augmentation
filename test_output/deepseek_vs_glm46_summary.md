## Deepseek vs GLM46: Cross-Augmentation Comparison

- Filters: selected_k = 5000 | sigma_resid_factor = 0.5
- Models: RANDOMFOREST
- Outputs: CSV saved to test_output

### Metrics Table

Model | Deepseek Holdout R² | GLM46 Holdout R² | LLaMA3 Holdout R² | ΔR² (Deepseek - GLM46) | ΔR² (Deepseek - LLaMA3) | ΔR² (GLM46 - LLaMA3) | Deepseek CV std | GLM46 CV std | LLaMA3 CV std
--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:
RANDOMFOREST | 0.0000 | 0.0000 | 0.1792 | 0.0000 | -0.1792 | -0.1792 | 0.0000 | 0.0000 | 0.0411

### Notes
- Each model appears once per augmentation method; statistics aggregate across available runs.
- ΔR² positive indicates Deepseek performed better than GLM46.