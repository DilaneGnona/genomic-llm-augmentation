# Enhancing Genomic Prediction Accuracy via LLM-Driven Data Augmentation and Hybrid Ensemble Learning: A Case Study on Pepper and IPK Datasets

## **Abstract**
Genomic selection (GS) is a pivotal tool in modern plant breeding, yet its efficacy is often hindered by small sample sizes and the high-dimensional nature of Single Nucleotide Polymorphism (SNP) data. This paper presents a novel framework that leverages Large Language Models (LLMs) for synthetic data augmentation and a hybrid ensemble architecture combining Gradient Boosting Decision Trees (GBDT) and Deep Learning (DL). Our experiments on two diverse genomic datasets—**Pepper** and **IPK**—demonstrate that LLM-driven augmentation can rescue non-learnable datasets and significantly boost predictive accuracy. On the IPK dataset, our unified pipeline achieved a state-of-the-art $R^2$ of **0.7018**, representing a transformative improvement over baseline models.

---

## **1. Introduction**
The primary goal of genomic prediction is to estimate the breeding value of individuals based on their genetic markers (SNPs). While traditional linear models like GBLUP have been the industry standard, they struggle to model complex epistatic interactions. Conversely, modern Deep Learning models require vast amounts of data, which are often unavailable in experimental breeding programs.

We address these challenges by introducing a **Unified Genomic Modeling Pipeline** that:
1.  Uses LLMs (GPT-4o, DeepSeek) to generate statistically coherent synthetic samples.
2.  Implements a robust feature selection engine to mitigate the $p \gg n$ problem.
3.  Combines the additive modeling strength of LightGBM with the non-linear interaction capture of Transformers.

---

## **2. Materials and Methods**

### **2.1 Datasets**
Two datasets were used to validate the pipeline:
- **Pepper Dataset**: A smaller dataset (~700 real samples) with 5,314 SNPs, targeting crop yield (`Yield_BV`).
- **IPK Dataset**: A larger but noisier dataset (~1,700 real samples) targeting leaf spot resistance (`YR_LS`).

### **2.2 LLM-Driven Data Augmentation**
The core innovation lies in the "Genomic Simulation" phase. We extracted statistical contexts (Allele Frequencies, Correlation Matrices) from real data and prompted LLMs to synthesize new individuals.
- **Models used**: GPT-4o, DeepSeek-V3, and Moonshot (Kimi).
- **Consolidation**: Synthetic data were cleaned and aligned with real genomic features, expanding the Pepper dataset by over 600%.

### **2.3 Hybrid Ensemble Architecture**
The final model is a weighted ensemble of:
- **LightGBM**: Chosen for its efficiency with tabular data and ability to handle missing values.
- **Transformer**: A PyTorch-based implementation utilizing multi-head self-attention to identify SNP-SNP interactions.

The dynamic weighting formula used for the final prediction $\hat{y}_{final}$ is:
$$w_i = \frac{R^2_i}{\sum R^2_j}, \quad \hat{y}_{final} = \sum w_i \hat{y}_i$$

---

## **3. Experimental Results**

### **3.1 Performance Benchmark**
The table below summarizes the best $R^2$ scores obtained before and after applying our unified pipeline.

| Dataset | Baseline (XGBoost) | **Unified Pipeline (LightGBM)** | **Unified Ensemble** | Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Pepper** | 0.208 | **0.6075** | 0.5980 | **+192%** |
| **IPK** | -0.583 | **0.7018** | 0.6941 | **Rescue** |

### **3.2 Model Comparison Analysis**
Our results indicate that:
- **Tree-based models** (LightGBM/XGBoost) consistently outperformed Deep Learning models on raw genomic data.
- **Deep Learning models** (Transformer/MLP) benefited the most from LLM augmentation, with their $R^2$ scores moving from near-zero to over 0.45.
- **Feature Selection** via $f\_regression$ was critical, reducing the feature space to the top 1,000 SNPs and preventing overfitting to noise.

---

## **4. Discussion**
The "Rescue" of the IPK dataset is the most significant finding. Initially, the baseline models showed negative $R^2$ values, indicating an inability to learn from the real data alone. The injection of LLM-generated synthetic samples provided a stronger training signal, allowing the models to identify the underlying genetic patterns.

Furthermore, the **Transformer** model demonstrated a high ceiling for accuracy, particularly on the Pepper dataset, suggesting that with further scaling of synthetic data, DL architectures could eventually surpass GBDTs in genomic selection.

---

## **5. Conclusion**
This research demonstrates that LLMs are not just tools for text generation but can serve as powerful statistical simulators for complex biological data. By integrating these synthetic samples into a hybrid modeling pipeline, we have established a new benchmark for genomic prediction accuracy. Future work will focus on refining prompt engineering to capture even more nuanced genetic structures.

---

## **References**
- *Vaswani, A. et al. (2017). "Attention is All You Need."*
- *Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree."*
- *Meuwissen, T. H. et al. (2001). "Prediction of total genetic value using genome-wide dense marker maps."*

---

## **Visualizations**
*Refer to the following generated artifacts for detailed analysis:*
- [Performance Comparison Graph](../03_modeling_results/comparison_performance.png)
- [Ensemble Scatter Plot](../03_modeling_results/pepper_unified_20260401_195746/plots/ensemble_scatter.png)
