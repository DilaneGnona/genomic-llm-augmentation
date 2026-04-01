#!/usr/bin/env python3
import argparse
import os
import sys
import json
import csv
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def discover_datasets(base_processed_dir: str, explicit: Optional[List[str]]) -> List[str]:
    if explicit and explicit != ["all"]:
        return explicit
    if not os.path.isdir(base_processed_dir):
        return []
    ds = []
    for name in os.listdir(base_processed_dir):
        d = os.path.join(base_processed_dir, name)
        if not os.path.isdir(d):
            continue
        x_path = os.path.join(d, "X.csv")
        y_path = os.path.join(d, "y.csv")
        if os.path.isfile(x_path) and os.path.isfile(y_path):
            ds.append(name)
    return sorted(ds)


def try_index(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer an explicit sample_id-like column
    for col in df.columns[:3]:
        if str(col).lower() in {"sample_id", "id", "sample", "individual"}:
            df = df.set_index(col)
            return df
    # Otherwise if the first column is non-numeric, treat it as index
    first = df.columns[0]
    if not pd.api.types.is_numeric_dtype(df[first]):
        df = df.set_index(first)
    return df


def load_X(dataset_dir: str) -> pd.DataFrame:
    path = os.path.join(dataset_dir, "X.csv")
    df = pd.read_csv(path)
    df = try_index(df)
    # keep only numeric features
    num_df = df.select_dtypes(include=[np.number])
    return num_df


def load_y(dataset_dir: str) -> pd.DataFrame:
    path = os.path.join(dataset_dir, "y.csv")
    df = pd.read_csv(path)
    df = try_index(df)
    return df


def compute_genotype_quality(X: pd.DataFrame) -> Dict[str, Any]:
    n_samples = X.shape[0]
    n_snps = X.shape[1]
    # missingness
    missing_rate_overall = float(X.isna().mean().mean()) if n_snps > 0 else None
    # per-feature NA fraction
    na_frac = X.isna().mean().values if n_snps > 0 else np.array([])
    # variance per SNP
    var = X.var(axis=0, ddof=1).values if n_snps > 0 else np.array([])
    usable_mask = (na_frac < 0.2) & (var > 1e-9)
    usable_snps = int(usable_mask.sum())

    # PCA capture ratio (top-k explained variance sum)
    pca_capture_20 = None
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        # simple impute: fill NA with column means
        X_imp = X.copy()
        X_imp = X_imp.fillna(X_imp.mean())
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_scaled = scaler.fit_transform(X_imp.values)
        k = max(1, min(20, min(X_scaled.shape) - 1))
        pca = PCA(n_components=k, svd_solver="auto", random_state=42)
        evr = pca.fit(X_scaled).explained_variance_ratio_
        pca_capture_20 = float(np.sum(evr))
    except Exception:
        pass

    # LD proxy: fraction of |corr|>0.2 among sampled pairs
    ld_fraction_high_corr = None
    try:
        # sample up to 200 features to keep it light
        rng = np.random.default_rng(42)
        cols = X.columns.tolist()
        if len(cols) > 200:
            cols = list(rng.choice(cols, size=200, replace=False))
        X_sub = X[cols].fillna(X[cols].mean())
        corr = np.corrcoef(X_sub.values, rowvar=False)
        # off-diagonal absolute correlations
        m = corr.shape[0]
        mask = ~np.eye(m, dtype=bool)
        vals = np.abs(corr[mask])
        ld_fraction_high_corr = float((vals > 0.2).mean())
    except Exception:
        pass

    return {
        "n_samples": int(n_samples),
        "n_snps": int(n_snps),
        "usable_snps": int(usable_snps),
        "missing_rate_overall": missing_rate_overall,
        "pca_capture_20": pca_capture_20,
        "ld_fraction_high_corr": ld_fraction_high_corr,
    }


def compute_phenotype_quality(y: pd.DataFrame) -> Dict[str, Any]:
    # identify numeric columns as potential targets
    num_cols = y.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in y.columns if c not in num_cols]
    missing_rate = float(y.isna().mean().mean()) if y.shape[1] > 0 else None

    # label balance: for numeric, compute skew and simple histogram; for categorical, largest class fraction
    label_balance = None
    target_type = None
    target_candidates = num_cols if num_cols else y.columns.tolist()

    if num_cols:
        target_type = "numeric"
        # choose the first numeric column as default target
        col = num_cols[0]
        s = y[col].dropna()
        # Fisher-Pearson skewness
        try:
            label_balance = float(s.skew())
        except Exception:
            label_balance = None
    else:
        target_type = "categorical"
        # coarse imbalance for the first column
        col = target_candidates[0] if target_candidates else None
        if col is not None:
            counts = y[col].value_counts(dropna=True)
            total = int(counts.sum())
            if total > 0:
                label_balance = float(counts.iloc[0] / total)

    return {
        "phenotype_columns": target_candidates,
        "phenotype_missing_rate": missing_rate,
        "target_type": target_type,
        "label_balance": label_balance,
        "has_numeric_target": bool(num_cols),
    }


def match_samples(X: pd.DataFrame, y: pd.DataFrame, dataset_dir: str) -> Tuple[List[str], float]:
    # Use intersection of indices
    x_ids = set(X.index.astype(str))
    y_ids = set(y.index.astype(str))
    common = sorted(x_ids & y_ids)
    denom = max(len(x_ids), len(y_ids)) if max(len(x_ids), len(y_ids)) > 0 else 1
    ratio = len(common) / denom
    # Try sample_map fallback
    if not common and os.path.isfile(os.path.join(dataset_dir, "sample_map.csv")):
        try:
            sm = pd.read_csv(os.path.join(dataset_dir, "sample_map.csv"))
            # assume columns like genotype_id, phenotype_id
            g_col = None
            p_col = None
            for c in sm.columns:
                lc = str(c).lower()
                if g_col is None and lc in {"genotype_id", "geno_id", "genotype", "id_x", "x_id"}:
                    g_col = c
                if p_col is None and lc in {"phenotype_id", "pheno_id", "phenotype", "id_y", "y_id"}:
                    p_col = c
            if g_col and p_col:
                map_dict = dict(zip(sm[g_col].astype(str), sm[p_col].astype(str)))
                mapped = [map_dict.get(i) for i in x_ids]
                mapped_set = set(m for m in mapped if m is not None)
                common = sorted(mapped_set & y_ids)
                denom = max(len(x_ids), len(y_ids)) if max(len(x_ids), len(y_ids)) > 0 else 1
                ratio = len(common) / denom
        except Exception:
            pass
    return common, ratio


def baseline_signal_r2(X: pd.DataFrame, y: pd.DataFrame, common_ids: List[str], target_col: Optional[str]) -> Optional[float]:
    if not common_ids:
        return None
    # Select target
    if target_col is None:
        num_cols = y.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return None
        target_col = num_cols[0]
    y_vec = y.loc[common_ids, target_col].astype(float)
    # Drop rows with NA in target
    mask = y_vec.notna()
    common_ids = list(np.array(common_ids)[mask.values])
    y_vec = y_vec[mask]
    if len(common_ids) < 20:
        return None

    # Align X and impute
    X_sub = X.loc[common_ids].copy()
    X_sub = X_sub.fillna(X_sub.mean())

    # Basic feature filtering: variance threshold
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    try:
        sel = VarianceThreshold(threshold=1e-6)
        X_sel = sel.fit_transform(X_sub.values)
        # Reduce dimensionality slightly if extremely large
        # Limit to at most 1000 features via random projection of columns
        if X_sel.shape[1] > 1000:
            rng = np.random.default_rng(42)
            idx = rng.choice(X_sel.shape[1], size=1000, replace=False)
            X_sel = X_sel[:, idx]

        scaler = StandardScaler(with_mean=True, with_std=True)
        X_scaled = scaler.fit_transform(X_sel)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_vec.values, test_size=0.2, random_state=42
        )
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = float(r2_score(y_test, y_pred))
        return r2
    except Exception:
        return None


def genotype_quality_bucket(g: Dict[str, Any]) -> str:
    # Heuristic buckets
    if g["n_samples"] >= 100 and g["usable_snps"] >= 100 and (g["missing_rate_overall"] is None or g["missing_rate_overall"] < 0.1):
        return "High"
    if g["n_samples"] >= 50 and g["usable_snps"] >= 50:
        return "Medium"
    return "Low"


def phenotype_quality_bucket(p: Dict[str, Any]) -> str:
    if p["has_numeric_target"] and (p["phenotype_missing_rate"] is None or p["phenotype_missing_rate"] < 0.1):
        return "High"
    if p["has_numeric_target"]:
        return "Medium"
    return "Low"


def augmentation_potential_bucket(g_bucket: str, p_bucket: str, overlap_ratio: float, baseline_r2: Optional[float]) -> str:
    if g_bucket == "High" and p_bucket == "High" and overlap_ratio >= 0.8 and (baseline_r2 or 0) >= 0.1:
        return "High"
    if overlap_ratio >= 0.5 and (baseline_r2 or 0) >= 0.05:
        return "Medium"
    return "Low"


def plot_pca_variance(dataset_name: str, out_dir: str, X: pd.DataFrame):
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        X_imp = X.fillna(X.mean())
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_scaled = scaler.fit_transform(X_imp.values)
        k = max(1, min(20, min(X_scaled.shape) - 1))
        pca = PCA(n_components=k, random_state=42)
        evr = pca.fit(X_scaled).explained_variance_ratio_
        plt.figure(figsize=(6, 3))
        plt.bar(range(1, len(evr) + 1), evr)
        plt.xlabel("PC")
        plt.ylabel("Explained Variance Ratio")
        plt.title(f"PCA Variance: {dataset_name}")
        plt.tight_layout()
        ensure_dir(out_dir)
        plt.savefig(os.path.join(out_dir, f"{dataset_name}_pca_variance.png"))
        plt.close()
    except Exception:
        pass


def plot_label_distribution(dataset_name: str, out_dir: str, y: pd.Series):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3))
        if pd.api.types.is_numeric_dtype(y):
            y = y.dropna()
            plt.hist(y.values, bins=20, color="steelblue")
            plt.ylabel("Count")
            plt.title(f"Label Distribution: {dataset_name}")
        else:
            counts = y.value_counts(dropna=True)
            counts.plot(kind="bar")
            plt.ylabel("Count")
            plt.title(f"Label Distribution: {dataset_name}")
        plt.tight_layout()
        ensure_dir(out_dir)
        plt.savefig(os.path.join(out_dir, f"{dataset_name}_label_distribution.png"))
        plt.close()
    except Exception:
        pass


def plot_overlap_vs_r2_heatmap(out_path: str, rows: List[Dict[str, Any]]):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        labels = [r["dataset_name"] for r in rows]
        overlaps = np.array([float(r["overlap_ratio"]) if r["overlap_ratio"] is not None else np.nan for r in rows])
        r2s = np.array([float(r["baseline_r2"]) if r["baseline_r2"] is not None else np.nan for r in rows])
        # Normalize to [0,1] for visualization
        ov_norm = np.nan_to_num((overlaps - np.nanmin(overlaps)) / (np.nanmax(overlaps) - np.nanmin(overlaps) + 1e-9))
        r2_norm = np.nan_to_num((r2s - np.nanmin(r2s)) / (np.nanmax(r2s) - np.nanmin(r2s) + 1e-9))
        data = np.vstack([ov_norm, r2_norm])
        plt.figure(figsize=(max(6, len(labels) * 0.8), 3))
        plt.imshow(data, aspect="auto", cmap="viridis")
        plt.colorbar(label="Normalized")
        plt.yticks([0, 1], ["Overlap", "Baseline R²"]) 
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                plt.text(j, i, f"{(overlaps if i==0 else r2s)[j]:.2f}", ha="center", va="center", color="white")
        plt.title("Overlap Ratio vs Baseline R²")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Analyze dataset quality for augmentation potential")
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset names or 'all'")
    parser.add_argument("--out_dir", required=True, help="Output directory for report and CSV")
    args = parser.parse_args()

    base = os.getcwd()
    processed_base = os.path.join(base, "02_processed_data")
    in_datasets = [d.strip() for d in args.datasets.split(",")]
    datasets = discover_datasets(processed_base, in_datasets)
    if not datasets:
        print("[ERROR] No datasets found to analyze.", file=sys.stderr)
        sys.exit(2)

    out_dir = args.out_dir
    ensure_dir(out_dir)
    plots_dir = os.path.join(out_dir, "plots", "dataset_quality")
    ensure_dir(plots_dir)

    rows: List[Dict[str, Any]] = []
    md_lines: List[str] = []
    md_lines.append("## Dataset Quality Assessment")
    md_lines.append("")
    md_lines.append("Analyzed datasets: {}".format(", ".join(datasets)))
    md_lines.append("")

    for ds in datasets:
        ds_dir = os.path.join(processed_base, ds)
        try:
            # Load
            try:
                X = load_X(ds_dir)
                y = load_y(ds_dir)
            except Exception as e:
                md_lines.append(f"### {ds}")
                md_lines.append(f"- Error loading data: {e}")
                rows.append({
                    "dataset_name": ds,
                    "n_samples": None,
                    "n_snps": None,
                    "phenotype_columns": None,
                    "overlap_ratio": None,
                    "baseline_r2": None,
                    "genotype_quality": "Low",
                    "phenotype_quality": "Low",
                    "augmentation_potential": "Low",
                })
                md_lines.append("")
                continue

            # Compute metrics
            gq = compute_genotype_quality(X)
            pq = compute_phenotype_quality(y)
            common_ids, overlap_ratio = match_samples(X, y, ds_dir)

            target_col = None
            if pq.get("has_numeric_target"):
                pcs = pq.get("phenotype_columns") or []
                target_col = pcs[0] if pcs else None
            baseline_r2 = baseline_signal_r2(X, y, common_ids, target_col)

            # Buckets
            g_bucket = genotype_quality_bucket(gq)
            p_bucket = phenotype_quality_bucket(pq)
            aug_bucket = augmentation_potential_bucket(g_bucket, p_bucket, overlap_ratio, baseline_r2)

            # plots
            plot_pca_variance(ds, plots_dir, X)
            # label hist for selected target (numeric preferred)
            try:
                label_series = y[target_col] if target_col else y[y.columns[0]]
            except Exception:
                label_series = y[y.columns[0]] if y.shape[1] > 0 else pd.Series([])
            plot_label_distribution(ds, plots_dir, label_series)

            # Markdown lines
            md_lines.append(f"### {ds}")
            md_lines.append("- Genotype: samples = {} | SNPs = {} | usable SNPs = {} | missing rate = {:.3f}".format(
                gq.get("n_samples"), gq.get("n_snps"), gq.get("usable_snps"), gq.get("missing_rate_overall") or 0.0))
            md_lines.append("- Structure: PCA capture (top 20) = {} | LD high-corr frac = {}".format(
                f"{gq['pca_capture_20']:.3f}" if gq.get("pca_capture_20") is not None else "n/a",
                f"{gq['ld_fraction_high_corr']:.3f}" if gq.get("ld_fraction_high_corr") is not None else "n/a",
            ))
            md_lines.append("- Phenotype: cols = {} | missing rate = {} | target type = {} | balance = {}".format(
                ",".join(map(str, pq.get("phenotype_columns") or [])) if pq.get("phenotype_columns") else "n/a",
                f"{pq['phenotype_missing_rate']:.3f}" if pq.get("phenotype_missing_rate") is not None else "n/a",
                pq.get("target_type"),
                f"{pq['label_balance']:.3f}" if pq.get("label_balance") is not None else "n/a",
            ))
            md_lines.append("- Linkage: overlap ratio = {:.3f} | baseline R² (Ridge) = {}".format(
                overlap_ratio, f"{baseline_r2:.3f}" if baseline_r2 is not None else "n/a"))
            md_lines.append("- Buckets: genotype = {} | phenotype = {} | augmentation potential = {}".format(
                g_bucket, p_bucket, aug_bucket))
            md_lines.append("")

            rows.append({
                "dataset_name": ds,
                "n_samples": gq.get("n_samples"),
                "n_snps": gq.get("n_snps"),
                "phenotype_columns": ";".join(map(str, pq.get("phenotype_columns") or [])) if pq.get("phenotype_columns") else "",
                "overlap_ratio": overlap_ratio,
                "baseline_r2": baseline_r2,
                "genotype_quality": g_bucket,
                "phenotype_quality": p_bucket,
                "augmentation_potential": aug_bucket,
            })
        except Exception as e:
            # Catch-all protection to avoid crashing the whole run for one dataset
            md_lines.append(f"### {ds}")
            md_lines.append(f"- Error during analysis: {e}")
            rows.append({
                "dataset_name": ds,
                "n_samples": None,
                "n_snps": None,
                "phenotype_columns": None,
                "overlap_ratio": None,
                "baseline_r2": None,
                "genotype_quality": "Low",
                "phenotype_quality": "Low",
                "augmentation_potential": "Low",
            })
            md_lines.append("")

    # Ranking and top candidates
    md_lines.append("### Top Candidates for Synthetic Data Generation")
    # sort by augmentation potential then baseline_r2 desc
    def rank_key(r):
        tier = {"High": 2, "Medium": 1, "Low": 0}.get(r["augmentation_potential"], 0)
        return (tier, r["baseline_r2"] if r["baseline_r2"] is not None else -1)
    ranked = sorted(rows, key=rank_key, reverse=True)
    for r in ranked[:3]:
        md_lines.append("- {} (potential: {}, baseline R²: {})".format(
            r["dataset_name"], r["augmentation_potential"],
            f"{r['baseline_r2']:.3f}" if r["baseline_r2"] is not None else "n/a"
        ))
    md_lines.append("")

    # Recommendations section
    highs = [r for r in rows if r.get("augmentation_potential") == "High"]
    meds = [r for r in rows if r.get("augmentation_potential") == "Medium"]
    md_lines.append("### Recommendations")
    if highs:
        md_lines.append("- Prioritize: " + ", ".join(r["dataset_name"] for r in highs))
        md_lines.append("- Rationale: strong genotype/phenotype quality, high overlap, baseline signal present")
    elif meds:
        md_lines.append("- Consider: " + ", ".join(r["dataset_name"] for r in meds))
        md_lines.append("- Rationale: acceptable linkage and signal; augment with careful tuning")
    else:
        # Best of Low: choose dataset with largest n_samples and SNPs
        best = sorted(rows, key=lambda r: (r.get("n_samples") or 0, r.get("n_snps") or 0), reverse=True)[0] if rows else None
        if best:
            md_lines.append(f"- Best candidate: {best['dataset_name']} (augmentation potential: {best['augmentation_potential']})")
            md_lines.append("- Rationale: larger sample size and SNP coverage improves augmentation robustness; baseline signal can be enhanced with synthetic data.")
            if best["dataset_name"].startswith("pepper"):
                md_lines.append("- Tip: use `selected_k=1000` and `sigma_resid_factor` in [0.25, 0.5]; verify sample mapping and target column consistency.")
            if best["dataset_name"] == "ipk_out_raw":
                md_lines.append("- Tip: verify `sample_map.csv` alignment; label scaling may explain negative baseline R².")
        else:
            md_lines.append("- No suitable datasets identified.")
    md_lines.append("")

    # Write outputs
    md_path = os.path.join(out_dir, "dataset_quality_assessment.md")
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
    except Exception as e:
        print(json.dumps({
            "error": "Failed to write markdown",
            "path": md_path,
            "exception": str(e)
        }))

    csv_path = os.path.join(out_dir, "dataset_quality_assessment.csv")
    if rows:
        cols = [
            "dataset_name", "n_samples", "n_snps", "phenotype_columns", "overlap_ratio",
            "baseline_r2", "genotype_quality", "phenotype_quality", "augmentation_potential"
        ]
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
        except Exception as e:
            print(json.dumps({
                "error": "Failed to write csv",
                "path": csv_path,
                "exception": str(e)
            }))

    # Heatmap of overlap vs baseline r2
    heatmap_path = os.path.join(plots_dir, "overlap_vs_baseline_r2_heatmap.png")
    plot_overlap_vs_r2_heatmap(heatmap_path, rows)

    summary = {
        "markdown": md_path,
        "csv": csv_path,
        "plots": {
            "heatmap": heatmap_path,
            "per_dataset": "stored under {}/".format(plots_dir)
        },
        "datasets_analyzed": datasets,
        "rows_count": len(rows),
        "rows": rows
    }
    # Persist JSON summary to disk for easier verification
    try:
        with open(os.path.join(out_dir, "dataset_quality_assessment.json"), "w", encoding="utf-8") as jf:
            json.dump(summary, jf, indent=2)
    except Exception:
        pass
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()