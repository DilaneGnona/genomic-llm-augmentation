#!/usr/bin/env python3
import os
import json
import csv
import importlib.util

BASE = os.getcwd()
PROCESSED = os.path.join(BASE, "02_processed_data")
OUT_DIR = os.path.join(BASE, "03_modeling_results")

spec = importlib.util.spec_from_file_location("aq", os.path.join(BASE, "scripts", "analyze_dataset_quality.py"))
aq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aq)

datasets = aq.discover_datasets(PROCESSED, ["all"]) or []
rows = []

for ds in datasets:
    ds_dir = os.path.join(PROCESSED, ds)
    try:
        X = aq.load_X(ds_dir)
        y = aq.load_y(ds_dir)
        gq = aq.compute_genotype_quality(X)
        pq = aq.compute_phenotype_quality(y)
        common_ids, overlap_ratio = aq.match_samples(X, y, ds_dir)
        target_col = None
        if pq.get("has_numeric_target"):
            pcs = pq.get("phenotype_columns") or []
            target_col = pcs[0] if pcs else None
        baseline_r2 = aq.baseline_signal_r2(X, y, common_ids, target_col)
        g_bucket = aq.genotype_quality_bucket(gq)
        p_bucket = aq.phenotype_quality_bucket(pq)
        aug_bucket = aq.augmentation_potential_bucket(g_bucket, p_bucket, overlap_ratio, baseline_r2)
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
        rows.append({
            "dataset_name": ds,
            "n_samples": None,
            "n_snps": None,
            "phenotype_columns": "",
            "overlap_ratio": None,
            "baseline_r2": None,
            "genotype_quality": "Low",
            "phenotype_quality": "Low",
            "augmentation_potential": "Low",
            "error": str(e),
        })

# Write outputs
os.makedirs(OUT_DIR, exist_ok=True)
md_path = os.path.join(OUT_DIR, "dataset_quality_assessment.md")
csv_path = os.path.join(OUT_DIR, "dataset_quality_assessment.csv")

md_lines = ["## Dataset Quality Assessment", "", "Analyzed datasets: " + ", ".join(datasets), ""]
for r in rows:
    md_lines.append(f"### {r['dataset_name']}")
    md_lines.append(f"- Genotype: samples = {r['n_samples']} | SNPs = {r['n_snps']}")
    md_lines.append(f"- Phenotype: cols = {r['phenotype_columns']}")
    ov = r['overlap_ratio']
    ov_str = f"{ov:.3f}" if ov is not None else "n/a"
    r2 = r['baseline_r2']
    r2_str = f"{r2:.3f}" if r2 is not None else "n/a"
    md_lines.append(f"- Linkage: overlap ratio = {ov_str} | baseline R² (Ridge) = {r2_str}")
    md_lines.append(f"- Buckets: genotype = {r['genotype_quality']} | phenotype = {r['phenotype_quality']} | augmentation potential = {r['augmentation_potential']}")
    md_lines.append("")

# Add Top Candidates before writing
def rank_key(r):
    tier = {"High": 2, "Medium": 1, "Low": 0}.get(r.get("augmentation_potential"), 0)
    r2 = r.get("baseline_r2")
    return (tier, r2 if r2 is not None else -1)
ranked = sorted(rows, key=rank_key, reverse=True)
md_lines.append("### Top Candidates for Synthetic Data Generation")
for r in ranked[:3]:
    r2 = r.get("baseline_r2")
    md_lines.append("- {} (potential: {}, baseline R²: {})".format(
        r.get("dataset_name"), r.get("augmentation_potential"),
        f"{r2:.3f}" if r2 is not None else "n/a"
    ))
md_lines.append("")

with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

# Recommendations section
try:
    highs = [r for r in rows if r.get("augmentation_potential") == "High"]
    meds = [r for r in rows if r.get("augmentation_potential") == "Medium"]
    md_rec = ["### Recommendations"]
    if highs:
        md_rec.append("- Prioritize: " + ", ".join(r["dataset_name"] for r in highs))
        md_rec.append("- Rationale: strong genotype/phenotype quality, high overlap, baseline signal present")
    elif meds:
        md_rec.append("- Consider: " + ", ".join(r["dataset_name"] for r in meds))
        md_rec.append("- Rationale: acceptable linkage and signal; augment with careful tuning")
    else:
        # Best of Low: choose dataset with largest n_samples and usable SNPs
        best = sorted(rows, key=lambda r: (r.get("n_samples") or 0, r.get("n_snps") or 0), reverse=True)[0] if rows else None
        if best:
            md_rec.append(f"- Best candidate: {best['dataset_name']} (augmentation potential: {best['augmentation_potential']})")
            md_rec.append("- Rationale: larger sample size and SNP coverage improves augmentation robustness; baseline signal can be enhanced with synthetic data.")
            # Add dataset-specific notes if known
            if best["dataset_name"].startswith("pepper"):
                md_rec.append("- Tip: use `selected_k=1000` and `sigma_resid_factor` in [0.25, 0.5]; verify sample mapping and target column consistency.")
            if best["dataset_name"] == "ipk_out_raw":
                md_rec.append("- Tip: verify `sample_map.csv` alignment; label scaling may explain negative baseline R².")
        else:
            md_rec.append("- No suitable datasets identified.")

    # Append recommendations to markdown file
    with open(md_path, "a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(md_rec) + "\n")
except Exception:
    pass

cols = [
    "dataset_name", "n_samples", "n_snps", "phenotype_columns", "overlap_ratio",
    "baseline_r2", "genotype_quality", "phenotype_quality", "augmentation_potential"
]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in cols})

print(json.dumps({
    "datasets": datasets,
    "rows_count": len(rows),
    "out_dir": OUT_DIR
}, indent=2))