import os
import pandas as pd

BASE_DIR = r"c:\Users\OMEN\Desktop\experiment_snp"
PEPPER_DIR = os.path.join(BASE_DIR, r"02_processed_data\pepper")

def clean_file(path):
    try:
        df = pd.read_csv(path)
        if "Sample_ID" in df.columns:
            df["Sample_ID"] = df["Sample_ID"].astype(str)
            df = df[~df["Sample_ID"].str.upper().isin(["POS", "REF", "ALT"])]
            df.to_csv(path, index=False)
            return
    except Exception:
        pass
    try:
        tmp_path = path + ".tmp"
        with open(path, "r", encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                if i == 0:
                    fout.write(line)
                    continue
                if line.startswith("POS,") or line.startswith("REF,") or line.startswith("ALT,"):
                    continue
                fout.write(line)
        os.replace(tmp_path, path)
    except Exception:
        pass

def main():
    clean_file(os.path.join(PEPPER_DIR, "X.csv"))
    clean_file(os.path.join(PEPPER_DIR, "pca_covariates.csv"))
    clean_file(os.path.join(PEPPER_DIR, "sample_map.csv"))

if __name__ == "__main__":
    main()
