import os
import json
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class ProjectConfig:
    def __init__(self, path="config.yaml"):
        self.data = {}
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                self.data = yaml.safe_load(f)
        except Exception:
            self.data = {}
        self.dataset = os.getenv("DATASET", self.data.get("DATASET"))
        self.paths = self.data.get("PATHS", {})
    def get_dataset(self):
        return self.dataset or "pepper"
    def get_processed_dir(self):
        base = self.paths.get("PROCESSED_DIR", "02_processed_data")
        return os.path.join(base, self.get_dataset())
    def get_raw_dir(self):
        base = self.paths.get("RAW_DIR", "01_raw_data")
        return os.path.join(base, self.get_dataset())
    def get_target_column(self, dataset=None):
        d = dataset or self.get_dataset()
        targets = self.data.get("TARGET_COLUMNS", {})
        return targets.get(d)
    def get_thresholds(self):
        return {
            "PCA_COMPONENTS": int(self.data.get("PCA_COMPONENTS", 5)),
            "MAF_THRESHOLD": float(self.data.get("MAF_THRESHOLD", 0.05)),
            "SNP_MISSINGNESS_THRESHOLD": float(self.data.get("SNP_MISSINGNESS_THRESHOLD", 0.05)),
            "SAMPLE_MISSINGNESS_THRESHOLD": float(self.data.get("SAMPLE_MISSINGNESS_THRESHOLD", 0.1)),
        }
    def get_secret(self, name):
        env_map = self.data.get("SECRETS", {})
        key = env_map.get(name, name)
        return os.getenv(key)
