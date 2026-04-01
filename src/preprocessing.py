from src.config import ProjectConfig

def preprocess_dataset(dataset=None):
    cfg = ProjectConfig()
    ds = dataset or cfg.get_dataset()
    return {"dataset": ds, "processed_dir": cfg.get_processed_dir()}
