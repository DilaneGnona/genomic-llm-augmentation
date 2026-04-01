from src.config import ProjectConfig

def test_config_load():
    cfg = ProjectConfig("config.yaml")
    assert cfg.get_dataset() in ["pepper", "ipk_out_raw", "pepper_10611831"]
    assert cfg.get_processed_dir().startswith("02_processed_data")
    th = cfg.get_thresholds()
    assert "PCA_COMPONENTS" in th
