from src.preprocessing import preprocess_dataset

def test_preprocess_dataset_returns_paths():
    res = preprocess_dataset("pepper")
    assert "processed_dir" in res
    assert res["dataset"] == "pepper"
