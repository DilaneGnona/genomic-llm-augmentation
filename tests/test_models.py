from src.models import build_model

def test_build_ridge_model():
    model = build_model("ridge")
    assert model is not None
