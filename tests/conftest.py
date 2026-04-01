import os
from src.config import ProjectConfig

def pytest_configure(config):
    os.environ.setdefault("DATASET", "pepper")
