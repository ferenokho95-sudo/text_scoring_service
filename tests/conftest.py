from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from scripts.train import train
from app.main import create_app


@pytest.fixture(scope="session")
def bundle(tmp_path_factory):
    target = tmp_path_factory.mktemp("model")
    train(Path(__file__).resolve().parents[1] / "data/demo_reviews.csv", target, track=False)
    return target


@pytest.fixture
def client(bundle):
    with TestClient(create_app(bundle, min_samples=5)) as api:
        yield api
