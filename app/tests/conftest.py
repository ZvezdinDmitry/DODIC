import matplotlib.pyplot as plt
import numpy as np
import pytest
from api.app import app
from api.models_inference import get_model


class MockModel:
    def annotate(self, image, part_overlap, part_conf, damage_conf):
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_damages = [("Царапина", "Бампер", 0.9)]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(dummy_img)
        ax.axis("off")
        return fig, dummy_damages


class MockModelFail:
    def annotate(self, image, part_overlap, part_conf, damage_conf):
        raise RuntimeError("Test internal error code")


@pytest.fixture
def mock_model():
    app.dependency_overrides[get_model] = lambda: MockModel()

    yield

    app.dependency_overrides.clear()


@pytest.fixture
def mock_fail_model():
    app.dependency_overrides[get_model] = lambda: MockModelFail()

    yield

    app.dependency_overrides.clear()
