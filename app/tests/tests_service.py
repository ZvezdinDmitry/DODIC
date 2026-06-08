import io

from fastapi.testclient import TestClient
from PIL import Image

from app.api.app import MAX_FILE_SIZE, app

client = TestClient(app)


def test_model_indefence(mock_model):
    fake_image = Image.new("RGB", (2, 2), color="white")
    img_byte_arr = io.BytesIO()
    fake_image.save(img_byte_arr, format="JPEG")
    fake_image = img_byte_arr.getvalue()

    files = {"file": ("test.jpg", fake_image, "image/jpeg")}
    data = {"damage_conf": 0.3, "part_conf": 0.2, "part_overlap": 0.2}

    response = client.post("/", data=data, files=files)

    assert response.status_code == 200
    assert "Царапина" in response.text
    assert "Бампер" in response.text


def test_model_internal_error(mock_fail_model):
    fake_image = Image.new("RGB", (2, 2), color="white")
    img_byte_arr = io.BytesIO()
    fake_image.save(img_byte_arr, format="JPEG")
    fake_image = img_byte_arr.getvalue()

    files = {"file": ("test.jpg", fake_image, "image/jpeg")}
    data = {"damage_conf": 0.3, "part_conf": 0.2, "part_overlap": 0.2}

    response = client.post("/", data=data, files=files)

    assert response.status_code == 500


def test_empty_file():
    fake_big_file = b""

    files = {"file": ("test.jpg", fake_big_file, "image/jpeg")}
    data = {"damage_conf": 0.3, "part_conf": 0.2, "part_overlap": 0.2}

    response = client.post("/", data=data, files=files)

    assert response.status_code == 400


def test_non_image():
    fake_big_file = b"just a string"

    files = {"file": ("test.jpg", fake_big_file, "image/jpeg")}
    data = {"damage_conf": 0.3, "part_conf": 0.2, "part_overlap": 0.2}

    response = client.post("/", data=data, files=files)

    assert response.status_code == 400


def test_big_file():
    fake_big_file = b"1" * MAX_FILE_SIZE * 2

    files = {"file": ("test.jpg", fake_big_file, "image/jpeg")}
    data = {"damage_conf": 0.3, "part_conf": 0.2, "part_overlap": 0.2}

    response = client.post("/", data=data, files=files)

    assert response.status_code == 413
