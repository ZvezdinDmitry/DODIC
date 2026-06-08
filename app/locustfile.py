import os
import random

from locust import HttpUser, between, task


class AppUser(HttpUser):
    wait_time = between(1, 5)

    STATIC_DIR = "/mnt/locust/static"

    def on_start(self):
        """Вызывается один раз при старте каждого виртуального пользователя"""
        if os.path.exists(self.STATIC_DIR):
            self.images = [
                f
                for f in os.listdir(self.STATIC_DIR)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        else:
            self.images = []

    @task(1)
    def check_health(self):
        self.client.get("/metrics")

    @task(3)
    def predict_car_parts(self):
        random_image_name = random.choice(self.images)
        image_path = os.path.join(self.STATIC_DIR, random_image_name)

        with open(image_path, "rb") as image:
            files = {"file": (random_image_name, image, "image/jpeg")}
            data = {"damage_conf": 0.3, "part_conf": 0.2, "part_overlap": 0.2}
            self.client.post("/", files=files, data=data)
