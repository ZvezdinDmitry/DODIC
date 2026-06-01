import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


class DamagesAnnotator:
    # Constants defining the class mappings
    DAMAGE_CLASSES = {
        0: "dent",
        1: "scratch",
        2: "crack",
        3: "glass_shatter",
        4: "lamp_broken",
        5: "tire_flat",
    }
    DAMAGES_TRASLATE = {
        "dent": "Вмятина",
        "scratch": "Царапина",
        "crack": "Трещина",
        "glass_shatter": "Разбитое стекло",
        "lamp_broken": "Разбитая фара",
        "tire_flat": "Спущенное колесо",
    }
    PART_CLASSES = {
        0: "_background_",
        1: "back_bumper",
        2: "back_door",
        3: "back_glass",
        4: "back_light",
        5: "front_bumper",
        6: "front_door",
        7: "front_glass",
        8: "front_light",
        9: "hood",
        10: "mirror",
        11: "trunk_tailgate",
        12: "wheel",
    }
    PARTS_TRANSLATE = {
        "_background_": "_background_",
        "back_bumper": "Бампер",
        "back_door": "Дверь",
        "back_glass": "Стекло",
        "back_light": "Фара",
        "front_bumper": "Бампер",
        "front_door": "Дверь",
        "front_glass": "Стекло",
        "front_light": "Фара",
        "hood": "Капот",
        "mirror": "Зеркало",
        "trunk_tailgate": "Багажник",
        "wheel": "Колесо",
        "body": "Кузов",
    }
    SELF_LOCATING_DAMAGES = {
        "tire_flat": "wheel",
        "lamp_broken": "front_light",
        "glass_shatter": "front_glass",
    }

    def __init__(self, damages_yolo_path: str, parts_yolo_path: str) -> None:
        self.damage_model = YOLO(damages_yolo_path, task="segment")
        self.part_model = YOLO(parts_yolo_path, task="segment")

    def map_damages_to_parts(
        self,
        damage_result: Results,
        part_result: Results,
        overlap_threshold: float = 0.2,
    ) -> list[tuple[str, str]]:
        mapped_damages = []

        if not damage_result.masks or not damage_result.boxes:
            return mapped_damages

        orig_h, orig_w = damage_result.orig_shape

        # Helper to extract and resize masks to original image dimensions for accurate pixel matching
        def extract_and_resize_masks(
            result: Results,
        ) -> tuple[np.ndarray, np.ndarray]:
            if not result.masks or not result.boxes:
                return np.array([]), np.array([])

            masks_data = result.masks.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            resized_masks = [
                cv2.resize(
                    m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                for m in masks_data
            ]
            return np.array(resized_masks), classes

        d_masks, d_classes = extract_and_resize_masks(damage_result)
        p_masks, p_classes = extract_and_resize_masks(part_result)

        # Create a union mask of all annotated parts (True where any part exists)
        union_part_mask = (
            np.any(p_masks, axis=0)
            if p_masks.size > 0
            else np.zeros((orig_h, orig_w), dtype=bool)
        )

        for d_idx, d_mask in enumerate(d_masks):
            d_name = self.DAMAGE_CLASSES.get(
                d_classes[d_idx], "unknown_damage"
            )
            damage_confidence = float(
                damage_result.boxes[d_idx].conf[0].cpu().numpy()
            )
            if d_name in self.SELF_LOCATING_DAMAGES:
                mapped_damages.append(
                    (
                        d_name,
                        self.SELF_LOCATING_DAMAGES[d_name],
                        damage_confidence,
                    )
                )
                continue

            d_area = np.sum(d_mask)
            if d_area == 0:
                continue

            # 1. Check intersections with known parts
            if p_masks.size > 0:
                for p_idx, p_mask in enumerate(p_masks):
                    intersection_area = np.sum(d_mask & p_mask)
                    if (intersection_area / d_area) >= overlap_threshold:
                        p_name = self.PART_CLASSES.get(
                            p_classes[p_idx], "unknown_part"
                        )
                        mapped_damages.append(
                            (d_name, p_name, damage_confidence)
                        )

            # 2. Check intersection with unannotated regions (the "body")
            unannotated_intersection = np.sum(d_mask & (~union_part_mask))
            if (unannotated_intersection / d_area) >= overlap_threshold:
                mapped_damages.append((d_name, "body", damage_confidence))

        # Return deduplicated list preserving order
        return list(dict.fromkeys(mapped_damages))

    def draw_damages(self, damage_result):
        annotated_damage = damage_result.plot(
            masks=True, boxes=True, labels=True
        )
        img_damage_rgb = cv2.cvtColor(annotated_damage, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_damage_rgb)
        ax.axis("off")
        return fig

    def rename_damages_parts(self, mapped_damages):
        for i, damage in enumerate(mapped_damages):
            d_name, p_name, conf = damage
            translated = (
                self.DAMAGES_TRASLATE[d_name],
                self.PARTS_TRANSLATE[p_name],
                conf,
            )
            mapped_damages[i] = translated

        return mapped_damages

    def annotate(
        self,
        image,
        overlap_threshold: float = 0.2,
        parts_conf: float = 0.15,
        damages_conf: float = 0.3,
    ):
        damage_result = self.damage_model.predict(
            image, conf=damages_conf, verbose=False
        )
        damage_result = damage_result[0]
        part_result = self.part_model.predict(
            image, conf=parts_conf, verbose=False
        )
        part_result = part_result[0]
        mapped_damages = self.map_damages_to_parts(
            damage_result, part_result, overlap_threshold
        )
        mapped_damages_translated = self.rename_damages_parts(mapped_damages)
        damages_fig = self.draw_damages(damage_result)
        return damages_fig, mapped_damages_translated


def get_model(
    damages_yolo_path: str = "models/damages_yolo.pt",
    parts_yolo_path: str = "models/parts_yolo.pt",
) -> DamagesAnnotator:
    model = DamagesAnnotator(damages_yolo_path, parts_yolo_path)
    return model
