import json
import shutil
from pathlib import Path

import kagglehub
from ultralytics.data.converter import convert_coco

CLASS_MAPPING = {
    "back_bumper": "back_bumper",
    "back_glass": "back_glass",
    "back_left_door": "back_door",
    "back_left_light": "back_light",
    "back_right_door": "back_door",
    "back_right_light": "back_light",
    "front_bumper": "front_bumper",
    "front_glass": "front_glass",
    "front_left_door": "front_door",
    "front_left_light": "front_light",
    "front_right_door": "front_door",
    "front_right_light": "front_light",
    "hood": "hood",
    "left_mirror": "mirror",
    "right_mirror": "mirror",
    "tailgate": "trunk_tailgate",
    "trunk": "trunk_tailgate",
    "wheel": "wheel",
    "_background_": "_background_",
}

CARPARTS_KAGGLE_PATH = "ruiite/car-parts-dataset"
CARDD_KAGGLE_PATH = "issamjebnouni/cardd"


def download_kaggle(dataset_path: str | Path, folder: str | Path):
    """Loads dataset from kaggle

    Args:
        dataset_path (str | Path): Link
        folder (str | Path): Where to save

    Returns:
        _type_: Path to loaded dataset
    """
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    path = kagglehub.dataset_download(
        str(dataset_path), output_dir=str(folder)
    )
    return path


def remap_coco(input_json: str | Path, output_json: str | Path):
    """Merge classes in COCO annotation.

    Args:
        input_json (str | Path):
        output_json (str | Path):
    """
    with open(input_json, "r") as f:
        data = json.load(f)

    new_class_names = sorted(list(set(CLASS_MAPPING.values())))

    new_categories = [
        {"id": i + 1, "name": name, "supercategory": "car_part"}
        for i, name in enumerate(new_class_names)
    ]

    name_to_new_id = {cat["name"]: cat["id"] for cat in new_categories}

    old_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
    old_id_to_new_id = {}

    for old_id, old_name in old_id_to_name.items():
        if old_name in CLASS_MAPPING:
            new_name = CLASS_MAPPING[old_name]
            old_id_to_new_id[old_id] = name_to_new_id[new_name]
        else:
            old_id_to_new_id[old_id] = name_to_new_id.get("object", old_id)

    new_annotations = []
    for ann in data["annotations"]:
        if ann["category_id"] in old_id_to_new_id:
            ann["category_id"] = old_id_to_new_id[ann["category_id"]]
            new_annotations.append(ann)

    new_data = {
        "images": data["images"],
        "annotations": new_annotations,
        "categories": new_categories,
    }

    with open(output_json, "w") as f:
        json.dump(new_data, f, indent=4)

    print(f"Done! New categories count: {len(new_categories)}")


def process_carparts_to_yolo(path_raw: str | Path, path_result: str | Path):
    """Converts Carparts from coco format to YOLO.

    Args:
        path_raw (str | Path):
        path_result (str):
    """
    path_raw = Path(path_raw)
    # remap train and val sets: merge some classes
    path_remapped = path_raw / "remapped"
    path_remapped.mkdir(exist_ok=True)
    remap_coco(
        path_raw / "trainingset/trainingset/annotations.json",
        path_raw / "remapped/train.json",
    )
    remap_coco(
        path_raw / "testset/testset/annotations.json",
        path_raw / "remapped/val.json",
    )

    # convert to YOLO format
    convert_coco(
        labels_dir=str(path_remapped),
        save_dir=str(path_result),
        cls91to80=False,
        use_segments=True,
    )

    shutil.move(
        path_raw / "trainingset/trainingset/JPEGImages",
        Path(path_result) / "images/train",
    )
    shutil.move(
        path_raw / "testset/testset/JPEGImages",
        Path(path_result) / "images/val",
    )
    shutil.copy("configs/carparts.yaml", Path(path_result) / "carparts.yaml")
    # shutil.rmtree(path_raw)
    print("Carparts converted successfully!")


def process_cardd_to_yolo(path_raw: str | Path, path_result: str | Path):
    """Converts CarDD from coco format to YOLO.

    Args:
        path_raw (str | Path):
        path_result (str):
    """
    convert_coco(
        labels_dir=str(path_raw),
        save_dir=str(path_result),
        cls91to80=False,
        use_segments=True,
    )
    path_result = Path(path_result)
    path_raw = Path(path_raw)
    shutil.move(
        path_raw / "train",
        path_result / "images/train",
    )
    shutil.move(
        path_raw / "val",
        path_result / "images/val",
    )
    shutil.move(
        path_raw / "test",
        path_result / "images/test",
    )
    shutil.copy("configs/cardd.yaml", path_result / "cardd.yaml")
    # shutil.rmtree(path_raw)
    print("CarDD converted successfully!")


def merge_and_create_empty_masks(src_path, path_result, prefix="part_"):
    """Adds images from Carparts to CarDD as negatives without damaged parts.

    Args:
        src_path (_type_):
        path_result (_type_):
        prefix (str, optional):. Defaults to "part_".
    """
    src_root = Path(src_path)
    result_root = Path(path_result)

    splits = ["train", "val"]

    for split in splits:
        src_img_dir = src_root / "images" / split
        result_img_dir = result_root / "images" / split
        result_mask_dir = result_root / "labels" / split

        result_img_dir.mkdir(parents=True, exist_ok=True)
        result_mask_dir.mkdir(parents=True, exist_ok=True)

        if not src_img_dir.exists():
            print(f"{split} folder not found")
            continue

        for img_path in src_img_dir.iterdir():
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                new_filename = f"{prefix}{img_path.name}"
                new_img_path = result_img_dir / new_filename

                shutil.copy2(img_path, new_img_path)

                mask_filename = Path(new_filename).stem + ".txt"
                with open(result_mask_dir / mask_filename, "w") as _:
                    pass


def load_and_process_datasets(data_folder="data"):
    """Runs all preprocess steps.

    Args:
        data_folder (str, optional): Defaults to "data".
    """
    data_folder = Path(data_folder)
    path_carparts = download_kaggle(
        CARPARTS_KAGGLE_PATH, data_folder / "carparts"
    )
    print("Carparts loaded")
    process_carparts_to_yolo(
        path_carparts, data_folder / "carparts_yolo_annotation"
    )

    path_cardd = download_kaggle(CARDD_KAGGLE_PATH, data_folder / "cardd")
    print("CarDD loaded")
    process_cardd_to_yolo(
        path_cardd, data_folder / "cardd_merged_yolo_annotation"
    )
    merge_and_create_empty_masks(
        data_folder / "carparts_yolo_annotation",
        data_folder / "cardd_merged_yolo_annotation",
    )


if __name__ == "__main__":
    load_and_process_datasets("data")
