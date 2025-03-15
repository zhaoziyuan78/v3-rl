import os
import json
import argparse
import numpy as np
from random import shuffle
from shutil import copyfile


def build_real_train_val_test(
    train_dir,
    test_dir,
    extra_dir,
    train_ann_path,
    test_ann_path,
    extra_ann_path,
    output_train_dir,
    output_val_dir,
    output_test_dir,
    output_train_ann_path,
    output_val_ann_path,
    output_test_ann_path,
    filter_out_single_digit=True,
    ratio=(0.8, 0.1, 0.1),
):
    with open(train_ann_path, "r") as f:
        train_ann = json.load(f)
    with open(test_ann_path, "r") as f:
        test_ann = json.load(f)
    with open(extra_ann_path, "r") as f:
        extra_ann = json.load(f)

    shuffle(extra_ann)
    shuffle(train_ann)
    shuffle(test_ann)

    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_val_dir, exist_ok=True)
    os.makedirs(output_test_dir, exist_ok=True)

    print("Building train set...")
    train_filepaths = []
    new_train_ann = []
    for ann in train_ann:
        filename = ann["name"]
        boxes = ann["boxes"]
        if filter_out_single_digit and len(boxes) == 1:
            continue
        else:
            train_filepaths.append(os.path.join(train_dir, filename))
            new_train_ann.append(ann)
    for ann in extra_ann[: int(len(extra_ann) * ratio[0])]:
        filename = ann["name"]
        boxes = ann["boxes"]
        if filter_out_single_digit and len(boxes) == 1:
            continue
        else:
            train_filepaths.append(os.path.join(extra_dir, filename))
            new_train_ann.append(ann)
            new_train_ann[-1]["name"] = "e" + new_train_ann[-1]["name"]

    print("Building val set...")
    val_filepaths = []
    new_val_ann = []
    for ann in test_ann[: int(len(test_ann) * 0.5)]:
        filename = ann["name"]
        boxes = ann["boxes"]
        if filter_out_single_digit and len(boxes) == 1:
            continue
        else:
            val_filepaths.append(os.path.join(test_dir, filename))
            new_val_ann.append(ann)
    for ann in extra_ann[
        int(len(extra_ann) * ratio[0]) : int(len(extra_ann) * (ratio[0] + ratio[1]))
    ]:
        filename = ann["name"]
        boxes = ann["boxes"]
        if filter_out_single_digit and len(boxes) == 1:
            continue
        else:
            val_filepaths.append(os.path.join(extra_dir, filename))
            new_val_ann.append(ann)
            new_val_ann[-1]["name"] = "e" + new_val_ann[-1]["name"]

    print("Building test set...")
    test_filepaths = []
    new_test_ann = []
    for ann in test_ann[int(len(test_ann) * 0.5) :]:
        filename = ann["name"]
        boxes = ann["boxes"]
        if filter_out_single_digit and len(boxes) == 1:
            continue
        else:
            test_filepaths.append(os.path.join(test_dir, filename))
            new_test_ann.append(ann)
    for ann in extra_ann[int(len(extra_ann) * (ratio[0] + ratio[1])) :]:
        filename = ann["name"]
        boxes = ann["boxes"]
        if filter_out_single_digit and len(boxes) == 1:
            continue
        else:
            test_filepaths.append(os.path.join(extra_dir, filename))
            new_test_ann.append(ann)
            new_test_ann[-1]["name"] = "e" + new_test_ann[-1]["name"]

    # copy files to build the new dataset
    print("Copying images...")
    for filepath in train_filepaths:
        if "extra" in filepath:
            filename = "e" + os.path.basename(filepath)
        else:
            filename = os.path.basename(filepath)
        copyfile(filepath, os.path.join(output_train_dir, filename))
    for filepath in val_filepaths:
        if "extra" in filepath:
            filename = "e" + os.path.basename(filepath)
        else:
            filename = os.path.basename(filepath)
        copyfile(filepath, os.path.join(output_val_dir, filename))
    for filepath in test_filepaths:
        if "extra" in filepath:
            filename = "e" + os.path.basename(filepath)
        else:
            filename = os.path.basename(filepath)
        copyfile(filepath, os.path.join(output_test_dir, filename))

    print("Building annotations...")
    with open(output_train_ann_path, "w") as f:
        json.dump(new_train_ann, f, indent=4)
    with open(output_val_ann_path, "w") as f:
        json.dump(new_val_ann, f, indent=4)
    with open(output_test_ann_path, "w") as f:
        json.dump(new_test_ann, f, indent=4)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/svhn")
    parser.add_argument("--save_dir", type=str, default="../data/svhn_plural_")

    args = parser.parse_args()
    data_dir = args.data_dir
    output_data_dir = args.save_dir

    train_ann_path = os.path.join(data_dir, "svhn_train_ann.json")
    test_ann_path = os.path.join(data_dir, "svhn_test_ann.json")
    extra_ann_path = os.path.join(data_dir, "svhn_extra_ann.json")

    build_real_train_val_test(
        os.path.join(data_dir, "train"),
        os.path.join(data_dir, "test"),
        os.path.join(data_dir, "extra"),
        train_ann_path,
        test_ann_path,
        extra_ann_path,
        os.path.join(output_data_dir, "train"),
        os.path.join(output_data_dir, "val"),
        os.path.join(output_data_dir, "test"),
        os.path.join(output_data_dir, "train/train_ann.json"),
        os.path.join(output_data_dir, "val/val_ann.json"),
        os.path.join(output_data_dir, "test/test_ann.json"),
    )
