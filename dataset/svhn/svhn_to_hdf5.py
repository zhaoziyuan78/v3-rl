"""
Convert the processed SVHN plural and cropped to HDF5 format.

How it's organized:
- .hdf5 file: partitions like train, val, test
- datasets: "data", "content_label", "style_label"
    - "data": [N, C, H, W] images
    - "content_label": [N] digit labels
    - "style_label": [N] origin image name
"""

import os
import argparse
import h5py
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/svhn_plural_cropped_")
    parser.add_argument("--save_dir", type=str, default="../data/SVHN_h5")

    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.save_dir

    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        print(f"Processing {split} set...")

        data = []
        content_label = []
        style_label = []

        for filepath in tqdm(glob(os.path.join(data_dir, split, "*.png"))):
            img = Image.open(filepath)
            img = img.convert("RGB")  # [C, H, W]
            img = np.array(img)

            data.append(img)

            filename = os.path.splitext(os.path.basename(filepath))[0]
            origin_name = filename.split("_")[0]
            digit_name = filename.split("_")[-1]

            style_label.append(origin_name)
            content_label.append(int(digit_name))

        with h5py.File(os.path.join(output_dir, f"{split}.hdf5"), "w") as f:
            f.create_dataset("data", data=data)
            f.create_dataset("content_label", data=content_label)
            f.create_dataset("style_label", data=style_label)
