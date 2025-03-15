import argparse
from glob import glob
import os
from tqdm import tqdm
import json

import PIL.Image as Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/svhn_plural_")
    parser.add_argument("--save_dir", type=str, default="../data/svhn_plural_cropped_")

    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.save_dir

    os.makedirs(output_dir, exist_ok=True)

    partitions = ["test", "val", "train"]

    for par in partitions:
        print("Processing", par)
        par_dir = os.path.join(data_dir, par)
        out_par_dir = os.path.join(output_dir, par)
        os.makedirs(out_par_dir, exist_ok=True)

        png_paths = glob(os.path.join(par_dir, "*.png"))
        ann_path = glob(os.path.join(par_dir, "*.json"))[0]
        ann = json.load(open(ann_path, "r"))
        ann_dict = {}
        for this_ann in ann:
            ann_dict[this_ann["name"]] = this_ann["boxes"]

        for png_path in tqdm(png_paths):
            name = os.path.basename(png_path)
            img = Image.open(png_path)
            img = img.convert("RGB")
            if name not in ann_dict:
                continue
            this_ann = ann_dict[name]
            for i, box in enumerate(this_ann):
                h, w, t, l = box["height"], box["width"], box["top"], box["left"]
                label = int(box["label"])
                if label == 10:
                    label = 0  # special case for SVHN
                img_box = img.crop((l, t, l + w, t + h))
                img_box = img_box.resize((32, 48))
                img_box.save(
                    os.path.join(out_par_dir, name.replace(".png", f"_{i}_{label}.png"))
                )
