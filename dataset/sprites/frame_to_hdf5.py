"""
Collect frames from sprites dataset to sequences and save them to .h5py files.

The dataset is organized as follows:
.hdf5: partitions (train, val, test)
    groups: a group contains the data of a style
    datasets: list of [data, content labels, style labels]
"""

import os
from glob import glob
import random
import argparse

import numpy as np
import h5py
from PIL import Image
import torch
from torchvision import transforms

num_direction = {"front": 0, "left": 1, "right": 2, "back": 3}
n_class = 6
n_frames = 8
n_directions = 3


def random_color_jitter(vid, brightness, contrast, saturation, hue):
    if brightness > 0:
        brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
    else:
        brightness_factor = None
    if contrast > 0:
        contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
    else:
        contrast_factor = None
    if saturation > 0:
        saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
    else:
        saturation_factor = None
    if hue > 0:
        hue_factor = random.uniform(-hue, hue)
    else:
        hue_factor = None
    vid_transforms = []
    if brightness is not None:
        vid_transforms.append(
            lambda img: transforms.functional.adjust_brightness(img, brightness_factor)
        )
    if saturation is not None:
        vid_transforms.append(
            lambda img: transforms.functional.adjust_saturation(img, saturation_factor)
        )
    if hue is not None:
        vid_transforms.append(
            lambda img: transforms.functional.adjust_hue(img, hue_factor)
        )
    if contrast is not None:
        vid_transforms.append(
            lambda img: transforms.functional.adjust_contrast(img, contrast_factor)
        )
    random.shuffle(vid_transforms)
    for transform in vid_transforms:
        vid = transform(vid)

    return vid


def load_seq(path, labels):
    num = ""
    for i in range(len(labels)):
        num = num + str(labels[i])

    # return sequence and label
    seq = []
    for action in ["walk", "spellcard", "slash"]:
        for direction in ["front", "left", "right"]:
            filename = (
                action + "/" + direction + "_" + num + "_4" + ".png"
            )  # for identifiability, use the middle frame
            img = Image.open(os.path.join(path, filename))
            img = img.convert("RGB")
            seq.append(np.array(img).transpose(2, 0, 1))

    perm = np.random.permutation(len(seq))
    seq = [seq[i] for i in perm]
    content_label = perm

    style_label = [int(num) for i in range(len(seq))]

    return np.asarray(seq), np.array(content_label), np.array(style_label)


def save_h5py(data_dir, save_path):
    hdf5 = h5py.File(save_path, "w")

    for body in range(6):
        for bottom in range(6):
            for top in range(6):
                if "train" in data_dir:
                    for hair in range(8):
                        labels = [body, bottom, top, hair]
                        seq, c, s = load_seq(data_dir, labels)

                        group = hdf5.create_group(str(s[0]))
                        group.create_dataset("seq", data=seq)
                        group.create_dataset("c", data=c)
                        group.create_dataset("s", data=s)
                elif "val" in data_dir:
                    for hair in range(8, 9):
                        labels = [body, bottom, top, hair]
                        seq, c, s = load_seq(data_dir, labels)

                        group = hdf5.create_group(str(s[0]))
                        group.create_dataset("seq", data=seq)
                        group.create_dataset("c", data=c)
                        group.create_dataset("s", data=s)
                elif "test" in data_dir:
                    for hair in range(9, 10):
                        labels = [body, bottom, top, hair]
                        seq, c, s = load_seq(data_dir, labels)

                        group = hdf5.create_group(str(s[0]))
                        group.create_dataset("seq", data=seq)
                        group.create_dataset("c", data=c)
                        group.create_dataset("s", data=s)

    hdf5.close()


# def save_seq_ood():
#     load_path = "frames_ood/"
#     save_dir = "../../../data/SpritesOOD/test"

#     actions = ["walk", "spellcard", "slash"]
#     directions = ["front", "left", "right"]

#     for body in range(6, 7):
#         for bottom in range(6, 7):
#             for top in range(6, 7):
#                 for hair in range(0, 10):
#                     labels = [body, bottom, top, hair]
#                     seq, c, s = load_seq(load_path, labels)
#                     # directly add variations here
#                     seq_tensor = torch.tensor(seq)
#                     c_str = "".join([str(i) for i in c])
#                     for i in range(100):
#                         seq_var = random_color_jitter(seq_tensor, 0.01, 0.01, 0.2, 0.5)
#                         seq_var = seq_var.numpy()
#                         # save to npy
#                         save_path = os.path.join(
#                             save_dir,
#                             f"{c_str}_{body}{bottom}{top}{hair}_{str(i).zfill(2)}.npy",
#                         )
#                         np.save(save_path, seq_var)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/sprites_frames_")
    parser.add_argument("--save_dir", type=str, default="../data/Sprites_h5")
    args = parser.parse_args()

    data_dir = args.data_dir
    os.makedirs(args.save_dir, exist_ok=True)

    for partition in ["train", "val", "test"]:
        if os.path.exists(os.path.join(data_dir, partition)):
            save_path = os.path.join(args.save_dir, partition) + ".hdf5"
            save_h5py(os.path.join(data_dir, partition), save_path)
        else:
            pass
