"""
This is the dataloader module for the Sprites dataset.
Generated with scripts in ./dataset/sprites/
"""

import os
import h5py
import random
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

S_LIST = []

C_LIST = [
    "Walk (Front)",
    "Walk (Left)",
    "Walk (Right)",
    "Spellcard (Front)",
    "Spellcard (Left)",
    "Spellcard (Right)",
    "Slash (Front)",
    "Slash (Left)",
    "Slash (Right)",
]


def random_color_jitter(vid, brightness, contrast, saturation, hue):
    """
    color jitter for a sequence of images
    """
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


class SpritesActionH5Dataset(Dataset):
    def __init__(
        self,
        data_path,
        n_fragments,
        fragment_len,
        portion=1,
        c_list=C_LIST,
        s_list=S_LIST,
    ):
        """
        data_path: path of the .hdf5 file, with or without the extension
        n_fragments: number of fragments to cut from each sample (for this dataset, it should be 2)
        fragment_len: both the width and height of the resized image
        portion: portion of the dataset to use
        """
        self.data_dir = data_path
        self.n_fragments = n_fragments
        self.fragment_len = fragment_len
        self.portion = portion
        self.c_list = c_list
        self.s_list = s_list

        if not self.data_dir.endswith(".hdf5"):
            self.data_dir += ".hdf5"
        assert os.path.exists(self.data_dir)

        # load all data into RAM
        self.all_data = []  # every element is a video (list of [video, content label, style label])
        with h5py.File(self.data_dir, "r") as hdf5:
            for _, group in tqdm(list(hdf5.items())):
                seq = group["seq"][:]
                content_label = group["c"][:]
                style_label = group["s"][:]

                self.all_data.append([seq, content_label, style_label])

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        seq, content_labels, style_labels = self.all_data[idx]

        # randomly pick n_fragments fragments from the images if there are more than n_fragments in the sequence
        n_images = len(seq)
        if n_images > self.n_fragments:
            n_fragments = self.n_fragments
            selected_indices = random.sample(range(n_images), n_fragments)
            selected_images = np.array([seq[i] for i in selected_indices])
            selected_content_labels = np.array(
                [content_labels[i] for i in selected_indices]
            )
            selected_style_labels = np.array(
                [style_labels[i] for i in selected_indices]
            )
        else:
            selected_images = seq
            selected_content_labels = content_labels
            selected_style_labels = style_labels

        # transform the images
        selected_images = torch.tensor(selected_images)
        selected_images = random_color_jitter(selected_images, 0.01, 0.01, 0.2, 0.5)

        # normalize
        selected_images = selected_images.float() / 255

        selected_content_labels = torch.tensor(selected_content_labels)

        return selected_images, selected_content_labels, selected_style_labels


def get_dataloader(
    data_path,
    batch_size,
    n_fragments=3,
    fragment_len=32,
    num_workers=0,
    portion=1,
    shuffle=True,
):
    dataset = SpritesActionH5Dataset(
        data_path=data_path,
        n_fragments=n_fragments,
        fragment_len=fragment_len,
        portion=portion,
        c_list=C_LIST,
        s_list=S_LIST,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataloader
