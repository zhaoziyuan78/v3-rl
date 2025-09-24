import os
import random
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image


S_LIST = [
    "black",
    "blue",
    "green",
    "red",
    "teal",
    "purple",
    "orange",
    "brown",
]

C_LIST = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


class PhoneNumsRLDataset(Dataset):
    def __init__(
        self,
        data_dir,
        n_fragments,
        fragment_len,
        portion=1,
        c_list=C_LIST,
        s_list=S_LIST,
    ):
        """
        data_dir: directory directly containing .png files
        n_fragments: number of fragments to cut from each sample
        fragment_len: the width of every fragment in pixels
        portion: portion of the dataset to use
        c_list: the list of possible contents
        s_list: the list of possible styles
        """
        self.data_dir = data_dir
        self.n_fragments = n_fragments
        self.fragment_len = fragment_len
        self.portion = portion
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.c_list = c_list
        self.s_list = s_list

        self.png_paths = glob(os.path.join(data_dir, f"*.png"))
        if portion != 1:
            random.shuffle(self.png_paths)
            self.png_paths = self.png_paths[: int(len(self.png_paths) * portion)]
            self.png_paths.sort()

    def __len__(self):
        return len(self.png_paths)

    def __getitem__(self, idx):
        """
        Pick a sample from the dataset
        Since we know how the data is structured, we can randomly pick some fragments from the sample
        In this case, the fragments are well-defined segments, i.e., digits
        """
        png_path = self.png_paths[idx]
        png_name = os.path.basename(png_path)
        png_name = png_name.split(".")[0]

        c_labels = [int(digit) for digit in png_name.split("_")[0]]
        s_label = png_name.split("_")[1]
        s_label = self.s_list.index(s_label)

        # load the image
        img = Image.open(png_path)
        mtx = np.array(img)  # has to keep the uint8 type
        mtx = self.transform(mtx)

        return mtx


def get_dataloader(
    data_dir,
    batch_size,
    n_fragments=2,
    fragment_len=32,
    num_workers=0,
    portion=1,
    shuffle=True,
    distributed=False,
):
    dataset = PhoneNumsRLDataset(
        data_dir=data_dir,
        n_fragments=n_fragments,
        fragment_len=fragment_len,
        portion=portion,
        c_list=C_LIST,
        s_list=S_LIST,
    )
    if not distributed:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
        return dataloader
    else:
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
        )
        return dataloader
