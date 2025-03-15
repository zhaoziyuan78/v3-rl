"""
This is the dataloader module for Librispeech segmented by phonemes (HDF5).
"""

import os
import h5py
import random
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

S_LIST = []
C_LIST = [
    "eh",  # 0
    "z",  # 1
    "s",  # 2
    "uw",  # 3
    "aw",  # 4
    "oy",  # 5
    "dx",  # 6
    "dh",  # 7
    "uh",  # 8
    "aa",  # 9
    "d",  # 10
    "p",  # 11
    "n",  # 12
    "ao",  # 13
    "ey",  # 14
    "hh",  # 15
    "y",  # 16
    "f",  # 17
    "r",  # 18
    "g",  # 19
    "v",  # 20
    "ah",  # 21
    "er",  # 22
    "ow",  # 23
    "sh",  # 24
    "b",  # 25
    "l",  # 26
    "k",  # 27
    "m",  # 28
    "ch",  # 29
    "ng",  # 30
    "t",  # 31
    "w",  # 32
    "ae",  # 33
    "iy",  # 34
    "th",  # 35
    "ay",  # 36
    "ih",  # 37
    "jh",  # 38
]  # so many phonemes are spoken in English


class LibrispeechH5Dataset(Dataset):
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
        data_path: path of the hdf5 file, with or without the extension
        n_fragments: number of fragments to cut from each sample (for this dataset, it refers to every phoneme)
        fragment_len: both the width and height of the resized image
        portion: portion of the dataset to use
        """
        self.data_path = data_path
        self.n_fragments = n_fragments
        self.fragment_len = fragment_len
        self.portion = portion
        self.c_list = c_list
        self.s_list = s_list

        if not self.data_dir.endswith(".hdf5"):
            self.data_dir += ".hdf5"
        assert os.path.exists(self.data_dir)

        # load the hdf5 file to get a shortcut to all data
        self.hdf5_file = h5py.File(self.data_dir, "r")
        self.index_to_dset = {}
        for i, (utt_id, dset) in enumerate(self.hdf5_file.items()):
            self.index_to_dset[i] = (utt_id, dset)

        print(f"Dataset size: {len(self.index_to_dset)}")

    def __len__(self):
        length = len(self.index_to_dset)
        if self.portion != 1:
            length = int(length * self.portion)
        return length

    def __getitem__(self, idx):
        """
        returns a tuple of:
        - fragments: a tensor of shape (n_fragments, fragment_len, fragment_len)
        - c_labels: a tensor of shape (n_fragments) containing the phoneme labels
        - s_label: a string containing the speaker id
        """
        utterance_id, dset = self.index_to_dset[idx]
        speaker_id = utterance_id.split("-")[0]
        phoneme_labels = dset.attrs["phonemes"]
        n_fragments = len(phoneme_labels)

        # randomly pick n_fragments fragments from the images
        if n_fragments >= self.n_fragments:
            n_fragments = self.n_fragments
            selected_start = random.choice(range(n_fragments - n_fragments + 1))
            selected_indices = list(range(selected_start, selected_start + n_fragments))
            selected_fragments = dset[selected_indices]
            selected_phonemes = [
                C_LIST.index(phoneme_labels[i]) for i in selected_indices
            ]
        # if there are not enough fragments, cycle until there are enough
        else:
            selected_indices = list(range(n_fragments))
            selected_fragments = dset[:]
            while len(selected_fragments) < self.n_fragments:
                selected_fragments = np.concatenate(
                    [selected_fragments, selected_fragments], axis=0
                )
                selected_indices += selected_indices
            selected_fragments = selected_fragments[: self.n_fragments]
            selected_indices = selected_indices[: self.n_fragments]
            selected_phonemes = [
                C_LIST.index(phoneme_labels[i]) for i in selected_indices
            ]

        # stack the fragments
        fragments = torch.tensor(np.stack(selected_fragments, axis=0))
        c_labels = torch.tensor(selected_phonemes)
        s_label = speaker_id

        return fragments, c_labels, s_label


def get_dataloader(
    data_dir,
    batch_size,
    n_fragments=100,
    fragment_len=32,
    num_workers=0,
    portion=1,
    shuffle=True,
    distributed=False,
):
    dataset = LibrispeechH5Dataset(
        data_dir=data_dir,
        n_fragments=n_fragments,
        fragment_len=fragment_len,
        portion=portion,
        c_list=C_LIST,
        s_list=S_LIST,
    )
    if not distributed:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=8 if num_workers > 0 else None,
        )
    else:
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=True,
            prefetch_factor=8 if num_workers > 0 else None,
        )
    return dataloader
