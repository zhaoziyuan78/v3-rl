"""
This is the dataloader module for the InsNotes music melody spectrogram dataset.
This dataset should have clear labels of pitches and clear labels of timbres.

The interesting thing about this study is that there are two notions of "batch".
The first "batch" is from different samples from the same global distribution. Though their styles might be in clusters.
In practice we use the standard "batch".
The second "batch" is from the same sample, but different fragments.
"""

import os
import random
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import FrequencyMasking

S_LIST = [
    "Soprano Sax",
    "Pipe Organ",
    "Accordion",
    "Viola",
    "Trumpet",
    "Muted Trumpet",
    "Oboe",
    "Clarinet",
    "Piccolo",
    "Pan Flute",
    "Harmonica",
    "Choir Aahs",
]

C_LIST = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]


class InsNotesDataset(Dataset):
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
        data_dir: directory directly containing .npy files
        n_fragments: number of fragments to cut from each sample
        fragment_len: length of each fragment (# windows. 16 windows at sr=16000 and hop_length=512 are about 0.5 seconds)
        portion: portion of the dataset to use
        """
        self.data_dir = data_dir
        self.n_fragments = n_fragments
        self.fragment_len = fragment_len
        self.portion = portion
        self.transform = FrequencyMasking(freq_mask_param=15)

        self.npy_paths = glob(os.path.join(data_dir, f"*.npy"))
        if portion != 1:
            random.shuffle(self.npy_paths)
            self.npy_paths = self.npy_paths[: int(len(self.npy_paths) * portion)]
            self.npy_paths.sort()

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        """
        For stft spectrogram, if using the dataset generating pipeline in "./dataset/insnotes/": at window_size=1024 and hop=512, there shouldn't be any overlap between fragments if fragment_len=32. The actual number of windows whose total length is equal to a note is 33. 1 / 16k * 512 * 33 = 1.056. The last window is not used.
        """
        npy_path = self.npy_paths[idx]
        npy_name = os.path.basename(npy_path)
        npy = np.load(
            npy_path
        )[
            :-1, :
        ]  # only when using vanilla stft: throw away the highest frequency bin, which is always near 0
        npy = torch.from_numpy(npy).float()

        # Cut n_fragments fragments from the sample. But should not be done randomly.
        fragments = []
        # there are two modes of cutting fragments.
        # 1. hard mode. cut n_fragment fragments all randomly, which means they are not consecutive
        # fragment_indices = np.random.choice(range(120), self.n_fragments, replace=False)
        # for i in fragment_indices:
        #     start_idx = (self.fragment_len+1) * i # +1 for the last window
        #     fragment = npy[:, start_idx : start_idx + self.fragment_len]
        #     fragments.append(fragment)
        # 2. easy mode. cut n_fragment fragments all consecutively.
        starting_fragment_idx = random.choice(range(120))
        fragment_indices = []
        for i in range(self.n_fragments):
            fragment_idx = starting_fragment_idx + i
            if fragment_idx >= 120:
                fragment_idx -= 120
            start_idx = (self.fragment_len + 1) * fragment_idx
            fragment = npy[:, start_idx : start_idx + self.fragment_len]
            fragments.append(fragment)
            fragment_indices.append(fragment_idx)
        fragment_indices = np.array(fragment_indices)
        # stack the fragments
        fragments = torch.stack(fragments, dim=0)

        s_idx = int(
            npy_name.split("_")[0][
                3:
            ]  # the data names are supposed to be like "ins001_000.npy"
        )
        s_labels = torch.tensor([s_idx for _ in range(self.n_fragments)])
        c_labels = torch.tensor(fragment_indices % 12)

        if self.transform is not None:
            fragments = self.transform(fragments)
            # in case of zeros, add a small number to avoid nan
            fragments[fragments == 0] = 1e-8

        return (
            fragments,
            c_labels,
            s_labels,
        )  # [n_fragments, n_fft, fragment_len], [n_fragments], [n_fragments]


def get_dataloader(
    data_dir,
    batch_size,
    n_fragments=16,
    fragment_len=32,
    num_workers=0,
    portion=1,
    shuffle=True,
):
    dataset = InsNotesDataset(
        data_dir=data_dir,
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
