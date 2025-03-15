"""
Totally successing the in-distribution dataloader, but change the styles
"""

from torch.utils.data import DataLoader

import dataloader.phonenums_dataloader
from dataloader.phonenums_dataloader import PhoneNumsDataset, C_LIST

S_LIST = [
    "pink",
    "salmon",
    "gold",
    "lime",
    "cyan",
    "magenta",
    "gray",
    "peru",
]

C_LIST = dataloader.phonenums_dataloader.C_LIST


def get_dataloader(
    data_dir,
    batch_size,
    n_fragments=2,
    fragment_len=32,
    num_workers=0,
    portion=1,
    shuffle=True,
):
    dataset = PhoneNumsDataset(
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
