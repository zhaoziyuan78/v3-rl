"""
Totally successed from the original dataloader, but the styles are changed
"""

from torch.utils.data import DataLoader

import dataloader.insnotes_dataloader
from dataloader.insnotes_dataloader import InsNotesDataset, C_LIST

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

C_LIST = dataloader.insnotes_dataloader.C_LIST


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
