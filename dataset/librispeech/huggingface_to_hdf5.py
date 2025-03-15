"""
Use the phoneme level segmentation by MFA to segment the audio into chunks.
Source: https://huggingface.co/datasets/gilkeyio/librispeech-alignments
You can download it to a local directory using huggingface-cli.

How it's organized:
- .hdf5 file: partitions like train, val, test
- datasets: utterances, named with speaker_id
Each dataset is a sequence, whose first dimension is the number of phonemes in the utterance.
Labels of the phonemes are stored in the dataset attributes.

From pilot statistics we know that phoneme length is very long-tailed. We only consider the majority when stretching (l = 64 when hop_length = 80).
"""

import os
from datasets import load_dataset
from tqdm import tqdm
import h5py
import time
import numpy as np
import librosa as lr
import argparse
from PIL import Image
import multiprocessing
from multiprocessing import Pool, Queue, Process, Manager

PHONEME_LIST = [
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
]


def process_utterance(utt):
    phoneme_labels = []
    specs = []
    for phoneme in utt["phonemes"]:
        try:
            phoneme_label = phoneme["phoneme"]
            if phoneme_label[-1].isdigit():
                phoneme_label = phoneme_label[:-1]
            phoneme_label = phoneme_label.lower()

            if phoneme_label not in PHONEME_LIST:
                # print(f"Skipping {phoneme_label} in {utt['id']}")
                continue

            sr = 16000

            phoneme_audio = utt["audio"]["array"][
                int(float(phoneme["start"]) * sr) : int(float(phoneme["end"]) * sr)
            ]
            phoneme_audio = phoneme_audio / np.max(np.abs(phoneme_audio))

            phoneme_spec = lr.feature.melspectrogram(
                y=phoneme_audio, sr=16000, n_mels=80, hop_length=80, n_fft=256
            )
            phoneme_spec = lr.power_to_db(phoneme_spec, ref=np.max)

            phoneme_spec_img = Image.fromarray(phoneme_spec)
            phoneme_spec_img = phoneme_spec_img.resize((64, 80), Image.BICUBIC)
            phoneme_spec = np.array(phoneme_spec_img)

            phoneme_labels.append(phoneme_label)
            specs.append(phoneme_spec)

        except Exception as e:
            continue

    specs = np.stack(specs, axis=0)
    return (utt["id"], phoneme_labels, specs)


def worker_fn(task_q, result_q):
    import numpy as np
    import librosa as lr
    from PIL import Image

    while True:
        utt = task_q.get()  # blocking, until there is a task
        if utt is None:  # stopper
            break
        result = process_utterance(utt)
        result_q.put(result)


def write_one_result(hdf5, result_q, pbar=None):
    utt_id, phonemes, specs = result_q.get()
    if utt_id is None:
        return
    if utt_id not in hdf5.keys():
        dset = hdf5.create_dataset(
            utt_id,
            shape=(specs.shape[0], specs.shape[1], specs.shape[2]),
        )
        dset[:] = specs
        dset.attrs["phonemes"] = phonemes
    if pbar is not None:
        pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/librispeech_alignments/data",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../data/Librispeech100_h5",
    )
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    multiprocessing.set_start_method("spawn")

    # load the dataset
    dataset = load_dataset(args.data_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    for split in [args.split]:
        hdf5 = h5py.File(os.path.join(args.save_dir, f"{split.lower()}.hdf5"), "w")
        if split == "val":
            split = "validation"
        print(f"Processing {split} set")

        with Manager() as manager:
            task_q = manager.Queue()
            result_q = manager.Queue()

            workers = []
            for _ in range(8):
                w = Process(target=worker_fn, args=(task_q, result_q))
                w.start()
                workers.append(w)

            pbar = tqdm(total=len(dataset[split]))
            for utt in dataset[split]:
                subset = utt["subset"]
                if "clean" not in subset:
                    pbar.update(1)
                    continue
                elif "100" not in subset and split == "train":
                    pbar.update(1)
                    continue
                task_q.put(utt)
                while task_q.qsize() > 16:
                    write_one_result(hdf5, result_q, pbar)

            print("Waiting for workers to finish...")

            # wait for all tasks to finish
            for _ in workers:
                task_q.put(None)

            print("Waiting for workers to join...")

            for w in workers:
                w.join()

            print("Waiting for results to be written...")

            while not result_q.empty():
                write_one_result(hdf5, result_q, pbar)

            print("done!")
            print("leftover tasks:", task_q.qsize())
            print("leftover results:", result_q.qsize())
        hdf5.close()
