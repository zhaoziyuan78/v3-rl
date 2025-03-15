import os
import argparse
from glob import glob
from tqdm import tqdm

import numpy as np
import librosa as lr


SR = 16000
N_FFT = 1024
HOP_LENGTH = 512


def process_dir(data_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    wav_paths = glob(os.path.join(data_dir, "*.wav"))
    for wav_path in tqdm(wav_paths):
        wav_name = os.path.basename(wav_path)
        npy_name = wav_name.replace(".wav", ".npy")
        y_t, _ = lr.load(wav_path, sr=SR)
        y_f = lr.stft(y_t, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mag_f = np.abs(y_f)
        log_mag_f = np.log(mag_f + 1e-5)
        np.save(os.path.join(save_dir, npy_name), log_mag_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/insnotes_wav_",
        help="Directory to load the audio files from",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../data/insnotes_npy_",
        help="Directory to save the generated npy files",
    )

    args = parser.parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir

    process_dir(data_dir, save_dir)
