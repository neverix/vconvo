from torchaudio.datasets import VCTK
from torch.utils.data import DataLoader
import torchdata
import shutil
from pathlib import Path
import pyworld as pw
import numpy as np
import torch
import librosa
import random


data_path = "./data"
batch_size = 64
n_workers = 8
del_cache = False

n_stft = 512
n_bands = n_stft // 2 + 1
voice_bands = 1025
sr = 48000
min_level_db = -100
ref_level_db = 20
fmin = 80
fmax = 255

seed = 228
torch.random.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def normalize(S):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)


def denormalize(S):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db


def amp_to_db(x):
    return ref_level_db * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return np.power(10.0, x * 0.05)


def spectrogram(x):
    s = np.abs(librosa.stft(x, n_stft))
    s = amp_to_db(s) - ref_level_db
    return normalize(s).T


def despectrogram(x):
    s = denormalize(x.T)
    s = db_to_amp(s + ref_level_db)
    y = librosa.istft(s)
    return y


def ffill(arr):
    prev = np.arange(len(arr))
    prev[arr == 0] = 0
    prev = np.maximum.accumulate(prev)
    return arr[prev]


def transform(x):
    x = x[0].numpy().astype(np.float64)
    s_ = spectrogram(x)

    f0, sp, ap = pw.wav2world(x, sr)
    # i know this isn't necessary anymore, but i can't just let the 0_o disappear
    f0_o = f0.copy()  # 0_o
    f0[f0 != 0] = 150

    y = pw.synthesize(f0, sp, ap, sr)
    s = spectrogram(y)

    sp_v = sp * (1 - ap)
    sp_v = normalize(amp_to_db(sp_v))
    v = np.mean(sp_v[f0_o != 0], axis=0)

    pitches, magnitudes = librosa.piptrack(S=s.T, sr=sr, fmin=fmin, fmax=fmax)
    f0 = pitches.argmax(axis=0).astype(np.float64)
    f0 = np.flip(ffill(np.flip(f0)))
    f0 = ffill(f0)
    voiced = (f0 != 0).astype(np.float64)
    f0 -= np.mean(f0)
    if np.var(f0) != 0:
        f0 /= np.var(f0)

    return s, s_, voiced, f0, v, x


def dataset():
    vctk = VCTK("./data")
    vctk = torchdata.datasets.WrapDataset(vctk)
    vctk = vctk\
        .map(torchdata.maps.To(transform, 0))\
        .map(torchdata.maps.Flatten())
    vctk = vctk.map(torchdata.maps.ToAll(lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x))

    cache = Path("./data/cache")
    if del_cache and cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    vctk = vctk.cache(torchdata.cachers.Pickle(cache))

    vctk = vctk.map(lambda x: x if (x[1] != 0).any() else None)
    return vctk


def collate(batch):
    batch = [i for i in batch if i is not None]
    *x, v, y, _sr, _txt, spk, utr = zip(*batch)
    x = [torch.nn.utils.rnn.pad_sequence(list(i), batch_first=True) for i in x]
    s, s_, voiced, f0 = x
    return s, s_, voiced, f0, v, y, spk, utr


def main():
    data = dataset()
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collate, num_workers=n_workers)

    for i, _ in enumerate(dataloader):
        print(f"\rBatch {i + 1} done!", end='')
    print()
    print("Done!")


if __name__ == '__main__':
    main()
