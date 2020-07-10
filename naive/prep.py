import librosa as lr
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import pyworld as pw
import torch


sr = 22050
n_fft = 2048
hop = 22
hop = sr // hop
win = hop * 2
n_bands = 14
fmin = 100
fstep = 800


def save(d, fn="out"):
    lr.output.write_wav(f"out/{fn}.wav", d, sr, norm=True)


def main(x):
    x, sr_ = lr.core.load(x)
    x = lr.core.resample(x, sr_, sr)
    x = x.astype(np.float64)

    f0, sp, ap = pw.wav2world(x, sr)
    f0_orig = f0.copy()
    f0[f0 != 0] = 150
    x = pw.synthesize(f0, sp, ap, sr)

    s = lr.stft(x, n_fft, hop_length=hop, win_length=win)
    s = np.abs(s)

    f = lr.fft_frequencies(sr, n_fft)
    def spec_band(freq):
        return np.argmin(np.abs(f - freq))
    s = s.T

    masks = np.zeros((n_bands,) + s.shape)
    bands = np.zeros((n_bands, s.shape[0]))
    for i in range(n_bands):
        f_start = fmin + fstep * i
        f_end = f_start + fstep
        band_start = spec_band(f_start)
        band_end = spec_band(f_end)

        mask = np.zeros_like(s)
        mask[:, band_start:band_end] = 1
        masks[i] = mask

        band = s * mask
        band = np.mean(band, axis=-1)
        bands[i] = band

    s = np.zeros_like(s)
    for i in range(n_bands):
        mask = masks[i]
        band = bands[i]
        s += mask * band[:, None]
    s = s.T

    y = lr.griffinlim(s, hop_length=hop, win_length=win)
    save(y, "target")


main("in/hareme.wav")
