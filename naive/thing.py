import librosa as lr
from numba import jit
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


n_harmonics = 10
fmin = 80
fmax = 160
sr = 44100


def save(d, fn="out"):
    lr.output.write_wav(f"out/{fn}.wav", d, sr)


def main(x):
    global sr
    x, sr = lr.core.load(x)
    x = x.astype(np.float64)
    s = lr.stft(x)

    freqs = np.arange(fmin, fmax + 1)
    masks = freq_mask(freqs)
    print(masks.shape)


def freq_mask(freq):
    period = np.floor_divide(sr, freq)
    sine = 0
    return period


main("in/me.wav")
