import librosa as lr
import numpy as np
from scipy.ndimage.filters import uniform_filter
from matplotlib import pyplot as plt
import pyworld as pw
import torch


sr = 22050
n_mels = 6
hop_length = 1024


def save(d, fn="out"):
    lr.output.write_wav(f"out/{fn}.wav", d, sr, norm=True)


def invlogamplitude(S):
    return 10.0 * (S/10.0)


def main(x):
    x, sr_ = lr.core.load(x)
    x = lr.core.resample(x, sr_, sr)
    x = x.astype(np.float64)

    f0, sp, ap = pw.wav2world(x, sr)
    f0[f0 != 0] = 150
    x = pw.synthesize(f0, sp, ap, sr)

    for i in range(5):
        f0, sp, ap = pw.wav2world(x, sr)
        sp *= ap
        ap = np.ones_like(ap)
        x = pw.synthesize(f0, sp, ap, sr)

    save(x, "target")


main("in/hareme.wav")
