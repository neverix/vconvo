import librosa as lr
import numpy as np
from scipy.ndimage.filters import uniform_filter
from matplotlib import pyplot as plt
import pyworld as pw
import torch


sr = 22050
n_mfcc = 32
n_mels = 256
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
    f0_orig = f0.copy()
    f0[f0 != 0] = 150
    x = pw.synthesize(f0, sp, ap, sr)

    mfcc = lr.feature.mfcc(x, n_mfcc=n_mfcc, hop_length=hop_length)
    mfcc = mfcc[:8]
    y = lr.feature.inverse.mfcc_to_audio(mfcc, n_mels, hop_length=hop_length)

    save(y, "target")


main("in/hareme.wav")
