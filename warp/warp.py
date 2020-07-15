import librosa
import pyworld as pw
import numpy as np
from synth import synthesize
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


sr = 22050
hop_length = 128
win_length = 256
n_fft = 4096
n_mfcc = 64


def main(content, voice, text):
    content = read(content)
    voice = read(voice)
    voice, sr_ = synthesize(voice, text)
    voice = librosa.resample(voice, sr_, sr)

    save(content, "content")
    save(voice, "voice")

    c = warp(voice, content)
    c = vol(c, content)
    c = freq(c, content)
    save(c, "result")


def warp(a, b):
    a_mfcc = mfcc(a)
    b_mfcc = mfcc(b)
    _, wp = librosa.sequence.dtw(a_mfcc.T, b_mfcc.T)
    a_spec = stft(a)[:a_mfcc.shape[0]]
    b_spec = stft(b)[:b_mfcc.shape[0]]
    b_spec[wp[:, 1]] = a_spec[wp[:, 0]]
    return istft(b_spec)


def freq(a, b):
    f0, sp, ap = pw_extract(a)
    f0_source, *_ = pw_extract(b)
    f0_source[f0_source != 0] -= np.median(f0_source[f0_source != 0])
    f0 = (f0_source + np.median(f0[f0 != 0]))[:len(f0)] * (f0 != 0)
    f0 = f0[:len(sp)]
    sp = sp[:len(f0)]
    ap = ap[:len(f0)]
    c = pw_synth(f0, sp, ap)
    return c


def vol(a, b):
    a_stft = stft(a)
    b_stft = stft(b)
    c_stft = a_stft / a_stft.sum(axis=0) * b_stft.sum(axis=0)
    return istft(c_stft)


def stft(x):
    return librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length).T


def istft(x):
    return librosa.istft(x.T, hop_length=hop_length, win_length=win_length)


def mfcc(x):
    f0, sp, ap = pw_extract(x)
    f0[f0 != 0] = 150
    y = pw_synth(f0, sp, ap)
    mfcc = librosa.feature.mfcc(y, sr, n_mfcc=n_mfcc, hop_length=hop_length, win_length=win_length)
    return mfcc.T[:stft(x).shape[0]]


def pw_extract(x):
    x = x.astype(np.float64)
    f0, t = pw.harvest(x, sr)
    f0 = pw.stonemask(x, f0, t, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    return f0, sp, ap


def pw_synth(f0, sp, ap):
    y = pw.synthesize(f0, sp, ap, sr)
    y[np.isnan(y)] = 0
    y[np.isinf(y)] = 0
    return y


def read(fn="in", full_name=False):
    if not full_name:
        fn = f"in/{fn}.wav"
    x, sr_ = librosa.core.load(fn)
    x = librosa.core.resample(x, sr_, sr)
    return x


def save(d, fn="out"):
    librosa.output.write_wav(f"out/{fn}.wav", d, sr, norm=True)


if __name__ == '__main__':
    main("mebama", "hellom", "What we've said consistently is that there has to "
         "be a political settlement to bring about genuine peace in the region.")
