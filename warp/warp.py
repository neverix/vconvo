import librosa
import pyworld as pw
import numpy as np
from synth import synthesize
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


sr = 22050
hop_length = 64
win_length = 128
n_fft = 4096
n_mfcc = 16


def main(content, voice, text):
    content = read(content)
    voice = read(voice)
    voice, sr_ = synthesize(voice, text)
    voice = librosa.resample(voice, sr_, sr)

    save(content, "content")
    save(voice, "voice")

    c = warp(voice, content)
    # c = freq(c, content)
    save(c, "result")


def warp(a, b):
    a_stft = stft(a)
    b_stft = stft(b)
    a_mfcc = mfcc(a)
    b_mfcc = mfcc(b)

    dist = cdist(b_mfcc, a_mfcc)
    prev = np.zeros_like(dist, dtype=np.int32)
    for x in range(1, dist.shape[0]):
        for y in range(dist.shape[1]):
            sl = dist[x - 1, :y+1]
            dist[x, y] += sl.min()
            prev[x, y] = sl.argmin()

    c_stft = np.zeros_like(b_stft)
    x = dist.shape[0]
    y = dist[-1].argmin()
    while x > 0:
        x -= 1
        c_stft[x] = a_stft[y]
        y = prev[x, y]
    c = istft(c_stft)
    return c


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


def amplitude(a):
    return librosa.core.amplitude_to_db(np.abs(librosa.stft(a, hop_length=hop_length, win_length=win_length).sum(axis=0)))


def stft(x):
    return librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length).T


def istft(x):
    return librosa.istft(x.T, hop_length=hop_length, win_length=win_length)


def mfcc(x):
    return librosa.feature.mfcc(x, sr, n_mfcc=n_mfcc, hop_length=hop_length, win_length=win_length).T


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
