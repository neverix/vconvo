import librosa
import pyworld as pw
import numpy as np
import matplotlib.pyplot as plt


sr = 22050
hop_length = 128
win_length = 256
n_fft = 4096
n_mfcc = 32


def main(content, voice):
    content = read(content)
    voice = read(voice)

    save(content, "content")
    save(voice, "voice")

    c_spec = warp(content, voice)
    c_spec = vol(content, c_spec)
    c = istft(c_spec)
    c = freq(content, c)
    save(c, "result")


def warp(a, b):
    a_mfcc = mfcc(a)
    b_mfcc = mfcc(b)
    _, wp = librosa.sequence.dtw(a_mfcc.T, b_mfcc.T)
    a_spec = np.zeros_like(stft(a))
    b_spec = stft(b)
    a_spec[wp[:, 0]] = b_spec[wp[:, 1]]
    return a_spec


def vol(a, b_spec):
    a_spec = stft(a)
    c_spec = a_spec / a_spec.sum(axis=0) * b_spec.sum(axis=0)
    return c_spec


def freq(a, b):
    f0_source, *_ = pw_extract(a)
    f0, sp, ap = pw_extract(b)
    f0_source[f0_source != 0] -= np.median(f0_source[f0_source != 0])
    f0 = (f0_source + np.median(f0[f0 != 0]))[:len(f0)] * (f0 != 0)
    f0 = f0[:len(sp)]
    sp = sp[:len(f0)]
    ap = ap[:len(f0)]
    c = pw_synth(f0, sp, ap)
    return c


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


def read(fn="in"):
    x, sr_ = librosa.core.load(f"in/{fn}.wav")
    x = librosa.core.resample(x, sr_, sr)
    return x


def save(d, fn="out"):
    librosa.output.write_wav(f"out/{fn}.wav", d, sr, norm=True)


if __name__ == '__main__':
    main("hellom", "hello")
