import librosa
import pyworld as pw
import numpy as np
from synth import synthesize
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import optimize


sr = 22050
hop_length = 512
compare_hop_length = 4410
win_length = 1024
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
    n_samples = len(b)
    n_bins = n_samples // compare_hop_length
    samples = np.random.uniform(0, 1/n_bins, n_bins)
    b_mfcc = mfcc(b)

    def synth(samples):
        c = np.zeros(n_samples)
        samples = np.abs(samples)
        steps = samples / np.sum(samples) * len(a)
        a_start = b_start = 0
        b_step = compare_hop_length
        for a_step in steps:
            a_step = int(a_step)
            a_end = a_start + a_step
            b_end = b_start + b_step
            c[b_start:b_end] = librosa.effects.time_stretch(a[a_start:a_end], a_step / b_step)
            a_start += a_step
            b_start += b_step
        return c

    def cost(samples):
        c = synth(samples)
        c_mfcc = mfcc(c)
        distance = np.linalg.norm(b_mfcc - c_mfcc)
        return distance

    result, *_ = optimize.basinhopping(cost, samples, stepsize=0.01, disp=True)
    c = synth(result)
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
