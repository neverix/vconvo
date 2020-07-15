import librosa
import pyworld as pw
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


sr = 22050
hop_length = 128
win_length = 256
n_fft = 4096
n_mfcc = 6


def main(source, voice):
    source = read(source)
    voice = read(voice)

    save(source, "source")
    save(voice, "voice")

    target = warp(source, voice)
    target = freq(source, voice, target)
    save(target, "target")


def warp(source, voice):
    voice_spec = stft(voice)

    source_feats = features(source)
    voice_feats = features(voice)

    voice_spec = voice_spec[:len(voice_feats)]
    voice_feats = voice_feats[:len(voice_spec)]

    dist = cdist(source_feats, voice_feats)
    target_spec = voice_spec[dist.argmin(axis=1)]
    target = istft(target_spec)

    return target


def freq(a, b, c):
    f0_source, *_ = pw_extract(a)
    f0, *_ = pw_extract(b)
    f0_target, sp, ap = pw_extract(c)
    f0_source[f0_source != 0] -= np.median(f0_source[f0_source != 0])
    f0 = (f0_source + np.median(f0[f0 != 0]))[:len(f0_target)]
    c = pw_synth(f0, sp, ap)
    return c


def features(x):
    return mfcc(norm_pitch(x))  # TODO


def norm_pitch(x):
    f0, sp, ap = pw_extract(x)
    f0[f0 != 0] = 150
    y = pw_synth(f0, sp, ap)
    return y


def mfcc(x):
    return librosa.feature.mfcc(x, sr, n_mfcc=n_mfcc, hop_length=hop_length, win_length=win_length).T


def stft(x):
    return librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length).T


def istft(x):
    return librosa.istft(x.T, hop_length=hop_length, win_length=win_length)


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
    main("hare", "p4")
