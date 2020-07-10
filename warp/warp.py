import librosa
import pyworld as pw
import numpy as np
from synth import synthesize


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
    c = freq(c, content)
    save(c, "result")


def warp(a, b):
    a_mfcc = mfcc(a)
    b_mfcc = mfcc(b)
    _, wp = librosa.sequence.dtw(b_mfcc.T, a_mfcc.T)
    a_spec = stft(a)
    b_spec = np.zeros_like(stft(b))
    b_spec[wp[:, 0]] = a_spec[wp[:, 1]]
    c = istft(b_spec)
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
    main("mebama", "hellom", "What we've said consistently is that there has to"
         "be a political settlement to bring about genuine peace in the region.")
