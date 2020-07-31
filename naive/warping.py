import librosa
import pyworld as pw
import numpy as np
import matplotlib.pyplot as plt


sr = 22050
hop_length = 128
win_length = 256
n_fft = 4096
n_mfcc = 64


def main(content, voice_content, voice):
    content = read(content)
    voice_content = read(voice_content)
    voice = read(voice)

    save(content, "content")
    save(voice_content, "voice_content")
    save(voice, "voice")

    a, b = warp(voice_content, voice)

    save(content, "result")


def req


def warp(a, b):
    a = norm_pitch(a)
    b = norm_pitch(b)

    a_mfcc = mfcc(a)
    b_mfcc = mfcc(b)

    a_spec = stft(a)
    b_spec = stft(b)

    _, wp = librosa.sequence.dtw(a_mfcc.T, b_mfcc.T)
    a_spec = a_spec[wp[::-1, 0]]
    b_spec = b_spec[wp[::-1, 1]]
    return a_spec, b_spec


def norm_pitch(x):
    f0, sp, ap = pw_extract(x)
    f0[f0 != 0] = 150
    y = pw_synth(f0, sp, ap)
    return y


def vol(a, b_spec):
    a_spec = stft(a)
    c_spec = a_spec / a_spec.sum(axis=0) * b_spec.sum(axis=0)
    return c_spec


def freq(a, b, c):
    f0_source, *_ = pw_extract(a)
    f0, *_ = pw_extract(b)
    f0_target, sp, ap = pw_extract(c)
    f0_source[f0_source != 0] -= np.median(f0_source[f0_source != 0])
    f0 = (f0_source + np.median(f0[f0 != 0]))[:len(f0_target)]
    c = pw_synth(f0, sp, ap)
    return c


def stft(x):
    return librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length).T


def istft(s):
    return librosa.istft(s.T, hop_length=hop_length, win_length=win_length)


def imfcc(m):
    return librosa.feature.inverse.mfcc_to_audio(m.T, sr=sr, hop_length=hop_length, win_length=win_length)


def mfcc_spec(s):
    return mfcc(istft(s))


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
    main("hello", "mebama", "obama")
