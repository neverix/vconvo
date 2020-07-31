import librosa
import pyworld as pw
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from math import inf


sr = 22050
hop_length = 128
win_length = 256
n_fft = 256
n_mfcc = 4
max_iter = 1000
select_iter = 10


def main(content, source, voice):
    content = read(content)
    source = read(source)
    voice = read(voice)

    save(content, "content")
    save(source, "source")
    save(voice, "voice")

    a, b = align(source, voice)
    max_score = -inf
    best_model = None
    for i in range(select_iter):
        model = MLPRegressor(hidden_layer_sizes=tuple(), activation="identity", max_iter=max_iter)
        model.fit(a, b)
        score = model.score(a, b)
        print(score)
        if score > max_score:
            max_score = score
            best_model = model

    model = best_model
    content_spec = np.abs(stft(norm(content)))
    content = librosa.griffinlim(model.predict(content_spec).T, hop_length=hop_length, win_length=win_length)
    save(content, "target")


def align(a, b):
    a = norm(a)
    b = norm(b)

    a_mfcc = mfcc(a)
    b_mfcc = mfcc(b)

    a_stft = np.abs(stft(a))[:len(a_mfcc)]
    b_stft = np.abs(stft(b))[:len(b_mfcc)]

    _, wp = librosa.sequence.dtw(a_mfcc.T, b_mfcc.T)
    return a_stft[wp[:, 0]], b_stft[wp[:, 1]]


def norm(x):
    f0, sp, ap = pw_extract(x)
    f0[f0 != 0] = 150
    return pw_synth(f0, sp, ap)


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
    main("hareme", "mebama", "obama")
