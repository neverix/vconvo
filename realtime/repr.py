import librosa
import pyworld as pw
import numpy as np
from scipy.ndimage.filters import median_filter
import matplotlib.pyplot as plt


sr = 22050
hop_length = 256
win_length = 512
fmin = 50
fmax = 3000
min_db = -100
max_db = 20
n_fft = 4096


def main(source):
    source = read(source)
    save(source, "source")

    r = repr(source)
    t = irepr(r)
    save(t, "target")


def repr(x):
    f0, sp, ap = pw_extract(x)
    f0[f0 != 0] = 150
    source_same = pw_synth(f0, sp * (1 - ap), np.zeros_like(ap))
    source_spec = stft(source_same)

    source_noise = pw_synth(f0, sp * ap, np.ones_like(ap))
    source_noise_spec = librosa.amplitude_to_db(stft(source_noise), max_db)
    source_noise_spec = np.zeros_like(source_noise_spec)

    source_spec = librosa.amplitude_to_db(source_spec + source_noise_spec, max_db)
    source_spec = np.clip(source_spec, min_db, max_db)

    freqs = librosa.fft_frequencies(sr, n_fft)
    source_spec[:, :np.abs(freqs - fmin).argmin()] = min_db
    source_spec[:, np.abs(freqs - fmax).argmin():] = min_db

    source_spec = median_filter(source_spec, (2, 33))

    return source_spec


def irepr(r):
    target_spec = librosa.db_to_amplitude(r, ref=max_db).T
    target = librosa.griffinlim(target_spec, hop_length=hop_length, win_length=win_length)
    return target


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
    main("mebama")
