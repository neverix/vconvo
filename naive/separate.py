import librosa as lr
import numpy as np
from matplotlib import pyplot as plt
import pyworld as pw


sr = 22050
n_mels = 12
n_mfcc = 8
hop_length = 512


def save(d, fn="out"):
    lr.output.write_wav(f"out/{fn}.wav", d, sr, norm=True)


def main(x):
    x, sr_ = lr.core.load(x)
    x = lr.core.resample(x, sr_, sr)
    x = x.astype(np.float64)

    f0, sp, ap = pw.wav2world(x, sr)
    f0_orig = f0.copy()
    f0[f0 != 0] = 150

    sp_f = sp * ap
    ap_f = np.ones_like(ap)
    x = pw.synthesize(f0, sp_f, ap_f, sr)

    mels = lr.feature.melspectrogram(x, n_mels=n_mels, sr=sr, hop_length=hop_length)
    y_f = lr.feature.inverse.mel_to_audio(mels, sr=sr, hop_length=hop_length)

    sp_v = sp * (1 - ap) * np.clip((f0 != 0).astype(np.float32)[:, None], 0.001, np.inf)
    ap_v = np.zeros_like(ap)
    y_v = pw.synthesize(f0, sp_v, ap_v, sr)

    mfccs = lr.feature.mfcc(y_v, n_mfcc=n_mfcc, sr=sr, hop_length=hop_length)
    y_v = lr.feature.inverse.mfcc_to_audio(mfccs, sr=sr, hop_length=hop_length).astype(np.float64)
    save(y_v, "vocal")

    f0, sp_v, ap_v = pw.wav2world(y_v, sr)
    y_v = pw.synthesize(f0_orig[:len(sp_v)], sp_v, ap_v, sr)
    y = y_v[:len(y_f)] + y_f[:len(y_v)]
    save(y)
    save(y_f, "fricative")


main("in/hareme.wav")
