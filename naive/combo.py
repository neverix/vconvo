import soundfile as sf
import pyworld as pw
import numpy as np
import librosa as lr
from scipy.signal import wiener, medfilt2d
from matplotlib import pyplot as plt


sr = 44100
n_harmonics = 12
fmin = 100
fmax = 130
n_fft = 4096


def save(d, fn="out"):
    sf.write(f"out/{fn}.wav", d, sr)


def main(x):
    global sr
    x, sr = sf.read(x)
    if len(x.shape) > 1:
        x = x[:, 0].copy(order="C")
    save(x, "source")

    f0, t = pw.dio(x, sr)
    f0 = pw.stonemask(x, f0, t, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)

    # sp_n = ap * sp
    # ap_n = np.ones_like(ap)

    # sp_v = sp * (1 - ap)
    # ap_v = np.zeros_like(ap)

    # sp_v = np.tile(np.mean(sp_v, axis=0), (sp_v.shape[0], 1))
    # f0[f0 != np.nan] = 120

    # sp_v *= np.linspace(0.1, 1000, num=sp_v.shape[1]).T / 1000
    # f0 += 120

    # sp_v *= np.linspace(45, 0.0001, numscipy.ndimage.median_filter¶=sp_v.shape[1]).T / 60
    # f0 -= 45

    f0[f0 != 0] = 150
    sp *= ap
    ap = np.zeros_like(ap)
    sp = wiener(sp, mysize=13)
    sp = medfilt2d(sp, (1, 21))

    # y_n = pw.synthesize(f0, sp_n, ap_n, sr)
    # y_v = pw.synthesize(f0, sp_v, ap_v, sr)
    # y = y_n[:len(y_v)] + y_v[:len(y_n)]

    y = pw.synthesize(f0, sp, ap, sr)

    # save(y_n, "noise")
    # save(y_v, "voice")
    save(y, "target")


if __name__ == '__main__':
    main("in/hareme.wav")
