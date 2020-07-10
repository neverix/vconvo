import soundfile as sf
import pyworld as pw
import numpy as np
import librosa as lr
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

    sp_n = ap * sp
    ap_n = np.ones_like(ap)

    sp_v = sp * (1 - ap)
    ap_v = np.zeros_like(ap)

    # sp_v *= np.linspace(0.1, 1000, num=sp_v.shape[1]).T / 1000
    # f0 += 120

    # sp_v *= np.linspace(45, 0.0001, num=sp_v.shape[1]).T / 60
    # f0 -= 45

    y_n = pw.synthesize(f0, sp_n, ap_n, sr)
    y_v = pw.synthesize(f0, sp_v, ap_v, sr)

    save(y_n, "noise")
    save(y_v, "voice")

    s = lr.stft(y_v, n_fft=n_fft)
    # plt.imshow(np.abs(s).T[:, :200])
    # plt.show()
    if len(s.shape) > 2:
        s = s[0]
    pitches, magnitudes = lr.core.piptrack(S=s, sr=sr, fmin=fmin, fmax=fmax)
    pitch = pitches.argmax(axis=0).astype(np.float64)
    pitch[pitch == 0] = np.nan

    # plt.plot(pitch)
    # plt.show()

    harmonics = np.zeros((n_harmonics, s.shape[1], s.shape[0]), dtype=np.complex)
    masks = np.zeros((n_harmonics, s.shape[1], s.shape[0]), dtype=np.bool_)
    for harmonic in range(n_harmonics):
        period_start = np.floor(pitch * (harmonic + 0.5)).astype(np.int64).reshape((-1, 1))
        period_end = np.floor(pitch * (harmonic + 1.5)).astype(np.int64).reshape((-1, 1))
        mask, _ = np.mgrid[0:s.shape[0], 0:s.shape[1]]
        mask = np.logical_and(mask.T >= period_start, mask.T < period_end)
        masks[harmonic] = mask
        harmonics[harmonic] = s.copy().T * mask

    harmonics = np.sum(np.abs(harmonics), axis=-1)
    harmonics_total = np.sum(harmonics, axis=0)
    harmonics /= np.clip(harmonics_total, 1, np.inf)

    # plt.imshow(harmonics)
    # plt.show()

    spectrum, _ = np.mgrid[0:s.shape[0], 0:s.shape[1]]
    spectrum = spectrum.T * np.pi / pitch[:, None]
    spectrum = np.float_power(np.cos(spectrum) / 2 + 1, 1.5)

    # plt.imshow(spectrum)
    # plt.show()

    new_spectrum = np.zeros_like(spectrum)
    for harmonic in range(n_harmonics):
        mask = masks[harmonic]
        modification = harmonics[harmonic, :, None]
        modification = np.tile(modification, (1, mask.shape[1]))
        modification[np.logical_not(mask)] = 0
        new_spectrum += spectrum * modification
    spectrum = new_spectrum
    spectrum[spectrum == np.nan] = 0

    plt.imshow(spectrum)
    plt.show()

    y_v_t = lr.istft(spectrum)

    y = y_n[:len(y_v_t)] + y_v_t[:len(y_n)]
    save(y, "target")


if __name__ == '__main__':
    main("in/me.wav")
