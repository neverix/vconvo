import librosa
import numpy as np
from scipy.spatial.distance import euclidean
from dtw import dtw
from matplotlib import pyplot as plt


sr = 22050
n_fft = 2048
n_mfcc = 16


def main(voice, self, content):
    # read audio files
    voice = read(voice)
    self = read(self)
    content = read(content)

    # dynamic time warping to align the time domain features
    mfcc_voice = mfcc(voice)
    mfcc_self = mfcc(self)
    *_, (path_self, path_voice) = dtw(mfcc_self, mfcc_voice, lambda a, b: np.linalg.norm(a - b))
    voice = voice.T[path_voice].T
    self = self.T[path_self].T

    # save before conversion for reference
    save(voice, "voice")
    save(self, "self")
    save(content, "content")

    # get the distance matrix
    dist = dist_mat(content, self)
    # uncomment if you want a cool visualization
    # plt.imshow(dist)
    # plt.show()

    # do the thing
    result = warp(dist, self)

    # save the result
    save(result, "result")


def warp(a, b):
    a = a[..., np.newaxis]
    b = b.T[np.newaxis, ...]
    c = (a * b).sum(axis=1).T
    return c


def dist_mat(a, b):
    a = mfcc(a)
    b = mfcc(b)
    dist = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    dist = np.linalg.norm(dist, axis=-1)
    dist /= dist.sum(axis=-1)[:, np.newaxis]
    return dist


def mfcc(x):
    return librosa.feature.mfcc(S=x, n_mfcc=n_mfcc).T


def read(fn="in"):
    x, sr_ = librosa.core.load(f"in/{fn}.wav")
    x = librosa.core.resample(x, sr_, sr)
    x = librosa.stft(x, n_fft)
    x = librosa.amplitude_to_db(np.abs(x))
    return x


def save(d, fn="out"):
    d = librosa.db_to_amplitude(d)
    d = librosa.griffinlim(d)
    librosa.output.write_wav(f"out/{fn}.wav", d, sr, norm=True)


if __name__ == '__main__':
    main("obama", "mebama", "change")
