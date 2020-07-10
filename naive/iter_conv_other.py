import librosa
import numpy as np
from torch.optim.lbfgs import LBFGS
from matplotlib import pyplot as plt
import torch


n_fft = 2048
sr = 22050
content_weight = 1
style_weight = 1e+3
n_iter = 200


def main(source, target, self):
    source = read(source)
    target = read(target)
    self = read(self)

    save(source, "source")
    save(target, "target")
    save(self, "self")

    target = torch.from_numpy(target).cuda()
    source = torch.from_numpy(source).cuda()
    self = torch.from_numpy(self).cuda()

    target_content = content(target, self).detach()
    target_style = style(source).detach()

    target = torch.nn.Parameter(target, requires_grad=True)

    optim = LBFGS([target])
    criterion = torch.nn.MSELoss()

    i = 0
    while i < n_iter:
        loss_ = 0

        def closure():
            nonlocal i, loss_
            optim.zero_grad()

            source_content = content(target, self)
            source_style = style(target)

            content_loss = criterion(source_content, target_content)
            style_loss = criterion(source_style, target_style)
            loss = content_loss * content_weight + style_loss * style_weight

            i += 1
            loss_ = float(loss)

            loss.backward()
            return loss

        optim.step(closure)

        print(f"\r{i}/{n_iter} - loss: {float(loss_):0.2f}", end='')

    target = target.detach().cpu().numpy()
    save(target, "result")


def content(x, y):
    c = x.unsqueeze(-1) * y.unsqueeze(-2)
    # uncomment if you want a cool visualization of the gram matrix
    # plt.imshow(c.mean(0).cpu()) and plt.show()
    return c


def style(x):
    x = x.transpose(0, 1)
    s = x.unsqueeze(-1) * x.unsqueeze(-2)
    s = s.mean(0)
    # same here
    # plt.imshow(s.cpu()) and plt.show()
    return s


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
    main("obama", "change", "mebama")
