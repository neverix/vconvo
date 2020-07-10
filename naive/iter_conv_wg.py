import librosa
import numpy as np
from torch.optim.lbfgs import LBFGS
from matplotlib import pyplot as plt
import waveglow_vocoder
import torch


sr = 16000
content_weight = 1e-3
style_weight = 1e+3
n_iter = 400
wv = waveglow_vocoder.WaveGlowVocoder()


def main(source, target):
    source = read(source)
    target = read(target)

    save(source, "source")
    save(target, "target")

    with torch.no_grad():
        target = torch.from_numpy(target).cuda()
        target = wv.wav2mel(target)[0]
        source = torch.from_numpy(source).cuda()
        source = wv.wav2mel(source)[0]

    target_content = content(target).detach()
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

            source_content = content(target)
            source_style = style(target)

            content_loss = criterion(source_content, target_content)
            style_loss = criterion(source_style, target_style)
            loss = content_loss * content_weight + style_loss * style_weight

            i += 1
            loss_ = float(loss)

            loss.backward()
            return loss

        optim.step(closure)

        print(f"\r{i}/{n_iter} - loss: {float(loss_):0.5f}", end='')

    with torch.no_grad():
        target = wv.mel2wav(target.unsqueeze(0))
        target = target.clamp(0, 1)[0]
        target = target.detach().cpu().numpy()
        save(target, "result")


def content(x):
    c = x.unsqueeze(-1) * x.unsqueeze(-2)
    c = c.mean(0)
    # uncomment if you want a cool visualization of the gram matrix
    # plt.imshow(c.cpu()) and plt.show()
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
    return x


def save(d, fn="out"):
    librosa.output.write_wav(f"out/{fn}.wav", d, sr, norm=True)


if __name__ == '__main__':
    main("p4", "p6")
