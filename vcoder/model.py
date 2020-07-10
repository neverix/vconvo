from torch.utils.data import DataLoader
import data
import torch.nn as nn
import torch
import librosa
import torch.nn.functional as F
import shutil
from pathlib import Path


batch_size = 8
test_batch_size = 64
device = "cuda"
cpu = "cpu"
lr = 1e-4
n_epochs = 100
train_split = 0.8
out_dir = Path("out")
save_path = out_dir / "model.pt"
audio_dir = out_dir / "aud"
reuse = True


def save(d, fn="out"):
    librosa.output.write_wav(audio_dir / f"{fn}.wav", d, data.sr)


def main():
    if not reuse:
        shutil.rmtree(out_dir, ignore_errors=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

    dataset = data.dataset()
    len_train = int(len(dataset) * train_split)
    len_test = len(dataset) - len_train
    train, test = torch.utils.data.dataset.random_split(dataset, [len_train, len_test])
    train_dl = DataLoader(train, batch_size=batch_size, collate_fn=data.collate)
    test_dl = DataLoader(test, batch_size=test_batch_size, collate_fn=data.collate)

    model = Vocoder().to(device)
    if reuse:
        model.load_state_dict(torch.load(save_path))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(n_epochs):
        print(f"batch {i+1}/{n_epochs}")
        l = t = 0
        for s, s_, voiced, f0, v, *_ in train_dl:
            s = s.to(device)
            f0 = f0.float().to(device)
            voiced = voiced.float().to(device)
            v = torch.cat([i.unsqueeze(0) for i in v])
            v = v.float().to(device)

            model.train()
            optimizer.zero_grad()
            try:
                s_, _ = model(s, voiced, f0, v)
            except AssertionError:
                continue

            loss = criterion(s, s_)
            loss.backward()
            optimizer.step()

            l += float(loss)
            t += 1
            print(f"\rloss: {l / t}", end='')
        print()

        l = t = 0
        with torch.no_grad():
            for s, st, voiced, f0, v, y, spk, utr, *_ in test_dl:
                s = s.to(device)
                f0 = f0.float().to(device)
                voiced = voiced.float().to(device)
                v = torch.cat([i.unsqueeze(0) for i in v])
                v = v.float().to(device)

                model.eval()
                s_, _ = model(s, voiced, f0, v)

                loss = criterion(s, s_)
                l += float(loss)
                t += 1

                for s, s_, spk, utr in zip(s_.detach(), st.detach(), spk, utr):
                    s = s.to(cpu).numpy()
                    s_ = s_.to(cpu).numpy()
                    x = data.despectrogram(s)
                    x_ = data.despectrogram(s_)
                    save(x, f"{spk}_{utr}_orig")
                    save(x_, f"{spk}_{utr}_tgt")
        print(f"val loss: {l / t}")

        torch.save(model.state_dict(), save_path)
        print("model saved")


in_feats = data.n_bands
mid_feats = 32
pre_feats = 256
enc_kernel_size = 17
dropout = 0.2


class Vocoder(nn.Module):
    def __init__(self):
        super(Vocoder, self).__init__()
        self.encoder = nn.Conv1d(in_feats, mid_feats, enc_kernel_size, padding=enc_kernel_size // 2)
        self.i_n = nn.InstanceNorm1d(mid_feats)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(mid_feats + 2 + data.voice_bands, pre_feats, enc_kernel_size, padding=enc_kernel_size // 2),
            nn.BatchNorm1d(pre_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(pre_feats, pre_feats, enc_kernel_size, padding=enc_kernel_size // 2),
            nn.BatchNorm1d(pre_feats),
            nn.ReLU(),
            nn.ConvTranspose1d(pre_feats, in_feats, enc_kernel_size, padding=enc_kernel_size // 2)
        )

    def forward(self, s, voiced, f0, v):
        assert not torch.any(torch.isnan(v))
        s = s.transpose(1, 2)
        emb = self.encoder(s)
        emb = F.relu(emb)
        emb = self.i_n(emb)
        emb = F.dropout(emb, p=dropout)
        emb_ = torch.cat([emb, voiced.unsqueeze(1), f0.unsqueeze(1)], dim=1)
        emb_ = torch.cat([emb_, v.unsqueeze(-1).repeat(1, 1, emb.shape[-1])], dim=1)
        s_ = self.decoder(emb_)
        s_ = F.relu(s_)
        s_ = s_.transpose(1, 2)
        return s_, emb


if __name__ == '__main__':
    main()

