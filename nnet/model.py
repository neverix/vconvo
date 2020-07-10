import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
import numpy as np
import random
from prep import seed, voice_encoder
import resemblyzer
import resemblyzer.hparams
import gc


device = "cuda"
batch_size = 8
test_batch_size = 8
embedding_size = 256
bottleneck_size = 128
n_layers = 3
total_downsample = 16
pre_downsample = 4
post_downsample = 4
n_epochs = 500
lr = 1e-4
gru_hidden = 512
num_mels = resemblyzer.hparams.mel_n_channels
recon_coeff = 2/3
voice_coeff = 1/3


def main(out_path: Path, train_path: Path, test_path: Path):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_dl = DataLoader(LSDataset(train_path), batch_size, collate_fn=LSDataset.collate_fn, shuffle=True)
    test_dl = DataLoader(LSDataset(test_path), test_batch_size, collate_fn=LSDataset.collate_fn, shuffle=True)
    model = AutoEncoder().to(device)
    criterion = nn.MSELoss()
    voice_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(n_epochs):
        print(f"{i+1}/{n_epochs}")
        gc.collect()

        loss_, total = 0, 0
        for x, v in train_dl:
            model.train()
            voice_encoder.train()
            optimizer.zero_grad()
            x = x.detach().to(device)
            v = v.detach().to(device)

            x_ = model(x, v)
            recon_loss = criterion(x, x_)

            v_ = v.roll(1, 0)
            x_ = model(x, v_)
            v__ = voice_encoder(x_)
            voice_loss = voice_criterion(v_, v__)

            loss = recon_loss * recon_coeff + voice_loss * voice_coeff
            loss.backward()
            optimizer.step()

            loss_ += float(loss)
            total += recon_coeff + voice_coeff
            print(f"\rloss: {loss_ / total}", end="")
        print()

        loss, total = 0, 0
        with torch.no_grad():
            for x, v in test_dl:
                model.eval()
                voice_encoder.eval()
                x = x.to(device)
                v = v.to(device)

                x_ = model(x, v)
                recon_loss = criterion(x, x_)

                v_ = v.roll(1, 0)
                x_ = model(x, v_)
                v__ = voice_encoder(x_)
                voice_loss = voice_criterion(v_, v__)

                loss += float(recon_loss * recon_coeff + voice_loss * voice_coeff)
                total += recon_coeff + voice_coeff
        print(f"val loss: {loss / total}")
        print()

        torch.save(model.state_dict(), out_path / "model.pt")


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, v):
        e = self.encoder(x)
        e = self.embedding(e, v)
        x = self.decoder(e)
        return x

    def embedding(self, e, v):
        v = v.unsqueeze(1).repeat(1, e.shape[1], 1)
        x = torch.cat((e, v), -1)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(num_mels, gru_hidden // 2, n_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(gru_hidden, bottleneck_size)

    def forward(self, x):
        x = x[:, ::pre_downsample, :]
        x, _ = self.gru(x)
        x = F.relu(self.linear(x))
        x = x[:, ::post_downsample, :]
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.gru = nn.GRU(bottleneck_size + embedding_size, gru_hidden // 2, n_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(gru_hidden, num_mels)

    def forward(self, x):
        x = x.repeat(1, pre_downsample, 1)
        x, _ = self.gru(x)
        x = F.relu(self.linear(x))
        x = x.repeat(1, post_downsample, 1)
        return x


class LSDataset(Dataset):
    def __init__(self, prep_path: Path):
        self.prep_path = prep_path
        index = pickle.load((prep_path / "index.pkl").open('rb'))
        idx = []
        for embedding, paths in index:
            for path in paths:
                idx.append((embedding, path))
        self.index = idx

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        embedding, path = self.index[idx]
        wav = np.load(path)
        mel = resemblyzer.wav_to_mel_spectrogram(wav)
        return torch.from_numpy(mel), torch.from_numpy(embedding)

    @staticmethod
    def collate_fn(batch):
        batch_x, batch_v = [torch.nn.utils.rnn.pad_sequence(list(seq), batch_first=True) for seq in zip(*batch)]
        length = batch_x.shape[1]
        length -= length % total_downsample
        batch_x = batch_x[:, :length, :]
        return batch_x, batch_v


if __name__ == '__main__':
    main(Path("data/prep"), Path("data/prep/train"), Path("data/prep/test"))
