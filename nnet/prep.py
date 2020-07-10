from pathlib import Path
import pickle
import random
import resemblyzer
import numpy as np


seed = 42
n_samples = 5
voice_encoder = resemblyzer.VoiceEncoder("cuda")
for param in voice_encoder.parameters():
    param.requires_grad = False


def main(dataset_path: Path, out_path: Path):
    random.seed(seed)
    np.random.seed(seed)

    out_wavs = out_path / "wavs"
    out_wavs.mkdir(parents=True, exist_ok=True)
    speakers = list(dataset_path.iterdir())
    index = []
    for i, speaker in enumerate(speakers):
        print(f"\r{i+1}/{len(speakers)}", end='')
        wavs = list(speaker.rglob("*.wav"))
        if len(wavs) > n_samples:
            wavs = random.sample(wavs, n_samples)
        paths = []
        wavs_rmb = [resemblyzer.preprocess_wav(wav) for wav in wavs]
        for wav, rmb in zip(wavs, wavs_rmb):
            path = out_wavs / wav.with_suffix(".npy").name
            np.save(path, rmb)
            paths.append(path)
        voice_embedding = voice_encoder.embed_speaker(wavs_rmb)
        index.append((voice_embedding, paths))
    pickle.dump(index, (out_path / "index.pkl").open('wb'))
    print()


if __name__ == '__main__':
    main(Path("data/ls/train"), Path("data/prep/train"))
    main(Path("data/ls/test"), Path("data/prep/test"))
