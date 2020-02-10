from torch.utils.data import Dataset
from pathlib import Path
import itertools
from rnn_examples.utils import unicodeToAscii
import string
import torch


class NamesDataset(Dataset):
    def __init__(self, data_dir: Path):

        self.idx2char = list(string.ascii_lowercase + "'")
        self.idx2lang = []
        self.lines = []
        self.labels = []
        for file in data_dir.glob("*.txt"):
            lang_name = file.name.split(".")[0]
            self.idx2lang.append(lang_name)

            for line in file.open("rt", encoding="utf8", errors="strict"):
                line = line.strip().lower()
                if len(line) == 0:
                    continue
                line = unicodeToAscii(line, allowed_letters=self.idx2char)
                self.lines.append(line)
                self.labels.append(lang_name)

        self.lang2idx = {lang: idx for idx, lang in enumerate(self.idx2lang)}
        self.char2idx = {char: idx for idx, char in enumerate(self.idx2char)}

        self.x = [[self.char2idx[c] for c in n] for n in self.lines]
        self.y = [self.lang2idx[l] for l in self.labels]

    def get_vocab_size(self):
        return len(self.char2idx)

    def get_n_classes(self):
        return len(self.lang2idx)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), self.y[idx]


if __name__ == "__main__":
    dset = NamesDataset(data_dir=Path("data/names"))
    print(dset[7])
    print(dset.lines[7])
