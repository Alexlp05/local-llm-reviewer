from __future__ import annotations

from pathlib import Path
from torch.utils.data import Dataset, Subset

import numpy as np
import torch


START, END, PAD = "<", ">", " "


class CharNamesDataset(Dataset):
    def __init__(self, path: Path, max_name_length: int) -> None:
        super().__init__()
        self.max_name_length = max_name_length
        self.path = path
        self.raw = self.path.read_text()
        self.names = [name.lower().strip() for name in self.raw.split("\n") if name]
        self.vocab = sorted(list(set("".join(self.names)))) + [START, END, PAD]
        self.stoi = { c: i for i, c in enumerate(self.vocab) }
        self.itos = { i: c for i, c in enumerate(self.vocab) }

    def split(self, frac: float = 0.1) -> tuple[Dataset, Dataset, Dataset]:
        split = int(np.floor(frac * len(self)))
        idxs = np.random.choice(range(len(self)), size=len(self), replace=False)
        train_set = Subset(self, idxs[2 * split:])
        valid_set = Subset(self, idxs[split:2 * split])
        test_set = Subset(self, idxs[:split])
        return train_set, valid_set, test_set

    def encode(self, name: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in name], dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        return "".join(self.itos[int(t.item())] for t in tokens)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.names[idx][:self.max_name_length]
        tokens = self.encode(f"{START}{name}{END}")
        x = torch.full((self.max_name_length + 2, ), self.stoi[PAD], dtype=torch.long)
        y = torch.full((self.max_name_length + 2, ), self.stoi[PAD], dtype=torch.long)
        x[:len(tokens)] = tokens
        y[:len(tokens)-1] = tokens[1:]
        y[len(tokens):] = -1
        return x, y

    def __repr__(self) -> str:
        return f"CharNamesDataset<{self.path=:}, {self.max_name_length=:}>"

    def __str__(self) -> str:
        def fix(x: torch.Tensor) -> torch.Tensor:
            x[x == -1] = self.stoi[PAD]
            return x

        return (
            f"{repr(self)}\n"
            f"-----------------------------------\n"
            f"number of samples: {len(self.names)}\n"
            f"longest name: {max(map(len, self.names))}\n"
            f"vocab size: {len(self.vocab)}\n"
            f"vocab: {''.join(self.vocab)}\n"
            f"-----------------------------------\n"
            f"name: 'test'\n"
            f"encoded tokens: {self.encode('test').tolist()}\n"
            f"decoded name: '{self.decode(self.encode('test'))}'\n"
            f"-----------------------------------\n"
            f"x: {list(self[0][0].shape)} {fix(self[0][0]).tolist()} '{self.decode(fix(self[0][0]))}'\n"
            f"y: {list(self[0][1].shape)} {fix(self[0][1]).tolist()} '{self.decode(fix(self[0][1]))}'\n"
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader


    dataset = CharNamesDataset(Path("./data/names.txt"), 10)
    print(dataset)
