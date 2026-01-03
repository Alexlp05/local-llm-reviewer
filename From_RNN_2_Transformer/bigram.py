from __future__ import annotations

from matplotlib.pyplot import xlabel, ylabel
import torch
import torch.nn as nn


class BiGram(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros((vocab_size, vocab_size), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits[x]


if __name__ == "__main__":
    from dataset import CharNamesDataset, END, PAD, START
    from itertools import cycle
    from pathlib import Path
    from torch.optim.adamw import AdamW
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import torch.nn.functional as F


    epochs = 10
    batch_size = 32
    lr = 5e-4
    valid_every_n_step = 500

    dataset = CharNamesDataset(Path("./data/names.txt"), 10)
    train_set, valid_set, test_set = dataset.split()
    print(dataset)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    bigram = BiGram(len(dataset.vocab))
    optim = AdamW(bigram.parameters(), lr=lr)

    bigram.train()
    train_iter = cycle(iter(train_loader))
    train_pbar = tqdm(range(epochs * len(train_loader)), desc="Train")
    valid_loss = np.finfo(np.float32).max
    for step in train_pbar:
        x, y = next(train_iter)
        logits = bigram(x)
        B, W, V = logits.size()
        loss = F.cross_entropy(logits.view(-1, V), y.view(-1), ignore_index=-1)
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)

        if step > 0 and step % valid_every_n_step == 0:
            with torch.inference_mode():
                valid_loss = 0
                for x, y in valid_loader:
                    logits = bigram(x)
                    B, W, V = logits.size()
                    valid_loss += F.cross_entropy(logits.view(-1, V), y.view(-1), ignore_index=-1).item()
                valid_loss /= len(valid_loader)

        train_pbar.set_postfix(loss=f"{loss.item():.2f}", valid_loss=f"{valid_loss:.2f}")

    with torch.inference_mode():
        test_loss = 0
        for x, y in test_loader:
            logits = bigram(x)
            B, W, V = logits.size()
            test_loss += F.cross_entropy(logits.view(B * W, V), y.view(B * W), ignore_index=-1).item()
        test_loss /= len(test_loader)
        print(f"test_loss={test_loss:.2f}")

    T = F.softmax(bigram.logits.data, dim=-1).numpy().T
    ax = sns.heatmap(T, xticklabels=dataset.vocab, yticklabels=dataset.vocab)
    ax.set_title("Bigram Transition Matrix -> Softmax(logits)")
    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()
    plt.savefig("bigram.logits.png")

    temp = 1.0
    top_k = 5
    while True:
        print("Name Generator (Bi-Gram)")
        print("-----------------------------------")
        user_input = input("q (quit) | m (manual) | a (auto): ")
        if user_input == "q": break
        if user_input == "m": user_input = input("Enter First Character (quit): ")
        if user_input == "a": user_input = START

        idx = dataset.encode(START if user_input == "" else user_input[0])
        tokens = torch.full((1, 1), idx.item(), dtype=torch.long)
        for _ in range(1, dataset.max_name_length):
            idxs = tokens[:, -1:]
            logits = bigram(idxs)[:, -1, :] / temp
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idxs = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, idxs), dim=1)

        name = dataset.decode(tokens.view(-1))
        name = name.replace(START, "")
        end = name.find(END)
        name = name[:end] if end > 0 else name

        print(f"tokens: {tokens.tolist()}")
        print(f"name: {name[:-1]}")
        print("-----------------------------------")
