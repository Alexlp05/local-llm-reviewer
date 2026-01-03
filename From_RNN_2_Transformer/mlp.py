from __future__ import annotations

from matplotlib.pyplot import xlabel, ylabel
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, t_emb_dim: int, h_dim: int, ctx_size: int, vocab_size: int) -> None:
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.h_dim = h_dim
        self.ctx_size = ctx_size
        self.vocab_size = vocab_size

        self.t_emb = nn.Embedding(self.vocab_size, self.t_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.ctx_size * self.t_emb_dim, self.h_dim), nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.vocab_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_embs = []
        for c in range(self.ctx_size):
            t_emb = self.t_emb(t)
            t_embs.append(t_emb)
            t = torch.roll(t, 1, 1)
            t[:, 0] = self.vocab_size - 1
        t_embs = torch.cat(t_embs, dim=-1)
        return self.mlp(t_embs)


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

    ctx_size = 5
    t_emb_dim = 64
    h_dim = 64

    dataset = CharNamesDataset(Path("./data/names.txt"), 10)
    train_set, valid_set, test_set = dataset.split()
    print(dataset)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    mlp = MLP(t_emb_dim, h_dim, ctx_size, len(dataset.vocab))
    optim = AdamW(mlp.parameters(), lr=lr)

    mlp.train()
    train_iter = cycle(iter(train_loader))
    train_pbar = tqdm(range(epochs * len(train_loader)), desc="Train")
    valid_loss = np.finfo(np.float32).max
    for step in train_pbar:
        x, y = next(train_iter)
        logits = mlp(x)
        B, W, V = logits.size()
        loss = F.cross_entropy(logits.view(-1, V), y.view(-1), ignore_index=-1)
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)

        if step > 0 and step % valid_every_n_step == 0:
            with torch.inference_mode():
                valid_loss = 0
                for x, y in valid_loader:
                    logits = mlp(x)
                    B, W, V = logits.size()
                    valid_loss += F.cross_entropy(logits.view(-1, V), y.view(-1), ignore_index=-1).item()
                valid_loss /= len(valid_loader)

        train_pbar.set_postfix(loss=f"{loss.item():.2f}", valid_loss=f"{valid_loss:.2f}")

    with torch.inference_mode():
        test_loss = 0
        for x, y in test_loader:
            logits = mlp(x)
            B, W, V = logits.size()
            test_loss += F.cross_entropy(logits.view(B * W, V), y.view(B * W), ignore_index=-1).item()
        test_loss /= len(test_loader)
        print(f"test_loss={test_loss:.2f}")

    E = F.softmax(mlp.t_emb.weight.data, dim=-1).numpy().T
    ax = sns.heatmap(E, xticklabels=dataset.vocab, yticklabels=list(map(str, range(t_emb_dim))))
    ax.set_title("MLP Token Embedding")
    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()
    plt.savefig("mlp.t_emb.png")

    temp = 1.0
    top_k = 5
    while True:
        print("Name Generator (MLP)")
        print("-----------------------------------")
        user_input = input("q (quit) | m (manual) | a (auto): ")
        if user_input == "q": break
        if user_input == "m": user_input = input("Enter First Character (quit): ")
        if user_input == "a": user_input = START

        idx = dataset.encode(START if user_input == "" else user_input[0])
        tokens = torch.full((1, 1), idx.item(), dtype=torch.long)
        for _ in range(1, dataset.max_name_length):
            idxs = tokens if tokens.size(1) <= ctx_size else tokens[:, -ctx_size:]
            logits = mlp(idxs)[:, -1, :] / temp
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
