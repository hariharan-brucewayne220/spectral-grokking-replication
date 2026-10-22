"""Full-batch SGD training on modular addition.

Two implementations:
  * `train_reference`: plain nn.Module + torch.optim.SGD, exactly as the paper describes.
  * `train_batched`: G copies of the MLP trained at once with hand-written gradients,
    one (lr, wd) pair per copy. Used for the sweeps; tested against the reference.

Both use MSE between logits and one-hot targets averaged over all n*p coordinates,
coupled L2 weight decay (PyTorch SGD `weight_decay`), and no momentum.
"""
import torch
import torch.nn as nn

from .data import ModAddData
from .model import init_weights, make_mlp


def train_reference(data: ModAddData, lr: float, wd: float, steps: int,
                    width: int = 256, seed: int = 0, eval_every: int = 100):
    Xtr, ytr, Xte, yte = data.split()
    p = data.p
    Ytr = nn.functional.one_hot(ytr, p).float()
    model = make_mlp(2 * p, width, p, seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0)
    tr_acc, te_acc = [], []
    for s in range(steps + 1):
        if s % eval_every == 0:
            with torch.no_grad():
                tr_acc.append((model(Xtr).argmax(1) == ytr).float().mean().item())
                te_acc.append((model(Xte).argmax(1) == yte).float().mean().item())
        if s == steps:
            break
        opt.zero_grad()
        nn.functional.mse_loss(model(Xtr), Ytr).backward()
        opt.step()
    return torch.tensor(tr_acc), torch.tensor(te_acc), model


def train_batched(data: ModAddData, lrs, wds, steps: int, width: int = 256,
                  seed: int = 0, eval_every: int = 100):
    """Train len(lrs) models in parallel. Returns (train_acc, test_acc), each [G, steps//eval_every + 1].

    The one-hot input means layer 1 is a lookup: h = W1[a] + W1[p + b].
    Its gradient is the matching scatter-add.
    """
    p = data.p
    lrs = torch.as_tensor(lrs, dtype=torch.float32).view(-1, 1, 1)
    wds = torch.as_tensor(wds, dtype=torch.float32).view(-1, 1, 1)
    G = lrs.shape[0]
    W1_0, W2_0 = init_weights(2 * p, width, p, seed)
    W1 = W1_0.expand(G, -1, -1).clone()
    W2 = W2_0.expand(G, -1, -1).clone()

    tr = data.train
    a_tr, b_tr, y_tr = data.a[tr], data.b[tr] + p, data.y[tr]
    a_te, b_te, y_te = data.a[~tr], data.b[~tr] + p, data.y[~tr]
    n = len(y_tr)
    Y = nn.functional.one_hot(y_tr, p).float()
    scale = 2.0 / (n * p)  # d/dF of mean((F - Y)^2)

    def acc(a, b, y):
        h = torch.relu(W1[:, a] + W1[:, b])
        return ((h @ W2).argmax(-1) == y).float().mean(-1)

    n_eval = steps // eval_every + 1
    tr_acc = torch.empty(G, n_eval)
    te_acc = torch.empty(G, n_eval)
    with torch.no_grad():
        for s in range(steps + 1):
            if s % eval_every == 0:
                k = s // eval_every
                tr_acc[:, k] = acc(a_tr, b_tr, y_tr)
                te_acc[:, k] = acc(a_te, b_te, y_te)
            if s == steps:
                break
            pre = W1[:, a_tr] + W1[:, b_tr]  # [G, n, width]
            h = torch.relu(pre)
            dF = (h @ W2 - Y) * scale  # [G, n, p]
            gW2 = h.transpose(1, 2) @ dF  # [G, width, p]
            dpre = (dF @ W2.transpose(1, 2)) * (pre > 0)
            gW1 = torch.zeros_like(W1)
            gW1.index_add_(1, a_tr, dpre)
            gW1.index_add_(1, b_tr, dpre)
            # PyTorch SGD with coupled decay: w <- w - lr * (g + wd * w)
            W1.sub_(lrs * (gW1 + wds * W1))
            W2.sub_(lrs * (gW2 + wds * W2))
    return tr_acc, te_acc
