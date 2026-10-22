"""Modular-addition datasets used in the paper (Appendix G)."""
from dataclasses import dataclass

import numpy as np
import torch

# Appendix G.2: training differences selected by NumPy seed 2027 for p = 23.
PAPER_DIFFS_23 = (0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 16, 17, 19, 20, 22)


@dataclass
class ModAddData:
    p: int
    a: torch.Tensor  # first operand, int64 [n]
    b: torch.Tensor  # second operand, int64 [n]
    y: torch.Tensor  # label (a + b) mod p, int64 [n]
    train: torch.Tensor  # bool mask [n]

    @property
    def X(self) -> torch.Tensor:
        """Concatenated one-hot inputs, shape [n, 2p]."""
        n = len(self.a)
        X = torch.zeros(n, 2 * self.p)
        X[torch.arange(n), self.a] = 1.0
        X[torch.arange(n), self.p + self.b] = 1.0
        return X

    def split(self):
        X, y = self.X, self.y
        return X[self.train], y[self.train], X[~self.train], y[~self.train]


def difference_split(p: int = 23, diffs=PAPER_DIFFS_23) -> ModAddData:
    """All p^2 ordered pairs; train iff (a - b) mod p is in `diffs` (shift-invariant split)."""
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    a, b = a.reshape(-1), b.reshape(-1)
    diffs = torch.tensor(sorted(diffs))
    train = torch.isin((a - b) % p, diffs)
    return ModAddData(p, a, b, (a + b) % p, train)


def unordered_random_split(p: int = 97, seed: int = 0) -> ModAddData:
    """Appendix G.1: the p(p+1)/2 unordered pairs (a <= b), NumPy-permuted, first half trains."""
    pairs = [(a, b) for a in range(p) for b in range(a, p)]
    perm = np.random.default_rng(seed).permutation(len(pairs))
    n_train = len(pairs) // 2
    train = torch.zeros(len(pairs), dtype=torch.bool)
    train[torch.as_tensor(perm[:n_train])] = True
    a = torch.tensor([q[0] for q in pairs])
    b = torch.tensor([q[1] for q in pairs])
    return ModAddData(p, a, b, (a + b) % p, train)
