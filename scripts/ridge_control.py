"""Control for claim C6: does the ridge identity become exact once GD reaches a stationary point?

Same tiny problem (24 mod-23 examples, width 32, wd 1e-3, lr 0.2, float64), two activations:
  square: phi(z) = z^2  (smooth, 3-homogeneous, so c = wd*n*p*3/2)
  relu:   phi(z) = max(z, 0)  (2-homogeneous but not smooth)
Usage: python scripts/ridge_control.py {square|relu} 0.2 200000
"""
import sys, torch
from torch.func import jacrev
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from grok.data import difference_split
torch.manual_seed(0); torch.set_num_threads(1)
d = difference_split(); X, y, _, _ = d.split(); X, y = X[:24].double(), y[:24]
n, p, wd, lr = len(y), 23, 1e-3, float(sys.argv[2])
Y = torch.nn.functional.one_hot(y, p).double()
act = {"relu": (torch.relu, 2), "square": (lambda z: z * z, 3)}[sys.argv[1]]
phi, D = act
W1 = (torch.randn(46, 32) * (2/46)**.5).double().requires_grad_(); W2 = (torch.randn(32, p) * (2/32)**.5).double().requires_grad_()
f = lambda a, b: (phi(X @ a) @ b)
for s in range(int(sys.argv[3]) + 1):
    loss = ((f(W1, W2) - Y) ** 2).mean()
    g1, g2 = torch.autograd.grad(loss, (W1, W2))
    if s % (int(sys.argv[3]) // 4) == 0:
        with torch.no_grad():
            full = torch.cat([(g1 + wd*W1).flatten(), (g2 + wd*W2).flatten()]).norm(); data = torch.cat([g1.flatten(), g2.flatten()]).norm()
        J1, J2 = jacrev(f, argnums=(0, 1))(W1.detach(), W2.detach())
        J = torch.cat([J1.reshape(n*p, -1), J2.reshape(n*p, -1)], 1); K = J @ J.T
        c = wd * n * p * D / 2
        sig = (f(W1, W2) - Y).detach().flatten()
        star = -c * torch.linalg.solve(K + c*torch.eye(n*p, dtype=K.dtype), Y.flatten())
        z = (X @ W1).detach(); near = (z.abs() < 1e-4 * z.abs().max()).float().mean()
        print(sys.argv[1], s, f"grad ratio {full/data:.2e}  rel err {((sig-star).norm()/sig.norm()):.2e}  pre-acts near 0: {near:.3f}", flush=True)
    with torch.no_grad():
        W1 -= lr * (g1 + wd * W1); W2 -= lr * (g2 + wd * W2)
