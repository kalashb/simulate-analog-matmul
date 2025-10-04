import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from analogsim.modules import AnalogLinear
from analogsim.noise import NoiseConfig

def make_data(n=2000, d=8, k=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    W_true = torch.randn(d, k, generator=g)
    y = (X @ W_true + 0.3*torch.randn(n, k, generator=g)).argmax(dim=1)
    return X, y

def train(noise_kind="gaussian", sigma=0.05, epochs=20, lr=1e-2):
    X, y = make_data()
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    model = nn.Sequential(
        AnalogLinear(8, 32, noise_cfg=NoiseConfig(kind=noise_kind, sigma=sigma)),
        nn.ReLU(),
        AnalogLinear(32, 3, noise_cfg=NoiseConfig(kind=noise_kind, sigma=sigma)),
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs+1):
        total = 0.0
        correct = 0
        n = 0
        for xb, yb in dl:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

            total += loss.item() * xb.size(0)
            pred = out.argmax(1)
            correct += (pred == yb).sum().item()
            n += xb.size(0)
        print(f"epoch {epoch:02d} | loss {total/n:.4f} | acc {correct/n:.3f}")

if __name__ == '__main__':
    train()
