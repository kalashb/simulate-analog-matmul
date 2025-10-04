import argparse
import json
import time
import torch
from analogsim.noise import NoiseConfig
from analogsim.modules import AnalogLinear
from torch import nn

def run_demo(args):
    torch.manual_seed(args.seed)
    X = torch.randn(args.n, args.d)
    W_true = torch.randn(args.d, args.k)
    y = (X @ W_true + 0.3*torch.randn(args.n, args.k)).argmax(dim=1)

    model = nn.Sequential(
        AnalogLinear(args.d, args.h, noise_cfg=NoiseConfig(kind=args.noise, sigma=args.sigma, alpha=args.alpha, clip=args.clip)),
        nn.ReLU(),
        AnalogLinear(args.h, args.k, noise_cfg=NoiseConfig(kind=args.noise, sigma=args.sigma, alpha=args.alpha, clip=args.clip)),
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(1, args.epochs+1):
        opt.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
    elapsed = time.time() - start

    acc = (out.argmax(1) == y).float().mean().item()
    result = {
        "noise": args.noise, "sigma": args.sigma, "alpha": args.alpha, "clip": args.clip,
        "epochs": args.epochs, "lr": args.lr, "acc": round(acc, 4), "time_s": round(elapsed, 3)
    }
    print(json.dumps(result, indent=2))

def main():
    p = argparse.ArgumentParser(description="AnalogSim CLI")
    p.add_argument("--noise", default="gaussian", choices=["none","gaussian","proportional","clipped"])
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.02)
    p.add_argument("--clip", type=float, default=6.0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--h", type=int, default=32)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run_demo(args)

if __name__ == "__main__":
    main()
