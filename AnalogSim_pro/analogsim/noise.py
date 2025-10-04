from dataclasses import dataclass
import torch

@dataclass
class NoiseConfig:
    kind: str = "gaussian"          # 'gaussian' | 'proportional' | 'clipped' | 'none'
    sigma: float = 0.05             # stddev for gaussian
    alpha: float = 0.02             # proportionality for signal-dependent noise
    clip: float = 6.0               # clipping threshold (in linear op output units)
    stuck_prob: float = 0.0         # probability a weight is 'stuck' (0 value)
    seed: int | None = None

def make_noise(cfg: NoiseConfig, y_ideal: torch.Tensor, *, device=None, dtype=None):
    device = device or y_ideal.device
    dtype = dtype or y_ideal.dtype
    g = torch.Generator(device=device)
    if cfg.seed is not None:
        g.manual_seed(cfg.seed)

    if cfg.kind == "none":
        return torch.zeros_like(y_ideal)

    if cfg.kind == "gaussian":
        return torch.normal(mean=0.0, std=cfg.sigma, size=y_ideal.shape, generator=g, device=device, dtype=dtype)

    if cfg.kind == "proportional":
        # std proportional to magnitude of signal
        std = cfg.alpha * (y_ideal.abs() + 1e-6)
        return torch.randn_like(y_ideal, generator=g) * std

    if cfg.kind == "clipped":
        noise = torch.normal(mean=0.0, std=cfg.sigma, size=y_ideal.shape, generator=g, device=device, dtype=dtype)
        y = y_ideal + noise
        y = torch.clamp(y, -cfg.clip, cfg.clip)
        return y - y_ideal

    # default
    return torch.zeros_like(y_ideal)
