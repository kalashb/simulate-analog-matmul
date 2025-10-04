import torch
from torch import Tensor
from .noise import NoiseConfig, make_noise

class AnalogMatMulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A: Tensor, B: Tensor, cfg: NoiseConfig | None):
        y_ideal = A @ B
        cfg = cfg or NoiseConfig(kind="none")
        noise = make_noise(cfg, y_ideal, device=y_ideal.device, dtype=y_ideal.dtype)
        y_noisy = y_ideal + noise

        # Save tensors for backward if needed (we assume noise is not learnable here)
        ctx.save_for_backward(A, B)
        return y_noisy

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        A, B = ctx.saved_tensors
        # Gradient flows through ideal matmul (expected for unbiased noise)
        dA = grad_output @ B.t()
        dB = A.t() @ grad_output
        # None for cfg
        return dA, dB, None
