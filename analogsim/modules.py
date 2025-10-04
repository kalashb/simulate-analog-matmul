import torch
from torch import nn
from .ops import AnalogMatMulFn
from .noise import NoiseConfig

class AnalogLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, noise_cfg: NoiseConfig | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()
        self.noise_cfg = noise_cfg

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.weight.size(0)
            bound = 1 / fan_in ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        y = AnalogMatMulFn.apply(x, self.weight, self.noise_cfg)
        if self.bias is not None:
            y = y + self.bias
        return y

class AnalogConv2d(nn.Module):
    # simple im2col-based conv using analog matmul for demonstration
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, noise_cfg: NoiseConfig | None = None):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_h = kernel_w = kernel_size
        else:
            kernel_h, kernel_w = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.kernel_size = (kernel_h, kernel_w)
        self.noise_cfg = noise_cfg
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_h, kernel_w))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.weight.numel() / self.weight.shape[0]
            bound = 1 / fan_in ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def _im2col(self, x):
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        x_pad = torch.nn.functional.pad(x, (pw, pw, ph, ph))
        H_out = (H + 2*ph - kh)//sh + 1
        W_out = (W + 2*pw - kw)//sw + 1
        patches = x_pad.unfold(2, kh, sh).unfold(3, kw, sw)  # (N, C, H_out, W_out, kh, kw)
        cols = patches.contiguous().view(N, C*kh*kw, H_out*W_out)  # (N, K, L)
        return cols, H_out, W_out

    def forward(self, x):
        # im2col -> (N, K, L), weight -> (K, O), output -> (N, O, L) -> (N, O, H_out, W_out)
        cols, H_out, W_out = self._im2col(x)
        N, K, L = cols.shape
        W = self.weight.view(self.weight.shape[0], -1).t()  # (K, O)
        y = AnalogMatMulFn.apply(cols.transpose(1,2), W, self.noise_cfg)  # (N, L, O)
        y = y.transpose(1,2).contiguous().view(N, -1, H_out, W_out)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y
