
# AnalogSim (alpha)

Differentiable analog matmul simulation for fast prototyping of in‑memory/analog ML effects.
Focus: simple, hackable, *learnable* noise/nonlinear models that work with PyTorch autograd.

## Features (v0.0.1)
- `AnalogMatMul` autograd op: (A @ B) with parametric noise and nonlinearity
- `AnalogLinear` / `AnalogConv2d` drop‑in modules
- Noise models: gaussian, signal‑proportional, clipping/saturation, stuck‑at faults
- Calibration helpers: fit noise params to a given SNR or empirical measurements
- Tiny demo: train a classifier under analog noise and plot loss/accuracy

## Quickstart
```bash
pip install -e .
python -m analogsim.examples.demo_classification
```

## Roadmap
- C++/CUDA extension for speed
- Per‑tile nonuniformity + temperature drift
- Weight programming error / retention drift
- Conv kernels + im2col analog path
```

