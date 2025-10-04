import torch
from analogsim.modules import AnalogLinear
from analogsim.noise import NoiseConfig

def test_forward_shapes():
    layer = AnalogLinear(4, 5, noise_cfg=NoiseConfig(kind='gaussian', sigma=0.01))
    x = torch.randn(2,4)
    y = layer(x)
    assert y.shape == (2,5)
