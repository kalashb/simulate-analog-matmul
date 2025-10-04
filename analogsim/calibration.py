import torch
from .noise import NoiseConfig

def snr_to_sigma(snr_db: float, signal_std: float) -> float:
    # SNR_dB = 20 * log10(signal_std / sigma) => sigma = signal_std / (10^(SNR_dB/20))
    return signal_std / (10 ** (snr_db / 20))

def calibrate_sigma_for_snr(y_samples: torch.Tensor, target_snr_db: float) -> float:
    signal_std = y_samples.std().item()
    return snr_to_sigma(target_snr_db, signal_std)
