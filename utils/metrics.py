import numpy as np


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    orig = original.astype(np.float64)
    recon = reconstructed.astype(np.float64)
    mse = np.mean((orig - recon) ** 2)
    if mse == 0:
        return 100.0
    max_val = 255.0 if orig.max() > 1.0 else 1.0
    return float(20 * np.log10(max_val / np.sqrt(mse)))


def compute_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    orig = original.astype(np.float64)
    recon = reconstructed.astype(np.float64)
    if orig.max() <= 1.0:
        orig = orig * 255.0
        recon = recon * 255.0
    mu1, mu2 = orig.mean(), recon.mean()
    sigma1 = orig.std()
    sigma2 = recon.std()
    sigma12 = np.mean((orig - mu1) * (recon - mu2))
    C1, C2 = 6.5025, 58.5225  # (0.01*255)^2, (0.03*255)^2
    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2)
    return float(num / den) if den != 0 else 1.0


def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2))


def compute_ncc(original: np.ndarray, reconstructed: np.ndarray) -> float:
    o = original.astype(np.float64).flatten()
    r = reconstructed.astype(np.float64).flatten()
    o -= o.mean()
    r -= r.mean()
    den = np.sqrt((o ** 2).sum() * (r ** 2).sum())
    if den == 0:
        return 0.0
    return float(np.dot(o, r) / den)


def compute_taf(mask: np.ndarray, threshold: float = 0.5) -> float:
    """Tamper Area Fraction — fraction of pixels classified as tampered."""
    return float(np.mean(mask > threshold))


def compute_ber(original_bits: np.ndarray, extracted_bits: np.ndarray) -> float:
    if len(original_bits) == 0:
        return 0.0
    return float(np.mean(original_bits != extracted_bits))


def all_metrics(original: np.ndarray, reconstructed: np.ndarray,
                original_bits=None,
                extracted_bits=None) -> dict:
    m = {
        "PSNR (dB)": round(compute_psnr(original, reconstructed), 4),
        "SSIM":       round(compute_ssim(original, reconstructed), 4),
        "MSE":        round(compute_mse(original, reconstructed), 4),
        "NCC":        round(compute_ncc(original, reconstructed), 4),
    }
    if original_bits is not None and extracted_bits is not None:
        m["BER"] = round(compute_ber(np.array(original_bits), np.array(extracted_bits)), 4)
    return m
