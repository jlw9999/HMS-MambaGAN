import numpy as np
from skimage.util import crop

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def mae_SynthRAD2023(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Compute Mean Absolute Error (MAE)

    Parameters
    ----------
    gt : np.ndarray
        Ground truth
    pred : np.ndarray
        Prediction
    mask : np.ndarray, optional
        Mask for voxels to include. The default is None (including all voxels).

    Returns
    -------
    mae : float
        Mean absolute error.
    """
    if mask is None:
        mask = np.ones_like(gt)
    else:
        mask = np.where(mask > 0, 1., 0.)

    mae_value = np.sum(np.abs(gt * mask - pred * mask)) / mask.sum()
    return float(mae_value)


def psnr_SynthRAD2023(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Peak Signal to Noise Ratio (PSNR) between two images.

    Parameters
    ----------
    img1 : np.ndarray
        First image.
    img2 : np.ndarray
        Second image.

    Returns
    -------
    float
        Peak signal to noise ratio.
    """
    psnr_value = peak_signal_noise_ratio(img1, img2, data_range=img1.max() - img1.min())
    return float(psnr_value)


def ssim_SynthRAD2023(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Compute Structural Similarity Index Metric (SSIM) between two images.

    Parameters
    ----------
    img1 : np.ndarray
        First image.
    img2 : np.ndarray
        Second image.
    mask : np.ndarray, optional
        Mask for voxels to include. The default is None (including all voxels).

    Returns
    -------
    float
        Structural similarity index metric.
    """
    ssim_value_full, _ = structural_similarity(img1, img2, full=True)

    return ssim_value_full

    # if mask is not None:
    #     pad = 3
    #     ssim_value_masked = (crop(ssim_map, pad)[crop(mask, pad).astype(bool)]).mean(dtype=np.float64)
    #     return ssim_value_masked
    # else:
    #     return ssim_value_full
