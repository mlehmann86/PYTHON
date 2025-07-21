import numpy as np

def gaussian(z, A, z0, sigma, offset):
    """
    Gaussian function for fitting the vertical dust density profile.

    Parameters:
    - z: Vertical coordinate array.
    - A: Amplitude of the Gaussian.
    - z0: Mean (center) of the Gaussian.
    - sigma: Standard deviation (related to scale height).
    - offset: Background offset.

    Returns:
    - Gaussian function evaluated at z.
    """
    return A * np.exp(-((z - z0) ** 2) / (2 * sigma ** 2)) + offset
