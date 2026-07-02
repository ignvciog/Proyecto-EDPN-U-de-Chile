import numpy as np
from scipy.special import erf 

def fvrel(model, cx, cxi, ar1, ar2, cmax, sr):
    """
    Calculate relative viscosity based on different models.

    Parameters:
        model: str - Model name ('ER52', 'vona11', 'costa09', 'costa09v11', 'EM2s')
        cx: float - Total crystal content
        cxi: float - Phenocryst content
        ar1: float - Aspect ratio of phenocrysts (width/length)
        ar2: float - Aspect ratio of microlites (width/length)
        cmax: float - Maximum packing fraction
        sr: float - Strain rate (1/s)

    Returns:
        relvisc: float - Relative viscosity
    """
    if model == 'er52':
        relvisc = (1 - cx / cmax) ** -2.5

    elif model == 'vona11':
        phimax1 = 0.656 * np.exp(-(np.log10(ar1) ** 2) / (2 * (1.08 ** 2)))
        phimax2 = 0.656 * np.exp(-(np.log10(ar2) ** 2) / (2 * (1.08 ** 2)))
        phimax = (cxi * phimax1 + (cx - cxi) * phimax2) / cx
        relvisc = (1 - cx / phimax) ** (-2 * (1 - 0.06 * np.log10(sr)))

    elif model == 'em2s':
        phimax1 = 0.656 * np.exp(-(np.log10(ar1) ** 2) / (2 * (1.08 ** 2)))
        phimax2 = 0.656 * np.exp(-(np.log10(ar2) ** 2) / (2 * (1.08 ** 2)))
        relvisc1 = (1 - cxi / phimax1) ** -2.5
        relvisc2 = (1 - (cx - cxi) / ((1 - cxi) * phimax2)) ** -2.5
        relvisc = relvisc1 * relvisc2

    elif model == 'costa09':
        phimax = 0.066499 * np.tanh(0.913424 * np.log10(sr) + 3.850623) + 0.591806
        delta = -6.301095 * np.tanh(0.818496 * np.log10(sr) + 2.86) + 7.462405
        alpha = -0.000378 * np.tanh(1.148101 * np.log10(sr) + 3.92) + 0.999572
        gamma = 3.987815 * np.tanh(0.8908 * np.log10(sr) + 3.24) + 5.099645

        f1 = 1 + (cx / phimax) ** delta
        f2 = 1 + (cx / phimax) ** gamma
        f3 = alpha * erf(np.sqrt(np.pi) * cx * f2 / (2 * alpha * phimax))
        relvisc = f1 / ((1 - f3) ** (2.5 * phimax))

    elif model == 'costa09v11':
        phimax = 0.274
        delta = 13 - 0.84
        alpha = 1 - 0.0327
        gamma = 0.84

        f1 = 1 + (cx / phimax) ** delta
        f2 = 1 + (cx / phimax) ** gamma
        f3 = alpha * erf(np.sqrt(np.pi) * cx * f2 / (2 * alpha * phimax))
        relvisc = f1 / ((1 - f3) ** (2.5 * phimax))

    else:
        raise ValueError(f"Unknown model: {model}")

    return relvisc
