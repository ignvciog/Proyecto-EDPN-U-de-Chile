"""Figura del leak de softplus vs tasa_xi (sin IDA)."""
import os

import matplotlib.pyplot as plt
import numpy as np

from umbrales_reg import EPS_XI, softplus, tasa_xi

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "umbrales_reg_figuras")
os.makedirs(OUT, exist_ok=True)


def main():
    eps = EPS_XI
    tcar = 2 * 3600
    xi0, xmax = 0.25, 0.50
    f2, xteo, f3, dx_ok = tasa_xi(-0.01, xi0, xi0, xmax, tcar, eps)
    dx_leak = float(
        softplus(
            (xmax - xi0)
            * float(softplus(-0.01, eps))
            * float(softplus(1.0 - xi0 / max(xteo, 1e-30), eps))
            / tcar,
            eps,
        )
    )
    L = np.linspace(0.0, 4000.0, 400)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(L, np.minimum(xmax, xi0 + dx_ok * L), lw=2.4, label="tasa_xi (como el original)")
    ax.plot(L, np.minimum(xmax, xi0 + dx_leak * L), lw=2.2, label="softplus extra sobre dx/dz")
    ax.axhline(0.252, color="0.4", lw=1, ls="--", label="como antes (~0.252)")
    ax.axhline(0.35, color="0.6", lw=1, ls=":", label="suave viejo (~0.35)")
    ax.set_xlabel("metros integrados con tasa filtrada")
    ax.set_ylabel("crystal content")
    ax.set_ylim(0.24, 0.52)
    ax.set_title("el leak hincha ξ; no es el EPS")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT, "11_leak_cristales.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    print("dx_ok", dx_ok, "dx_leak", dx_leak)


if __name__ == "__main__":
    main()
