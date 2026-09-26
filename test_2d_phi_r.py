"""Tests del 2D con φ(r,z): operator L_r, F_mg suave, una marcha corta."""
import numpy as np

from umbrales_reg import pesos_regimen


def test_pesos_suman_uno():
    phi = np.linspace(0.0, 0.95, 120)
    w1, w2, w3, w4 = pesos_regimen(phi, 0.20, 0.21, 0.70, 0.001, 0.001)
    assert np.allclose(w1 + w2 + w3 + w4, 1.0, atol=1e-12)


def test_tridiag_phi_varia_con_r():
    import RIconduit2D_phi_r as M

    M._configurar(16.0, 5e6, 4.0, 1243.15, 0.25, 12, {})
    N = M._G["N_r"]
    P = 5e7
    phi_flat = np.full(N, 0.15)
    phi_var = np.linspace(0.05, 0.25, N)
    Nd = np.full(N, 1e8)
    x = np.full(N, 0.25)
    ug = np.full(N, 8.0)
    dpdz = -8.0e4
    um_f, _, _ = M.solve_radial(P, phi_flat, Nd, x, ug, dpdz, n_picard=3)
    um_v, _, _ = M.solve_radial(P, phi_var, Nd, x, ug, dpdz, n_picard=3)
    assert um_f[-1] == 0.0 and um_v[-1] == 0.0
    assert um_f[0] > um_f[-2] > 0.0
    assert np.all(np.isfinite(um_f)) and np.all(np.isfinite(um_v))
    # φ(r) cambia el perfil: no es el 2D viejo con φ plano
    assert np.max(np.abs(um_f - um_v)) > 1e-6


def test_Fmg_suave_finito():
    import RIconduit2D_phi_r as M

    M._configurar(16.0, 5e6, 4.0, 1243.15, 0.25, 8, {})
    phi = np.array([1e-8, 0.05, 0.205, 0.40, 0.70, 0.90, 0.20, 0.10])
    um = np.full(8, 10.0)
    ug = np.full(8, 12.0)
    Nd = np.full(8, 1e13)
    visc = np.full(8, 1e5)
    F = M._Fmg_suave(um, ug, phi, Nd, 20.0, visc)
    assert np.all(np.isfinite(F))


def test_marcha_corta_phi_es_2d():
    import RIconduit2D_phi_r as M

    out = M.RIconduit2D_phi_r_f(N_r=8, max_steps=12, max_count=1, tol=2e-3, verbose=False)
    assert out["phi"].ndim == 2
    assert out["phi"].shape[1] == out["r"].size
    assert out["Nd"].shape == out["phi"].shape
    assert out["xi"].shape == out["phi"].shape
    assert out["um"].shape == out["phi"].shape
    assert np.all(np.isfinite(out["phi"]))
    assert out["um"][:, -1].max() < 1e-12
    # el campo existe en (r,z); si ya hay gas, eje y pared pueden diferir
    if np.max(out["phi"]) > 1e-6:
        assert out["phi"].shape[0] >= 2


if __name__ == "__main__":
    test_pesos_suman_uno()
    test_tridiag_phi_varia_con_r()
    test_Fmg_suave_finito()
    test_marcha_corta_phi_es_2d()
    print("ok")
