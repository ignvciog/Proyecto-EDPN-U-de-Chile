"""El leak de softplus sobre dx/dz, sin IDA."""
from umbrales_reg import softplus, tasa_xi, EPS_XI


def test_softplus_cero_no_es_cero():
    eps = 1e-4
    leak = float(softplus(0.0, eps))
    assert 0.6 * eps < leak < 0.8 * eps


def test_tasa_subsaturada_no_crece():
    f2, xteo, f3, dx = tasa_xi(-0.01, 0.25, 0.25, 0.50, 7200.0, 1e-4)
    assert dx < 1e-12
    leaked = float(softplus((0.50 - 0.25) * f2 * f3 / 7200.0, 1e-4))
    assert leaked > 5e-5


def test_tasa_igual_al_max0_si_lejos():
    f2_raw = 0.4
    x, xi, xmax, denom = 0.25, 0.25, 0.50, 7200.0
    f2, xteo, f3, dx = tasa_xi(f2_raw, x, xi, xmax, denom, 1e-12)
    f2_h = max(0.0, f2_raw)
    xteo_h = xi + (xmax - xi) * f2_h
    f3_h = max(0.0, 1.0 - x / xteo_h)
    dx_h = (xmax - xi) * f2_h * f3_h / denom
    assert abs(dx - dx_h) < 1e-12


def test_default_eps_xi_es_el_de_f2_f3():
    assert EPS_XI <= 1e-4


def test_cortar_deja_el_punto_que_cruzo():
    import sys
    import types
    import numpy as np

    if "scikits.odes" not in sys.modules:
        scikits = types.ModuleType("scikits")
        odes = types.ModuleType("scikits.odes")
        odes.dae = lambda *a, **k: None
        sys.modules["scikits"] = scikits
        sys.modules["scikits.odes"] = odes
    from RIconduitex5_5_suave import _cortar_cruce

    y0 = np.array([1e8, 0.05, 1e13, 0.25, 10.0, 10.0])
    y_all = np.array([
        [9e7, 0.10, 1e13, 0.25, 10.0, 10.0],
        [8e7, 0.25, 1e13, 0.25, 12.0, 15.0],
    ])
    t_all = np.array([-2000.0, -1000.0])
    y2, t2 = _cortar_cruce(y_all, t_all, y0, -3000.0, 0.20)
    assert abs(y2[-1, 1] - 0.25) < 1e-15
    assert abs(t2[-1] + 1000.0) < 1e-12


if __name__ == "__main__":
    test_softplus_cero_no_es_cero()
    test_tasa_subsaturada_no_crece()
    test_tasa_igual_al_max0_si_lejos()
    test_default_eps_xi_es_el_de_f2_f3()
    test_cortar_deja_el_punto_que_cruzo()
    print("ok")
