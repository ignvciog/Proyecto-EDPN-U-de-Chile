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


if __name__ == "__main__":
    test_softplus_cero_no_es_cero()
    test_tasa_subsaturada_no_crece()
    test_tasa_igual_al_max0_si_lejos()
    test_default_eps_xi_es_el_de_f2_f3()
    print("ok")
