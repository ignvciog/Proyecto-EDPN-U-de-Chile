"""Pesos y fuerzas del DAE unificado, sin IDA."""
import numpy as np

from umbrales_reg import pesos_regimen


def test_pesos_suman_uno():
    phi = np.linspace(0.0, 0.95, 200)
    w1, w2, w3, w4 = pesos_regimen(phi, 0.20, 0.21, 0.70, 0.001, 0.001)
    s = w1 + w2 + w3 + w4
    assert np.allclose(s, 1.0, atol=1e-12)


def test_fuerzas_finitas():
    import sys
    import types
    if "scikits.odes" not in sys.modules:
        scikits = types.ModuleType("scikits")
        odes = types.ModuleType("scikits.odes")
        odes.dae = lambda *a, **k: None
        sys.modules["scikits"] = scikits
        sys.modules["scikits.odes"] = odes
    import RIconduitex5_5_suave as S
    S.limphi1, S.limphi2, S.phicrit = 0.20, 0.21, 0.70
    S.eps_phi, S.eps_frag, S.eps_re = 0.001, 0.001, 10.0
    S.wr = 16.0
    S.cg = 8.0
    S.suppress_Fmw = False
    S.radial_Fmw_override = None
    for phi in (1e-8, 0.05, 0.205, 0.40, 0.70, 0.90):
        rb = max((phi / ((4 / 3) * np.pi * 1e13 * max(1 - phi, 1e-12))) ** (1 / 3), 1e-8)
        Fmw, Fgw, Fmg, w = S._fuerzas_unificadas(phi, rb, 20.0, 1e5, 10.0, 12.0)
        assert np.isfinite([Fmw, Fgw, Fmg]).all(), (phi, Fmw, Fgw, Fmg)
        assert abs(sum(w) - 1.0) < 1e-12


def test_cerca_de_cero_es_rama_1():
    import RIconduitex5_5_suave as S
    S.limphi1, S.limphi2, S.phicrit = 0.20, 0.21, 0.70
    S.eps_phi, S.eps_frag, S.eps_re = 0.001, 0.001, 10.0
    S.wr, S.cg = 16.0, 8.0
    S.suppress_Fmw = False
    S.radial_Fmw_override = None
    _, _, _, w = S._fuerzas_unificadas(0.02, 1e-4, 20.0, 1e5, 10.0, 12.0)
    assert w[0] > 0.99
    _, _, _, w4 = S._fuerzas_unificadas(0.95, 1e-3, 5.0, 1e4, 30.0, 80.0)
    assert w4[3] > 0.99


if __name__ == "__main__":
    test_pesos_suman_uno()
    test_fuerzas_finitas()
    test_cerca_de_cero_es_rama_1()
    print("ok")
