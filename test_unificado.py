"""Pesos y fuerzas del DAE unificado, sin IDA."""
import numpy as np

from umbrales_reg import pesos_regimen


def _stub_ida():
    import sys
    import types
    if "scikits.odes" not in sys.modules:
        scikits = types.ModuleType("scikits")
        odes = types.ModuleType("scikits.odes")
        odes.dae = lambda *a, **k: None
        sys.modules["scikits"] = scikits
        sys.modules["scikits.odes"] = odes


def _cargar_mod(nombre):
    _stub_ida()
    return __import__(nombre)


def _armar(mod):
    mod.limphi1, mod.limphi2, mod.phicrit = 0.20, 0.21, 0.70
    mod.eps_phi, mod.eps_frag, mod.eps_re = 0.001, 0.001, 10.0
    mod.wr = 16.0
    mod.cg = 8.0
    mod.suppress_Fmw = False
    mod.radial_Fmw_override = None
    return mod


def test_pesos_suman_uno():
    phi = np.linspace(0.0, 0.95, 200)
    w1, w2, w3, w4 = pesos_regimen(phi, 0.20, 0.21, 0.70, 0.001, 0.001)
    s = w1 + w2 + w3 + w4
    assert np.allclose(s, 1.0, atol=1e-12)


def test_import_unificado_no_pide_suave():
    U = _cargar_mod("RIconduitex5_5_unificado")
    assert hasattr(U, "RIconduitex5_5_unificado_f")
    src = open(U.__file__, encoding="utf-8").read()
    assert "from RIconduitex5_5_suave import" not in src


def test_fuerzas_finitas():
    for nombre in ("RIconduitex5_5_unificado", "RIconduitex5_5_suave"):
        S = _armar(_cargar_mod(nombre))
        for phi in (1e-8, 0.05, 0.205, 0.40, 0.70, 0.90):
            rb = max((phi / ((4 / 3) * np.pi * 1e13 * max(1 - phi, 1e-12))) ** (1 / 3), 1e-8)
            Fmw, Fgw, Fmg, w = S._fuerzas_unificadas(phi, rb, 20.0, 1e5, 10.0, 12.0)
            assert np.isfinite([Fmw, Fgw, Fmg]).all(), (nombre, phi, Fmw, Fgw, Fmg)
            assert abs(sum(w) - 1.0) < 1e-12


def test_cerca_de_cero_es_rama_1():
    for nombre in ("RIconduitex5_5_unificado", "RIconduitex5_5_suave"):
        S = _armar(_cargar_mod(nombre))
        _, _, _, w = S._fuerzas_unificadas(0.02, 1e-4, 20.0, 1e5, 10.0, 12.0)
        assert w[0] > 0.99
        _, _, _, w4 = S._fuerzas_unificadas(0.95, 1e-3, 5.0, 1e4, 30.0, 80.0)
        assert w4[3] > 0.99


if __name__ == "__main__":
    test_pesos_suman_uno()
    test_import_unificado_no_pide_suave()
    test_fuerzas_finitas()
    test_cerca_de_cero_es_rama_1()
    print("ok")
