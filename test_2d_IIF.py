"""Marcha en diferencias finitas de II.F: caudal, pared y el empinamiento de arriba."""
import numpy as np

import RIconduit2D_IIF as IIF


def _gradiente(sal, i0, i1):
    return (sal["P"][i1] - sal["P"][i0]) / (sal["z"][i1] - sal["z"][i0])


def test_el_tiro_alto_se_ahoga_y_se_empina():
    sal = IIF.marchar(23.0, n_r=8)
    assert sal["mensaje"] == "ahogado"
    assert -400.0 < sal["z"][-1] < -20.0
    assert np.isfinite(sal["P"]).all()
    assert np.isfinite(sal["um"]).all()
    assert np.isfinite(sal["ug"]).all()
    rel = np.max(np.abs(sal["Q"] - sal["Q_objetivo"]) / sal["Q_objetivo"])
    assert rel < 1e-8
    assert np.max(np.abs(sal["um"][:, -1])) == 0.0
    assert np.max(np.abs(sal["ug"][:, -1])) == 0.0
    assert sal["P"][-1] < sal["P"][0]
    assert sal["P"][-1] < 5e6
    assert sal["phi"][-1, 0] > 0.8
    assert sal["um_media"][-1] > 3.0 * sal["um_media"][0]
    profundo = np.where(sal["z"] < -5000)[0]
    assert sal["phi"][profundo[-1], 0] < 0.01
    medio = np.argmin(np.abs(sal["z"] + 2000.0))
    assert sal["um"][medio, 0] > sal["um"][medio, -2]
    cola = np.where(sal["z"] > sal["z"][-1] - 40.0)[0]
    g_profundo = _gradiente(sal, profundo[0], profundo[-1])
    g_cola = _gradiente(sal, cola[0], cola[-1])
    assert g_cola < 3.0 * g_profundo


def test_un_caudal_menor_llega_a_la_boca():
    sal = IIF.marchar(18.0, n_r=8)
    assert sal["mensaje"] == "boca"
    assert sal["z"][-1] > -1.0
    assert sal["P"][-1] < sal["P"][0]
    assert np.max(np.abs(sal["um"][:, -1])) == 0.0
    assert sal["phi"][-1, 0] > 0.5
    assert sal["um"][-1, 0] > sal["um"][-1, -2]
