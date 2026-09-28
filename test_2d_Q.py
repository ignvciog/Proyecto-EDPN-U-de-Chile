"""La columna sigue al 1D: Q constante, pared en reposo, gradiente más fuerte arriba."""
import numpy as np

import RIconduit2D_Q as Q


def _gradiente(sal, i0, i1):
    return (sal["P"][i1] - sal["P"][i0]) / (sal["z"][i1] - sal["z"][i0])


def test_marcha_conserva_Q_y_la_pared():
    sal = Q.marchar(n_r=8)
    assert sal["n_pasos"] >= 20
    assert sal["mensaje"] == "boca"
    assert np.isfinite(sal["P"]).all()
    assert np.isfinite(sal["um"]).all()
    assert np.isfinite(sal["ug"]).all()
    rel = np.max(np.abs(sal["Q"] - sal["Q_objetivo"]) / sal["Q_objetivo"])
    assert rel < 1e-8
    assert np.max(np.abs(sal["um"][:, -1])) == 0.0
    assert np.max(np.abs(sal["ug"][:, -1])) == 0.0
    assert sal["um"][-1, 0] > sal["um"][-1, -2]
    assert sal["P"][-1] < sal["P"][0]
    # El tiro llega cerca de P_atm y el gas se acelera, como en el 1D.
    assert sal["P"][-1] < 5e6
    assert sal["ug_media"][-1] > 5.0 * sal["ug_media"][0]
    assert sal["phi"][:, 0].max() > 0.7


def test_la_presion_se_empina_arriba():
    sal = Q.marchar(n_r=8)
    z = sal["z"]
    profundo = np.where(z < -5000)[0]
    arriba = np.where(z > -30)[0]
    g_profundo = _gradiente(sal, profundo[0], profundo[-1])
    g_arriba = _gradiente(sal, arriba[0], arriba[-1])
    # dP/dz es negativo. En los últimos metros, junto al sónico, es mucho más fuerte.
    assert g_arriba < 4.0 * g_profundo
    assert sal["phi"][profundo[-1], 0] < 0.05
    assert sal["phi"][-1, 0] > 0.5
