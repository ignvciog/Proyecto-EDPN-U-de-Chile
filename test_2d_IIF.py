"""Marcha en diferencias finitas de II.F: caudal, pared y el empinamiento de arriba."""
import numpy as np

import RIconduit2D_IIF as IIF


def _gradiente(sal, i0, i1):
    return (sal["P"][i1] - sal["P"][i0]) / (sal["z"][i1] - sal["z"][i0])


def test_el_tiro_alto_se_ahoga_y_se_empina():
    sal = IIF.marchar(23.0, n_r=8)
    assert sal["mensaje"] == "ahogado"
    assert -1600.0 < sal["z"][-1] < -900.0
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


def test_la_recta_sin_burbujas_llega_hasta_el_metro_de_exsolucion():
    sal = IIF.marchar(23.0, n_r=8)
    assert sal["z_ad"] > sal["z_sat"]
    assert abs((sal["z_ad"] - sal["z_sat"]) - 1.0) < 1e-9
    profundo = sal["z"] < sal["z_sat"] - 1.0
    assert np.max(sal["phi"][profundo]) == 0.0
    i_ad = int(np.argmin(np.abs(sal["z"] - sal["z_ad"])))
    assert sal["phi"][i_ad, 0] > 0.0
    assert np.max(sal["phi"][i_ad]) - np.min(sal["phi"][i_ad]) < 1e-12
    assert sal["um"][i_ad, -1] == 0.0
    assert sal["ug"][i_ad, -1] == 0.0
    assert np.allclose(sal["um"][i_ad], sal["ug"][i_ad])
    assert sal["um"][i_ad, 0] > sal["um"][i_ad, -2]
    base = np.argmin(np.abs(sal["z"] - sal["z"][0]))
    assert sal["um"][base, -1] == 0.0
    assert sal["um"][base, 0] > sal["vinicial"]


def test_los_limites_salen_del_numero_capilar():
    r, _ = IIF._malla(8, 16.0)
    co, pi = IIF._configurar_fd(16.0, 5.0e6, 4.0, 1243.15, 0.25)
    rho = IIF._rho_m(pi)
    mu = IIF._mu_liquido(pi, 0.25, 18.0 / 16.0, 1243.15, co)
    lim = IIF.limites_regiones(r, 18.0, 16.0, pi, rho, mu, co, 0.25, 1243.15)
    assert np.all((0.525 < lim["phicrit_ca"]) & (lim["phicrit_ca"] < 0.785))
    assert np.all((0.15 < lim["limphi1"]) & (lim["limphi1"] < 0.40))
    assert np.allclose(lim["limphi2"], lim["limphi1"] + 0.01)
    assert np.all(lim["Ca"] > 0.0)
    assert lim["extension"][0] > lim["extension"][-1]
    assert lim["cizalle"][-1] > lim["cizalle"][0]
    assert not np.isclose(lim["Ca"][0], lim["Ca"][-1])
    assert lim["Nd"] > 0.0
    assert abs(lim["phi_ref"] - 0.2) < 0.002


def test_un_caudal_menor_llega_a_la_boca():
    sal = IIF.marchar(10.0, n_r=8)
    assert sal["mensaje"] == "boca"
    assert sal["z"][-1] > -1.0
    assert sal["P"][-1] < sal["P"][0]
    assert np.max(np.abs(sal["um"][:, -1])) == 0.0
    assert sal["phi"][-1, 0] > 0.5
    assert sal["um"][-1, 0] > sal["um"][-1, -2]
