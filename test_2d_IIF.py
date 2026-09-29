"""Marcha en diferencias finitas de II.F: caudal, pared y el empinamiento de arriba."""
import numpy as np

import RIconduit2D_IIF as IIF


def test_el_tiro_alto_se_vuelve_sonico_con_deslizamiento():
    sal = IIF.marchar(23.0, n_r=8)
    assert sal["mensaje"] == "sonico"
    assert -1100.0 < sal["z"][-1] < -500.0
    assert np.isfinite(sal["P"]).all()
    assert np.isfinite(sal["um"]).all()
    assert np.isfinite(sal["ug"]).all()
    rel = np.max(np.abs(sal["Q"] - sal["Q_objetivo"]) / sal["Q_objetivo"])
    assert rel < 1e-6
    assert np.max(np.abs(sal["um"][:, -1])) == 0.0
    assert np.max(np.abs(sal["ug"][:, -1])) == 0.0
    assert 1.0e6 < sal["P"][-1] < 4.0e6
    assert sal["phi"][-1, 0] > 0.8
    assert sal["ug"][-1, 0] > sal["um"][-1, 0] + 200.0
    assert sal["z_frag"] < sal["z"][-1]
    i_frag = int(np.argmin(np.abs(sal["z"] - sal["z_frag"])))
    assert abs(sal["ug"][i_frag, 0] - sal["um"][i_frag, 0]) < 1.0
    profundo = np.where(sal["z"] < -5000)[0]
    assert sal["phi"][profundo[-1], 0] < 0.01
    medio = np.argmin(np.abs(sal["z"] + 2000.0))
    assert sal["um"][medio, 0] > sal["um"][medio, -2]


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


def test_dieciocho_sigue_despues_de_fragmentar_y_se_pone_sonico():
    sal = IIF.marchar(18.0, n_r=8)
    assert sal["mensaje"] == "sonico"
    assert sal["z_frag"] < -800.0
    assert sal["z"][-1] > -400.0
    assert 1.0e6 < sal["P"][-1] < 3.0e6
    assert sal["ug"][-1, 0] > sal["um"][-1, 0] + 200.0
    assert np.max(np.abs(sal["um"][:, -1])) == 0.0
    assert np.max(np.abs(sal["ug"][:, -1])) == 0.0
    rel = np.max(np.abs(sal["Q"] - sal["Q_objetivo"]) / sal["Q_objetivo"])
    assert rel < 1e-6


def test_el_tiro_con_el_cierre_del_1d_cae_en_la_ventana_sonica():
    import math
    from calbuco2015d import R, T1, Patm

    tiro = IIF.tirar_como_1d(n_r=8)
    assert tiro["convergido"]
    assert tiro["criterio"] == "sonico"
    assert tiro["n_pasos"] < 60
    sal = tiro["solucion"]
    vsound = 0.99 * math.sqrt(R * T1)
    ug = sal["ug_media"][-1]
    assert sal["z"][-1] >= -5.0
    assert sal["P"][-1] >= Patm
    assert 0.95 * vsound < ug <= 1.05 * vsound
    assert sal["ug_media"][-1] / sal["um_media"][-1] > 2.5
    assert sal["ug"][-1, 0] / sal["um"][-1, 0] > 2.5
    assert np.max(np.abs(sal["um"][:, -1])) == 0.0
    assert np.max(np.abs(sal["ug"][:, -1])) == 0.0


def test_un_caudal_menor_llega_a_la_boca():
    sal = IIF.marchar(10.0, n_r=8)
    assert sal["mensaje"] == "boca"
    assert sal["z"][-1] > -1.0
    assert sal["P"][-1] < sal["P"][0]
    assert np.max(np.abs(sal["um"][:, -1])) == 0.0
    assert sal["phi"][-1, 0] > 0.5
    assert sal["um"][-1, 0] > sal["um"][-1, -2]
