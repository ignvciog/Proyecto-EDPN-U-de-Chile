"""La marcha 2D con Q constante conserva el caudal y deja la pared en reposo."""
import numpy as np

from RIconduit2D_FD import _fg, _rho_g, _rho_m
import RIconduit2D_Q as Q


def test_marcha_conserva_Q_y_la_pared():
    sal = Q.marchar(1.0, n_r=8, dz=200.0)
    assert sal["n_pasos"] >= 5
    assert np.isfinite(sal["P"]).all()
    assert np.isfinite(sal["um"]).all()
    assert np.isfinite(sal["ug"]).all()
    rel = np.max(np.abs(sal["Q"] - sal["Q_objetivo"]) / sal["Q_objetivo"])
    assert rel < 1e-8
    assert np.max(np.abs(sal["um"][:, -1])) == 0.0
    assert np.max(np.abs(sal["ug"][:, -1])) == 0.0
    # Perfil de tubo: el eje va más rápido que el nodo junto a la pared.
    assert sal["um"][-1, 0] > sal["um"][-1, -2]
    assert sal["P"][-1] < sal["P"][0]


def test_masa_del_fundido_en_un_paso():
    sal = Q.marchar(1.0, n_r=8, dz=200.0)
    i = 0
    p0, p1 = float(sal["P"][-2]), float(sal["P"][-1])
    phi0, phi1 = float(sal["phi"][-2, i]), float(sal["phi"][-1, i])
    um0, um1 = float(sal["um"][-2, i]), float(sal["um"][-1, i])
    ug1 = float(sal["ug"][-1, i])
    n0 = _fg(p0, float(sal["xi"][-2, i]))
    n1 = _fg(p1, float(sal["xi"][-1, i]))
    fm0 = _rho_m(p0) * (1.0 - phi0) * um0
    fm1 = _rho_m(p1) * (1.0 - phi1) * um1
    jz = _rho_m(p1) * (1.0 - phi1) * um1 + _rho_g(p1) * phi1 * ug1
    residuo = abs((fm1 - fm0) + jz * (n1 - n0)) / max(abs(fm0), 1.0)
    assert residuo < 0.02
