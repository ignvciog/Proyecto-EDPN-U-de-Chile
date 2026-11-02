"""RIconduitex5_5_march: import + un tramo corto en z (sin IDA)."""
import numpy as np

from calbuco2015d import H, R, beta, C1, g, xi1, h2o1, overP1, radius1, T1, rcrust, limperl, limperh, lsup, linf
import RIconduitex5_5_march as M


def _setup_globals():
    M.h2o = h2o1
    M.wr = float(radius1)
    M.overP = float(overP1)
    M.T = float(T1)
    M.Tc = T1 - 273.15
    M.co = h2o1 / 100.0
    M.xi = float(xi1)
    M.cg = 8
    M.Pi = rcrust * g * abs(H) + overP1
    dis = min(C1 * M.Pi ** beta, M.co)
    from density import density
    from calbuco2015d import sio2, tio2, al2o3, feo, mgo, cao, na2o, k2o

    M.rho_m = density(sio2, tio2, al2o3, feo, mgo, cao, na2o, k2o, dis * 100, M.Tc, M.Pi / 1e6)
    M.rho_ti = M.rho_m
    M.q = 10.0 * M.rho_ti
    M.vinicial = 10.0
    M.Nd = 1e8
    M.xfinal = xi1
    M.n_eq = 1
    M.phicrit = 0.75
    M.limphi1 = limperl
    M.limphi2 = limperh
    M.xmax = 0.5


def test_march_segment_finite():
    _setup_globals()
    y0 = np.array([M.Pi, 0.05, M.Nd, M.xi, 10.0, 10.0])
    Y, Z = M.solv(H, -2000.0, y0, None, 1e-6, 1e-6, 400, 1, cond=None)
    assert Y.ndim == 2 and Y.shape[1] == 6
    assert Z.shape[0] == Y.shape[0]
    assert np.all(np.isfinite(Y))
    assert Z[0] == H and Z[-1] > H


if __name__ == "__main__":
    test_march_segment_finite()
    print("ok")
