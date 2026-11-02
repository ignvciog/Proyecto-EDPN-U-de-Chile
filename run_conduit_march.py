#!/usr/bin/env python3
"""Calbuco 2015 — conducto 1D sin IDA (solo numpy + módulos del repo)."""
from calbuco2015d import h2o1, overP1, radius1, T1, xi1
from RIconduitex5_5_march import RIconduitex5_5_march_f

if __name__ == "__main__":
    out = RIconduitex5_5_march_f(radius1, overP1, h2o1, T1, xi1)
    zsol, sol, count, vinicial, *_ = out
    print(f"\nTiro terminado: count={count}, vinicial={vinicial:.2f} m/s")
    print(f"  z_boca={zsol[-1]:.1f} m  P_boca={sol[-1, 0]/1e6:.3f} MPa  ug={sol[-1, 5]:.2f} m/s")
