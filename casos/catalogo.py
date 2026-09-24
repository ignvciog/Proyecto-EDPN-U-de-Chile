"""
Catalogo de casos del paper (Castruccio et al. 2025) y de calbuco2015d.py.

Los valores son los de los .m / .py, no reajustados. `model` va en minusculas
porque fvrel.py compara contra 'em2s', no 'EM2s'.
"""

from __future__ import annotations

CASOS: dict[str, dict] = {
    "calbuco2015": dict(
        nombre="Calbuco 2015",
        estilo_obs="sub-Pliniana",
        sio2=62.72, tio2=1.0, al2o3=17.0, feo=5.44, mno=0.15, mgo=2.19,
        cao=5.22, na2o=4.44, k2o=1.2, p2o5=0.1, f2o=0.1,
        h2o=4.0, C1=4.11e-6, beta=0.5,
        Tc=970.0, xi=0.25, xmax=0.5, tcar=2 * 3600,
        model="em2s", ar1=4.0, ar2=8.0,
        H=-7000.0, geometry="cylinder", radius1=16.0, dl=100.0,
        overP1=5.0e6, rcrust=2600.0, Fc=1.0,
    ),
    "vesuvius79": dict(
        nombre="Vesuvius 79",
        estilo_obs="Pliniana (pumita blanca)",
        sio2=58.19, tio2=0.07, al2o3=21.77, feo=1.15, mno=0.07, mgo=0.02,
        cao=1.2, na2o=5.39, k2o=11.45, p2o5=0.1, f2o=0.1,
        h2o=5.0, C1=4.11e-6, beta=0.5,
        Tc=850.0, xi=0.2, xmax=0.3, tcar=2 * 3600,
        model="em2s", ar1=4.0, ar2=8.0,
        H=-7500.0, geometry="cylinder", radius1=16.0, dl=100.0,
        overP1=5.0e6, rcrust=2700.0, Fc=1.0,
        nota="R y dP no vienen en el .m; se usan 16 m / 5 MPa como partida",
    ),
    "huaynaputina1600": dict(
        nombre="Huaynaputina 1600",
        estilo_obs="Pliniana",
        sio2=73.3, tio2=0.34, al2o3=14.84, feo=1.47, mno=0.0, mgo=0.32,
        cao=1.45, na2o=4.36, k2o=3.82, p2o5=0.1, f2o=0.0,
        h2o=6.0, C1=4.11e-6, beta=0.5,
        Tc=850.0, xi=0.17, xmax=0.3, tcar=2 * 3600,
        model="em2s", ar1=4.0, ar2=8.0,
        H=-8000.0, geometry="cylinder", radius1=16.0, dl=100.0,
        overP1=5.0e6, rcrust=2700.0, Fc=1.0,
        nota="R y dP no vienen en el .m; se usan 16 m / 5 MPa como partida",
    ),
    "caulle2011": dict(
        nombre="Cordon Caulle 2011",
        estilo_obs="sub-Pliniana -> lava",
        sio2=72.1, tio2=0.49, al2o3=14.00, feo=3.83, mno=0.13, mgo=0.49,
        cao=1.56, na2o=4.17, k2o=2.98, p2o5=0.07, f2o=0.0,
        h2o=4.0, C1=4.11e-6, beta=0.5,
        Tc=900.0, xi=0.05, xmax=0.10, tcar=2 * 3600,
        model="em2s", ar1=4.0, ar2=8.0,
        H=-5000.0, geometry="cylinder", radius1=7.0, dl=100.0,
        overP1=7.0e6, rcrust=2500.0, Fc=1.0,
        nota="el .m dice namevolc=St Helens 2004; H2O=4 (paper: 3.5)",
    ),
    "merapi2010": dict(
        nombre="Merapi 2010",
        estilo_obs="explosiva + domo",
        sio2=62.3, tio2=0.43, al2o3=16.66, feo=4.86, mno=0.2, mgo=0.83,
        cao=4.41, na2o=4.18, k2o=5.59, p2o5=0.15, f2o=0.1384,
        h2o=4.0, C1=4.11e-6, beta=0.5,
        Tc=970.0, xi=0.5, xmax=0.6, tcar=2 * 3600,
        model="em2s", ar1=4.0, ar2=8.0,
        H=-3500.0, geometry="cylinder", radius1=5.747, dl=100.0,
        overP1=9.7574e6, rcrust=2500.0, Fc=1.0,
    ),
    "sthelens2004": dict(
        nombre="St. Helens 2004",
        estilo_obs="domo (efusiva)",
        sio2=73.5, tio2=0.2, al2o3=13.8, feo=1.0, mno=0.02, mgo=0.08,
        cao=1.25, na2o=4.18, k2o=3.38, p2o5=0.05, f2o=0.0,
        h2o=3.0, C1=4.11e-6, beta=0.5,
        Tc=950.0, xi=0.4, xmax=0.7, tcar=3600,
        model="em2s", ar1=4.0, ar2=8.0,
        H=-5000.0, geometry="cylinder", radius1=12.11, dl=100.0,
        overP1=16.157e6, rcrust=2500.0, Fc=1.0,
    ),
    "villarrica2015": dict(
        nombre="Villarrica 2015",
        estilo_obs="hawaiana / estromboliana",
        sio2=54.14, tio2=1.78, al2o3=13.93, feo=11.32, mno=0.19, mgo=4.99,
        cao=8.84, na2o=3.38, k2o=0.95, p2o5=0.43, f2o=0.1,
        h2o=1.5, C1=6.80172e-8, beta=0.7,
        Tc=1130.0, xi=0.25, xmax=0.45, tcar=2 * 3600,
        model="em2s", ar1=4.0, ar2=8.0,
        H=-6000.0, geometry="dyke", radius1=0.9854, dl=200.0,
        overP1=1.8759e6, rcrust=2600.0, Fc=0.75,
    ),
    "pinatubo1991": dict(
        nombre="Pinatubo 1991",
        estilo_obs="Pliniana",
        sio2=69.37, tio2=0.26, al2o3=14.87, feo=2.56, mno=0.04, mgo=1.51,
        cao=3.6, na2o=3.47, k2o=2.36, p2o5=0.0, f2o=0.0,
        h2o=5.5, C1=4.11e-6, beta=0.5,
        Tc=780.0, xi=0.15, xmax=0.55, tcar=2 * 3600,
        model="em2s", ar1=4.0, ar2=8.0,
        H=-8000.0, geometry="cylinder", radius1=90.0, dl=100.0,
        overP1=15.97e6, rcrust=2600.0, Fc=1.0,
    ),
    "quizapu1932": dict(
        nombre="Quizapu 1932",
        estilo_obs="Pliniana",
        sio2=71.59, tio2=0.34, al2o3=15.18, feo=1.67, mno=0.07, mgo=0.34,
        cao=1.11, na2o=5.43, k2o=4.09, p2o5=0.04, f2o=0.0,
        h2o=5.0, C1=4.11e-6, beta=0.5,
        Tc=865.0, xi=0.15, xmax=0.5, tcar=2 * 3600,
        model="em2s", ar1=4.0, ar2=8.0,
        H=-5500.0, geometry="cylinder", radius1=30.0, dl=100.0,
        overP1=8.0e6, rcrust=2600.0, Fc=1.0,
    ),
}


def listar() -> None:
    print(f"{'clave':<20} {'nombre':<28} {'estilo obs.':<28} {'R':>8} {'dP MPa':>8}")
    for k, c in CASOS.items():
        print(f"{k:<20} {c['nombre']:<28} {c['estilo_obs']:<28} "
              f"{c['radius1']:8.2f} {c['overP1']/1e6:8.2f}")
