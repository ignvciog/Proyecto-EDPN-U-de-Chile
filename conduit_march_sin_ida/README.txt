Conducto 1D estacionario (Calbuco) — sin IDA
============================================

Contenido mínimo para correr RIconduitex5_5_march (RK4 en z, solo numpy + matplotlib).

Instalación:
  pip install -r requirements-simple.txt

Pruebas:
  python3 test_march_1d.py
  python3 run_conduit_march.py

Notebook:
  Abrir mainconduit5_5_march.ipynb (Jupyter, VS Code o Cursor) y ejecutar todas las celdas.

Archivos:
  calbuco2015d.py      — parámetros Calbuco 2015
  density.py, viscosity.py, fvrel.py — propiedades del fundido
  RIconduitex5_5_march.py — modelo + método de tiro
  run_conduit_march.py, test_march_1d.py — scripts de ejemplo

Repositorio: Proyecto EDPN U de Chile (rama cursor/conduit-march-sin-ida-39c0)
