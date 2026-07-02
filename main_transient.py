"""
main_transient.py
=================
Script principal para el modelo de conducto volcánico 1D TRANSIENTE.
Usa los parámetros de calbuco2015d.py y el modelo RIconduit1D_transient.py.

Flujo:
  1. Importa parámetros desde calbuco2015d.
  2. Corre el modelo ESTACIONARIO (RIconduitex5_5_f) para obtener vinicial_conv.
  3. Corre el modelo TRANSIENTE con vinicial_conv y τ_relax = 1 s (IMEX).
  4. Guarda resultados y genera gráficas.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import contextlib, os

# ── Parámetros del caso (Calbuco 2015) ───────────────────────────────────────
from calbuco2015d import (
    radius1, overP1, h2o1, T1, xi1,
    phicrit, H,
)

from RIconduitex5_5 import RIconduitex5_5_f
from RIconduit1D_transient import RIconduit1D_transient_f

# ── 1. Modelo estacionario → obtener vinicial_conv ───────────────────────────
print("=" * 60)
print("Paso 1: Modelo estacionario (RIconduitex5_5_f)...")
print("=" * 60)

with open(os.devnull, 'w') as f:
    with contextlib.redirect_stdout(f):
        result_stat = RIconduitex5_5_f(radius1, overP1, h2o1, T1, xi1)

zsol, sol, count, vinicial_conv, rho_m, *_ = result_stat
print(f"  vinicial convergido : {vinicial_conv:.4f} m/s")
print(f"  count               : {count}")
print()

# ── 2. Modelo transiente ──────────────────────────────────────────────────────
print("=" * 60)
print("Paso 2: Modelo transiente (RIconduit1D_transient_f)...")
print("=" * 60)

out = RIconduit1D_transient_f(
    radius            = radius1,
    Pressure          = overP1,
    wt                = h2o1,
    Temperature       = T1,
    content_crystal   = xi1,
    vinicial_override = vinicial_conv,   # rama explosiva
    phicrit_override  = phicrit,         # 0.8 (Calbuco)
    tau_relax         = 1.0,             # IMEX: casi instantáneo → replica estacionario
    N_z               = 200,
    T_max             = 3600.0,
    save_every        = 50,
    CFL               = 0.4,
)

# ── 3. Extraer resultados ─────────────────────────────────────────────────────
t_save   = out['t_save']        # [N_t]  tiempos guardados [s]
z        = out['z']             # [N_z]  profundidades [m]
phi_hist = out['phi_hist']      # [N_t, N_z]
P_hist   = out['P_hist']        # [N_t, N_z]
um_hist  = out['um_hist']       # [N_t, N_z]
ug_hist  = out['ug_hist']       # [N_t, N_z]

z_km = z / 1e3   # convertir a km

# ── 4. Gráficas ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
fig.suptitle(f"Evolución temporal del conducto — Calbuco 2015\n"
             f"τ_relax = 1 s  |  φ_crit = {phicrit}  |  r = {radius1} m  |  "
             f"q = {out['q']:.1f} kg/m²/s", fontsize=11)

# Colormap: tiempo
cmap   = cm.plasma
norm   = plt.Normalize(vmin=t_save.min(), vmax=t_save.max())
sm     = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Índices para etiquetas
idx_0    = 0
idx_mid  = len(t_save) // 2
idx_end  = len(t_save) - 1

# --- φ(z,t) ------------------------------------------------------------------
ax = axes[0]
for i, t in enumerate(t_save):
    color = cmap(norm(t))
    lw    = 2.0 if i in (idx_0, idx_mid, idx_end) else 0.6
    ax.plot(phi_hist[i], z_km, color=color, linewidth=lw)

# Línea φ_crit
ax.axvline(phicrit, color='red', linestyle='--', linewidth=1.2,
           label=f'φ_crit = {phicrit}')
# Etiquetas de tiempo seleccionadas
for idx in (idx_0, idx_mid, idx_end):
    ax.plot(phi_hist[idx, -1], z_km[-1], 'o', color=cmap(norm(t_save[idx])),
            markersize=5)

ax.set_xlabel('φ (fracción de gas/burbujas)')
ax.set_ylabel('Profundidad z (km)')
ax.set_title('φ(z, t)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
for idx, label in zip((idx_0, idx_mid, idx_end),
                      (f't={t_save[idx_0]:.0f}s',
                       f't={t_save[idx_mid]:.0f}s',
                       f't={t_save[idx_end]:.0f}s')):
    ax.annotate(label, xy=(phi_hist[idx, -1], z_km[-1]),
                xytext=(5, -10), textcoords='offset points', fontsize=7,
                color=cmap(norm(t_save[idx])))

# --- P(z,t) ------------------------------------------------------------------
ax = axes[1]
for i, t in enumerate(t_save):
    color = cmap(norm(t))
    lw    = 2.0 if i in (idx_0, idx_mid, idx_end) else 0.6
    ax.semilogx(P_hist[i] / 1e6, z_km, color=color, linewidth=lw)

ax.set_xlabel('Presión P [MPa]')
ax.set_title('P(z, t)')
ax.grid(True, alpha=0.3, which='both')

# --- um(z,t) -----------------------------------------------------------------
ax = axes[2]
for i, t in enumerate(t_save):
    color = cmap(norm(t))
    lw    = 2.0 if i in (idx_0, idx_mid, idx_end) else 0.6
    ax.plot(um_hist[i], z_km, color=color, linewidth=lw)

ax.set_xlabel('$u_m$ velocidad fundido (m/s)')
ax.set_title('$u_m$(z, t)')
ax.grid(True, alpha=0.3)

# Colorbar
cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
cbar.set_label('Tiempo t [s]')

plt.tight_layout()
plt.savefig('transient_calbuco2015.png', dpi=150, bbox_inches='tight')
print("\nGráfica guardada: transient_calbuco2015.png")
plt.show()

# ── 5. Resumen final ──────────────────────────────────────────────────────────
phi_final = phi_hist[-1]
z_frag_idx = np.where(phi_final >= phicrit)[0]

print("\n" + "=" * 60)
print("Resumen del estado final (t = {:.0f} s):".format(t_save[-1]))
print("=" * 60)
print(f"  φ_max           = {phi_final.max():.4f}")
print(f"  φ_vent (z=0)    = {phi_final[-1]:.4f}")
print(f"  nfrag           = {len(z_frag_idx)} nodos fragmentados")
if len(z_frag_idx) > 0:
    z_frag_m = z[z_frag_idx[0]]
    print(f"  Frente de frag  = {z_frag_m/1e3:.2f} km")
print(f"  P_vent          = {P_hist[-1, -1]/1e5:.2f} bar")
print(f"  um_vent         = {um_hist[-1, -1]:.1f} m/s")
print(f"  ug_vent         = {ug_hist[-1, -1]:.1f} m/s")
