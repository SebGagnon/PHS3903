import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
#  Simulation 2D acoustique (grille décalée) + CPML (sigma/kappa/alpha)
#
#  Modèle (forme vitesse/pression) :
#    ∂vx/∂t = -(1/ρ) ∂p/∂x  - amortissement
#    ∂vy/∂t = -(1/ρ) ∂p/∂y  - amortissement
#    ∂p/∂t  = -K (∂vx/∂x + ∂vy/∂y) - amortissement
#
#  CPML "canonique" :
#    - profils sigma, kappa, alpha dans la zone PML
#    - variables mémoire psi pour chaque dérivée
#
#  Source :
#    - train de pulses gaussiens en temps au centre
#  Affichage :
#    - échelle fixe (pas de vmin/vmax dynamiques)
# ======================================================================


# ----------------------------------------------------------------------
# Petites fonctions utilitaires (pour garder le main propre)
# ----------------------------------------------------------------------

def profils_pml_1d(n, epaisseur, sigma_max, kappa_max, alpha_max, ordre=3):
    """
    Construit les profils 1D sigma/kappa/alpha sur une dimension (taille n).

    - epaisseur : nb de points PML à gauche + à droite
    - ordre     : ordre du polynôme (souvent 2-4)
    """
    sigma = np.zeros(n, dtype=np.float64)
    kappa = np.ones(n, dtype=np.float64)
    alpha = np.zeros(n, dtype=np.float64)

    for i in range(n):
        dist_bord = min(i, n - 1 - i)  # distance au bord le plus proche
        if dist_bord < epaisseur:
            x = (epaisseur - dist_bord) / epaisseur  # 0..1 "dans" la PML
            sigma[i] = sigma_max * (x ** ordre)
            kappa[i] = 1.0 + (kappa_max - 1.0) * (x ** ordre)
            # alpha fort à l'interface, qui décroît vers le bord
            alpha[i] = alpha_max * (1.0 - x)

    return sigma, kappa, alpha


def coeffs_cpml(sigma, kappa, alpha, dt):
    """
    Coefficients CPML (a, b, 1/kappa) utilisés avec les variables mémoire psi.

    psi <- b*psi + a*d
    d_hat = (1/kappa)*d + psi
    """
    inv_kappa = 1.0 / kappa
    b = np.exp(-(sigma * inv_kappa + alpha) * dt)

    denom = sigma + kappa * alpha
    a = np.zeros_like(sigma)

    mask = denom > 0
    a[mask] = (sigma[mask] * (b[mask] - 1.0) / denom[mask]) * inv_kappa[mask]

    return a, b, inv_kappa


def broadcast_1d_vers_2d(profil_1d, n_autre, axe):
    """
    Étale un profil 1D en 2D par répétition.
    - axe=0 : profil varie en x (shape -> (nx, ny))
    - axe=1 : profil varie en y (shape -> (nx, ny))
    """
    if axe == 0:
        return np.repeat(profil_1d[:, None], n_autre, axis=1)
    else:
        return np.repeat(profil_1d[None, :], n_autre, axis=0)


def train_pulses_gaussiens(t, centres, tau):
    """
    Somme de gaussiennes en temps.
    tau = écart-type (en secondes) : plus grand => pulse plus long
    """
    inv2tau2 = 1.0 / (2.0 * tau * tau)
    s = 0.0
    for tc in centres:
        dtc = t - tc
        s += np.exp(-(dtc * dtc) * inv2tau2)
    return s


# ----------------------------------------------------------------------
# Paramètres du problème
# ----------------------------------------------------------------------

# Grille
nx, ny = 200, 200
dx = dy = 0.5

# Milieu
c = 3.0
rho = 0.01
K = rho * c**2

# Temps (CFL)
dt = 0.45 * dx / (5 * np.sqrt(2))
steps = 5000

# (Optionnel) perte uniforme "physique" (ici, juste gardée en param)
gamma = 0.01


# ----------------------------------------------------------------------
# Champs (grille décalée)
# ----------------------------------------------------------------------
# p : centres de cellules (nx, ny)
# vx: faces normales à x  (nx+1, ny)
# vy: faces normales à y  (nx, ny+1)
p  = np.zeros((nx, ny), dtype=np.float64)
vx = np.zeros((nx + 1, ny), dtype=np.float64)
vy = np.zeros((nx, ny + 1), dtype=np.float64)


# ----------------------------------------------------------------------
# Mise en place CPML
# ----------------------------------------------------------------------
ep_pml = 30
ordre = 3
sigma_max = 2.5
kappa_max = 6.0
alpha_max = 0.15

# Profils 1D aux bons emplacements (attention aux tailles selon le staggering)
sx_p,  kx_p,  ax_p  = profils_pml_1d(nx,     ep_pml, sigma_max, kappa_max, alpha_max, ordre=ordre)
sy_p,  ky_p,  ay_p  = profils_pml_1d(ny,     ep_pml, sigma_max, kappa_max, alpha_max, ordre=ordre)
sx_vx, kx_vx, ax_vx = profils_pml_1d(nx + 1, ep_pml, sigma_max, kappa_max, alpha_max, ordre=ordre)
sy_vy, ky_vy, ay_vy = profils_pml_1d(ny + 1, ep_pml, sigma_max, kappa_max, alpha_max, ordre=ordre)

# Broadcast 2D (on colle à la forme de chaque champ)
sigx_p = broadcast_1d_vers_2d(sx_p,  ny, axe=0)
kapx_p = broadcast_1d_vers_2d(kx_p,  ny, axe=0)
alpx_p = broadcast_1d_vers_2d(ax_p,  ny, axe=0)

sigy_p = broadcast_1d_vers_2d(sy_p,  nx, axe=1)
kapy_p = broadcast_1d_vers_2d(ky_p,  nx, axe=1)
alpy_p = broadcast_1d_vers_2d(ay_p,  nx, axe=1)

sigx_vx = broadcast_1d_vers_2d(sx_vx, ny, axe=0)
kapx_vx = broadcast_1d_vers_2d(kx_vx, ny, axe=0)
alpx_vx = broadcast_1d_vers_2d(ax_vx, ny, axe=0)

sigy_vy = broadcast_1d_vers_2d(sy_vy, nx, axe=1)
kapy_vy = broadcast_1d_vers_2d(ky_vy, nx, axe=1)
alpy_vy = broadcast_1d_vers_2d(ay_vy, nx, axe=1)

# Coeffs CPML (un set par dérivée)
# - dp/dx évalué sur vx
a_vx, b_vx, invkap_vx = coeffs_cpml(sigx_vx, kapx_vx, alpx_vx, dt)
# - dp/dy évalué sur vy
a_vy, b_vy, invkap_vy = coeffs_cpml(sigy_vy, kapy_vy, alpy_vy, dt)
# - dvx/dx évalué sur p
a_px, b_px, invkap_px = coeffs_cpml(sigx_p,  kapx_p,  alpx_p,  dt)
# - dvy/dy évalué sur p
a_py, b_py, invkap_py = coeffs_cpml(sigy_p,  kapy_p,  alpy_p,  dt)

# Variables mémoire CPML (mêmes tailles que les dérivées "corrigées")
psi_vx = np.zeros_like(vx)  # pour dp/dx (sur vx)
psi_vy = np.zeros_like(vy)  # pour dp/dy (sur vy)
psi_px = np.zeros_like(p)   # pour dvx/dx (sur p)
psi_py = np.zeros_like(p)   # pour dvy/dy (sur p)


# ----------------------------------------------------------------------
# Source (train de pulses)
# ----------------------------------------------------------------------
x0, y0 = nx // 2, ny // 2
A_src = 1.0
tau = 10 * dt
centres_pulses = [100 * dt, 460 * dt]


# ----------------------------------------------------------------------
# Affichage (échelle fixe)
# ----------------------------------------------------------------------
A0 = 0.05 * A_src

plt.ion()
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(p, origin="lower", cmap="RdBu_r", vmin=-A0, vmax=A0)
ax.set_title("Pression p (2D) - CPML")
plt.colorbar(im, ax=ax)


# ----------------------------------------------------------------------
# Boucle principale
# ----------------------------------------------------------------------
for it in range(steps):

    # --- Mise à jour vx (dp/dx est défini sur vx[1:nx, :])
    dpdx = (p[1:, :] - p[:-1, :]) / dx               # (nx-1, ny)
    psi_vx[1:nx, :] = b_vx[1:nx, :] * psi_vx[1:nx, :] + a_vx[1:nx, :] * dpdx
    dpdx_hat = invkap_vx[1:nx, :] * dpdx + psi_vx[1:nx, :]

    vx[1:nx, :] += dt * (
        -(1.0 / rho) * dpdx_hat
        - (sigx_vx[1:nx, :] + alpx_vx[1:nx, :]) * vx[1:nx, :]
    )

    # --- Mise à jour vy (dp/dy est défini sur vy[:, 1:ny])
    dpdy = (p[:, 1:] - p[:, :-1]) / dy               # (nx, ny-1)
    psi_vy[:, 1:ny] = b_vy[:, 1:ny] * psi_vy[:, 1:ny] + a_vy[:, 1:ny] * dpdy
    dpdy_hat = invkap_vy[:, 1:ny] * dpdy + psi_vy[:, 1:ny]

    vy[:, 1:ny] += dt * (
        -(1.0 / rho) * dpdy_hat
        - (sigy_vy[:, 1:ny] + alpy_vy[:, 1:ny]) * vy[:, 1:ny]
    )

    # --- Mise à jour p (divergence au centre)
    dvxdx = (vx[1:, :] - vx[:-1, :]) / dx            # (nx, ny)
    dvydy = (vy[:, 1:] - vy[:, :-1]) / dy            # (nx, ny)

    psi_px = b_px * psi_px + a_px * dvxdx
    psi_py = b_py * psi_py + a_py * dvydy

    div_hat = (invkap_px * dvxdx + psi_px) + (invkap_py * dvydy + psi_py)

    p += dt * (
        -K * div_hat
        - (sigx_p + alpx_p + sigy_p + alpy_p) * p
    )

    # --- Injection de la source (pulses gaussiens)
    t = it * dt
    p[x0, y0] += A_src * train_pulses_gaussiens(t, centres_pulses, tau)

    # --- Rafraîchissement affichage
    if it % 5 == 0:
        im.set_data(p)
        plt.pause(0.001)

plt.ioff()
plt.show()