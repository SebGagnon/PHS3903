import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
import random
from scipy.optimize import least_squares
from scipy.signal import hilbert
import time
from matplotlib.animation import FuncAnimation
import pickle

def laplacien_9_points(u, h):
    """laplacien à 9 points"""
    lap = np.zeros_like(u)
    lap[1:-1, 1:-1] = ( # u sans les frontieres
        4 * (
            u[2:, 1:-1] # u[i+1,j]
            + u[:-2, 1:-1] #u[i-1,j]
            + u[1:-1, 2:] #u[i,j+1]
            + u[1:-1, :-2] #u[i,j-1]
        )
        + (
            u[2:, 2:] # u[i+1,j+1]
            + u[2:, :-2] #u[i+1,j-1]
            + u[:-2, 2:] #u[i-1,j+1]
            + u[:-2, :-2] #u[i-1,j-1]
        )
        - 20 * u[1:-1, 1:-1] #u[i,j]
    ) / (6 * h**2)
    return lap

def source_gaussienne(nx, ny, t, x0, y0, sigma, A0, t0, tau):
    """construit la matrice de la source à chaque pas de temps"""
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")

    enveloppe_spatiale = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    enveloppe_temporelle = np.exp(-((t - t0)**2) / (2 * tau**2))

    return A0 * enveloppe_spatiale * enveloppe_temporelle

def conditions_frontieres(u):
    """applique les conditions frontières de Dirichlet aux bords"""
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    return u

def run_simulation(params):
    """lance une simulations à partir du dictionnaire des parametres
    Doit contenir : L , nx , ny , dt , t_total , rho , kappa , x0 , y0 , sigma , a0 , t0 , tau"""
    L = params["L"]
    nx = params["nx"]
    ny = params["ny"]
    dt = params["dt"]
    t_total = params["t_total"]
    rho = params["rho"]
    kappa = params["kappa"]

    x0 = params["x0"]
    y0 = params["y0"]
    sigma = params["sigma"]
    A0 = params["A0"]
    t0 = params["t0"]
    tau = params["tau"]
    gamma_eau = params["gamma_eau"]
    capteurs = params["capteurs"]

    gamma_pml = creer_pml(params)
    gamma_total = gamma_eau + gamma_pml

    c = np.sqrt(kappa / rho)
    h = L / nx
    nt = int(t_total / dt)

    u_nm1 = np.zeros((nx, ny))
    u_n = np.zeros((nx, ny))
    u_np1 = np.zeros((nx, ny))

    frames = []
    temps_frames = []

    pas_sauvegarde = 20
    a = gamma_total * dt / 2

    signaux_capteurs = [[] for _ in capteurs]
    temps = []

    for n in range(nt):
        t = n * dt

        lap = laplacien_9_points(u_n, h)
        
        source = source_gaussienne(nx, ny, t, x0, y0, sigma, A0, t0, tau)
        
        u_np1[1:-1, 1:-1] = (
            2 * u_n[1:-1, 1:-1]
                - (1 - a[1:-1, 1:-1]) * u_nm1[1:-1, 1:-1]
                + dt**2 * (
                c**2 * lap[1:-1, 1:-1]
                + source[1:-1, 1:-1]
                    )
                    ) / (1 + a[1:-1, 1:-1])

        u_np1 = conditions_frontieres(u_np1)

        temps.append(t)

        for i, (ix, iy) in enumerate(capteurs):
            signaux_capteurs[i].append(u_np1[ix, iy])

        if n % pas_sauvegarde == 0:
            frames.append(u_np1.copy())
            temps_frames.append(t)

        u_nm1[:, :] = u_n
        u_n[:, :] = u_np1

    return {
        "u_final": u_n,
        "frames": frames,
        "temps_frames": temps_frames,
        "c": c,
        "h": h,
        "nt": nt,
        "temps": np.array(temps),
        "signaux_capteurs": [np.array(signal) for signal in signaux_capteurs],
    }

def creer_pml(params):
    """Creation du pml"""
    nx = params["nx"]
    ny = params["ny"]
    epaisseur_ratio = params["epaisseur_pml_ratio"]
    puissance = params["puissance_pml"]
    gamma_max = params["gamma_max_pml"]

    # Convertit le ratio en épaisseur entière de PML
    epaisseur = int(epaisseur_ratio * nx)
    epaisseur = max(epaisseur, 1)

    gamma_pml = np.zeros((nx, ny))

    # Indices de ligne et de colonne pour chaque case
    lignes, cols = np.indices((nx, ny))

    # Distance minimale de chaque case à un bord
    dist = np.minimum.reduce([
        lignes,
        cols,
        nx - 1 - lignes,
        ny - 1 - cols
    ])

    # Sélectionne uniquement la zone du PML
    masque = dist < epaisseur

    # Normalisation
    s = (epaisseur - dist[masque]) / epaisseur

    # Fonction de puissance
    gamma_pml[masque] = gamma_max * s**puissance

    return gamma_pml

def animer_resultats_db(frames, temps_frames, L, duree_animation_ms=3000):
    fig, ax = plt.subplots(figsize=(8, 7))

    p_ref = 1e-6      # 1 µPa
    p_min = 1e-3      # plancher numérique = 1 mPa

    frames_db = [
        20 * np.log10(np.maximum(np.abs(frame), p_min) / p_ref)
        for frame in frames
    ]

    db_min = min(np.min(frame_db) for frame_db in frames_db)
    db_max = max(np.max(frame_db) for frame_db in frames_db)

    interval = duree_animation_ms / len(frames_db)

    img = ax.imshow(
        frames_db[0].T,
        origin="lower",
        cmap="RdBu_r",
        extent=[0, L, 0, L],
        vmin=db_min,
        vmax=db_max
    )

    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Pression [dB re 1 µPa]")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")

    def maj(k):
        img.set_array(frames_db[k].T)
        ax.set_title(f"Niveau de pression — t = {temps_frames[k]:.4f} s")
        return [img]

    ani = FuncAnimation(
        fig,
        maj,
        frames=len(frames_db),
        interval=interval,
        blit=False
    )

    plt.tight_layout()
    plt.show()
    return ani

def animer_resultats(frames, temps_frames, L, duree_animation_ms=3000):
    fig, ax = plt.subplots(figsize=(8, 7))

    frames_kpa = [frame / 1000.0 for frame in frames]

    amp_max = 0.25*max(np.max(np.abs(frame)) for frame in frames_kpa)
    if amp_max == 0:
        amp_max = 1.0

    interval = duree_animation_ms / len(frames_kpa)

    img = ax.imshow(
        frames_kpa[0].T,
        origin="lower",
        cmap="RdBu_r",
        extent=[0, L, 0, L],
        vmin=-amp_max,
        vmax=amp_max
    )

    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Pression instantanée [kPa]")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")

    def maj(k):
        img.set_array(frames_kpa[k].T)
        ax.set_title(f"Propagation de l'onde — t = {temps_frames[k]:.4f} s")
        return [img]

    ani = FuncAnimation(
        fig,
        maj,
        frames=len(frames_kpa),
        interval=interval,
        blit=False
    )

    plt.tight_layout()
    plt.show()

    return ani

params = {
    "L": 1000.0,
    "nx": 300,
    "ny": 300,
    "dt": 1e-3,
    "t_total": 1,
    "rho": 1000.0,
    "kappa": 2.2e9,
    "x0": 150,
    "y0": 150,
    "sigma": 1,
    "A0": 1000,
    "t0": 0.02,
    "tau": 0.015,
    "gamma_eau": 0.3,
    "epaisseur_pml_ratio": 0.2,
    "puissance_pml": 0.5,
    "gamma_max_pml": 50,
    "capteurs": [
    (175, 175),
    (175, 225),
    (175, 275),
                ],
}

resultats = run_simulation(params)
ani = animer_resultats(resultats['frames'],resultats['temps_frames'],params['L'])
temps = resultats["temps"]
signaux = resultats["signaux_capteurs"]

plt.figure(figsize=(8, 5))
for i, signal in enumerate(signaux):
    plt.plot(temps, signal, label=f"Capteur {i+1}")

plt.xlabel("Temps [s]")
plt.ylabel("Pression instantanée [Pa]")
plt.title("Signaux mesurés par les capteurs")
plt.legend()
plt.tight_layout()
plt.show()