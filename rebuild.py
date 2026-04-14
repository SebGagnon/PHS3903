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
from matplotlib.patches import Circle

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

def sourcetot(params, t):
    """Calcule la source totale comme somme des pulses des capteurs."""
    nx = params["nx"]
    ny = params["ny"]
    sigma = params["sigma"]
    A0 = params["A0"]
    tau = params["tau"]

    pulses = construction_pulses(params)

    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    source = np.zeros((nx, ny))

    for x0, y0, t0 in pulses:
        enveloppe_spatiale = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
        enveloppe_temporelle = np.exp(-((t - t0)**2) / (2 * tau**2))
        source += A0 * enveloppe_spatiale * enveloppe_temporelle

    return source

def conditions_frontieres(u):
    """applique les conditions frontières de Dirichlet aux bords"""
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    return u

def positions_capteurs(params):
    """Retourne seulement les positions des capteurs."""
    return [capteur["position"] for capteur in params["capteurs"]]

def couleurs_capteurs(params):
    """Liste de couleur pour les capteurs"""
    base = ["red", "blue", "green", "magenta", "orange", "white"]
    n = len(params["capteurs"])
    return [base[i % len(base)] for i in range(n)]



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

def masque_objet_circulaire(params, t):
    """Crée le masque circulaire de l'objet à l'instant t."""
    if "objet" not in params or params["objet"] is None:
        return np.zeros((params["nx"], params["ny"]), dtype=bool)

    nx = params["nx"]
    ny = params["ny"]

    x_c, y_c = position_objet_t(params, t)
    rayon = params["objet"]["rayon"]

    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    masque = (x - x_c)**2 + (y - y_c)**2 <= rayon**2

    return masque

def champ_vitesse(params, t):
    """
    champ de vitesse, avec ou sans objet. La vitesse est calculée à partir de rho et kappa dans les parametres
    """
    c_eau = np.sqrt(params["kappa"] / params["rho"])
    c = np.full((params["nx"], params["ny"]), c_eau)

    if "objet" in params and params["objet"] is not None:
        masque = masque_objet_circulaire(params, t)

        rho_objet = params["objet"]["rho"]
        kappa_objet = params["objet"]["kappa"]
        c_objet = np.sqrt(kappa_objet / rho_objet)

        c[masque] = c_objet

    return c

def champ_gamma_materiau(params, t):
    """
    Ajoute le gamma de l'objet si il y a un objet, sinon jsute celui de l'eau
    """
    gamma = np.full((params["nx"], params["ny"]), params["gamma_eau"])

    if "objet" in params and params["objet"] is not None:
        masque = masque_objet_circulaire(params, t)
        gamma_objet = params["objet"]["gamma"]
        gamma[masque] = gamma_objet

    return gamma

def position_objet_t(params, t):
    """Retourne la position de l'objet à l'instant t, en indices de grille."""
    if "objet" not in params or params["objet"] is None:
        return None

    x0, y0 = params["objet"]["position"]
    vx, vy = params["objet"].get("vitesse_m_s", (0.0, 0.0))

    hx = params["L"] / params["nx"]
    hy = params["L"] / params["ny"]

    x_t = x0 + (vx / hx) * t
    y_t = y0 + (vy / hy) * t

    return x_t, y_t

def trajectoire_objet_en_metres(params, trajectoire_indices):
    """Convertit une trajectoire en indices de grille vers des positions en mètres."""
    hx = params["L"] / params["nx"]
    hy = params["L"] / params["ny"]

    trajectoire_m = []
    for x_idx, y_idx in trajectoire_indices:
        x_m = (x_idx + 0.5) * hx
        y_m = (y_idx + 0.5) * hy
        trajectoire_m.append((x_m, y_m))

    return np.array(trajectoire_m)

def construction_pulses(params):
    """Construit la liste des pulses (ix, iy, t0) à partir de params['capteurs']."""
    pulses = []

    for capteur in params["capteurs"]:
        ix, iy = capteur["position"]
        for t0 in capteur["emissions"]:
            pulses.append((ix, iy, t0))

    return pulses

def creer_patch_objet(ax, params, t=0.0):
    """Crée le patch de l'objet à l'instant t."""
    if "objet" not in params or params["objet"] is None:
        return None

    nx = params["nx"]
    ny = params["ny"]
    L = params["L"]

    x_obj, y_obj = position_objet_t(params, t)
    rayon = params["objet"]["rayon"]

    hx = L / nx
    hy = L / ny

    x_m = (x_obj + 0.5) * hx
    y_m = (y_obj + 0.5) * hy
    rayon_m = rayon * hx

    patch = Circle(
        (x_m, y_m),
        rayon_m,
        edgecolor="black",
        facecolor="gray",
        alpha=0.25,
        linewidth=2
    )

    ax.add_patch(patch)
    return patch

def animer_resultats_db(params, resultats, duree_animation_ms=3000):
    """Animation du champ en dB."""
    frames = resultats["frames"]
    temps_frames = resultats["temps_frames"]
    L = params["L"]
    nx = params["nx"]
    ny = params["ny"]

    fig, ax = plt.subplots(figsize=(8, 7))

    p_ref = 1e-6
    p_min = 1e-3

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

    hx = L / nx
    hy = L / ny

    couleurs = couleurs_capteurs(params)

    for i, (capteur, couleur) in enumerate(zip(params["capteurs"], couleurs)):
        ix, iy = capteur["position"]
        x_m = (ix + 0.5) * hx
        y_m = (iy + 0.5) * hy

        ax.plot(x_m, y_m, "o", color=couleur, markersize=7)
        ax.text(
            x_m + 10, y_m + 10,
            f"C{i+1}",
            color=couleur,
            fontsize=10,
            weight="bold"
        )

    patch_objet = creer_patch_objet(ax, params, t=0.0)

    def maj(k):
        img.set_array(frames_db[k].T)
        ax.set_title(f"Pression au temps t = {temps_frames[k]:.4f} s")

        if patch_objet is not None:
            x_obj, y_obj = position_objet_t(params, temps_frames[k])
            hx = L / nx
            hy = L / ny
            patch_objet.center = ((x_obj + 0.5) * hx, (y_obj + 0.5) * hy)

        if patch_objet is not None:
            return [img, patch_objet]
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

def animer_resultats(params, resultats, duree_animation_ms=3000):
    """Animation, prends les params et les resultats d'une simul"""
    frames = resultats["frames"]
    temps_frames = resultats["temps_frames"]
    L = params["L"]
    nx = params["nx"]
    ny = params["ny"]

    fig, ax = plt.subplots(figsize=(8, 7))

    frames_kpa = [frame / 1000.0 for frame in frames]

    amp_max = 0.25 * max(np.max(np.abs(frame)) for frame in frames_kpa)
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

    hx = L / nx
    hy = L / ny

    couleurs = couleurs_capteurs(params)

    for i, (capteur, couleur) in enumerate(zip(params["capteurs"], couleurs)):
        ix, iy = capteur["position"]
        x_m = (ix + 0.5) * hx
        y_m = (iy + 0.5) * hy

        ax.plot(x_m, y_m, "o", color=couleur, markersize=7)
        ax.text(
            x_m + 10, y_m + 10,
            f"C{i+1}",
            color=couleur,
            fontsize=10,
            weight="bold"
        )

    patch_objet = creer_patch_objet(ax, params, t=0.0)

    def maj(k):
        img.set_array(frames_kpa[k].T)
        ax.set_title(f"Pression au temps t = {temps_frames[k]:.4f} s")

        if patch_objet is not None:
            x_obj, y_obj = position_objet_t(params, temps_frames[k])
            hx = L / nx
            hy = L / ny
            patch_objet.center = ((x_obj + 0.5) * hx, (y_obj + 0.5) * hy)

        if patch_objet is not None:
            return [img, patch_objet]
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

def plot_signaux(params, resultats):
    """Plot des mesures de chaque capteur."""
    temps = resultats["temps"]
    signaux = resultats["signaux_capteurs"]
    couleurs = couleurs_capteurs(params)

    plt.figure(figsize=(8, 5))

    for i, (signal, couleur) in enumerate(zip(signaux, couleurs)):
        plt.plot(temps, signal, color=couleur, label=f"Capteur {i+1}")

    plt.xlabel("Temps [s]")
    plt.ylabel("Pression instantanée [Pa]")
    plt.title("Signaux mesurés par les capteurs")
    plt.legend()
    plt.tight_layout()
    plt.show()

def tracer_trajectoire_objet(params, resultats):
    """Trace la trajectoire de l'objet en mètres."""
    traj = resultats["trajectoire_objet_m"]

    if traj is None or len(traj) == 0:
        print("Aucune trajectoire d'objet à tracer.")
        return

    plt.figure(figsize=(6, 6))
    plt.plot(traj[:, 0], traj[:, 1], "k-", label="Trajectoire")
    plt.plot(traj[0, 0], traj[0, 1], "go", label="Départ")
    plt.plot(traj[-1, 0], traj[-1, 1], "ro", label="Fin")

    for i, capteur in enumerate(params["capteurs"]):
        ix, iy = capteur["position"]
        hx = params["L"] / params["nx"]
        hy = params["L"] / params["ny"]
        x_m = (ix + 0.5) * hx
        y_m = (iy + 0.5) * hy
        plt.plot(x_m, y_m, "o", label=f"Capteur {i+1}")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Trajectoire de l'objet")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_simulation(params):

    """lance une simulation à partir du dictionnaire des parametres
    """

    L = params["L"]
    nx = params["nx"]
    ny = params["ny"]
    dt = params["dt"]
    t_total = params["t_total"]

    
    capteurs = positions_capteurs(params)

    gamma_pml = creer_pml(params)

    h = L / nx
    nt = int(t_total / dt)

    u_nm1 = np.zeros((nx, ny))
    u_n = np.zeros((nx, ny))
    u_np1 = np.zeros((nx, ny))

    frames = []
    temps_frames = []
    pas_sauvegarde = 20
    signaux_capteurs = [[] for _ in capteurs]
    temps = []
    trajectoire_objet = []

    for n in range(nt):

        t = n * dt

        if params.get("objet") is not None:
            trajectoire_objet.append(position_objet_t(params, t))

        c = champ_vitesse(params, t)
        gamma_materiau = champ_gamma_materiau(params, t)
        gamma_total = gamma_materiau + gamma_pml
        a = gamma_total * dt / 2

        lap = laplacien_9_points(u_n, h)
        
        source = sourcetot(params,t)
        
        u_np1[1:-1, 1:-1] = (
            2 * u_n[1:-1, 1:-1]
                - (1 - a[1:-1, 1:-1]) * u_nm1[1:-1, 1:-1]
                + dt**2 * (
                c[1:-1, 1:-1]**2 * lap[1:-1, 1:-1]
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
        "trajectoire_objet_indices": np.array(trajectoire_objet),
        "trajectoire_objet_m": trajectoire_objet_en_metres(params, trajectoire_objet) if len(trajectoire_objet) > 0 else None,
    }

L = 1000
nx = 350
ny = 350

params = {
    "L": L,
    "nx": nx,
    "ny": ny,
    "dt": 2e-4,
    "t_total": 0.3,
    "rho": 1000.0,
    "kappa": 2.2e9,
    "sigma": 1,
    "A0": 1000,
    "tau": 0.005,
    "gamma_eau": 0.3,
    "epaisseur_pml_ratio": 0.2,
    "puissance_pml": 0.5,
    "gamma_max_pml": 60,
    "capteurs": [
        {"nom": "Capteur 1", "position": (int(nx*0.5), int(ny*0.5)), "emissions": [0.02]},
        {"nom": "Capteur 2", "position": (int(nx*0.5), int(ny*0.4)), "emissions": []},
        {"nom": "Capteur 3", "position": (int(nx*0.6), int(ny*0.4)), "emissions": []},
    ],
    "objet": {
    "position": (220, 270),
    "rayon": 12,
    "rho": 7800.0,
    "kappa": 1.6e11,
    "gamma": 2.0,
    "vitesse_m_s": (0.0, 300.0),
},
}

resultats = run_simulation(params)
ani = animer_resultats_db(params, resultats)
tracer_trajectoire_objet(params, resultats)
plot_signaux(params, resultats)
# ani.save("objet_qui_bouge.mp4", writer="ffmpeg", fps=60)