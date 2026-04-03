import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random


def creer_pml(nx, ny, epaisseur, puissance, gamma_max):
    """Création du PML."""
    grille = np.zeros((nx, ny))
    lignes, cols = np.indices((nx, ny))
    dist = np.minimum.reduce([lignes, cols, nx - 1 - lignes, ny - 1 - cols])
    mask = dist < epaisseur
    s = (epaisseur - dist[mask]) / epaisseur
    grille[mask] = gamma_max * s**puissance
    return grille


def creer_pulses_aleatoires(nb_pulses, nx, ny, epaisseur_pml, t_depart=0.01, dt_pulse=0.02):
    """Crée une liste de pulses aléatoires pas dans le PML."""
    pulses = []
    marge = epaisseur_pml + 2

    for k in range(nb_pulses):
        x0 = random.randint(marge, nx - marge - 1)
        y0 = random.randint(marge, ny - marge - 1)
        t0 = t_depart + k * dt_pulse
        pulses.append((x0, y0, t0))

    return pulses


def source_sonar_multi(nx, ny, t, pulse_data, sigma=1, f=400, A0=5e7, tau=0.001):
    """Source totale en fonction du nombre et de la position des pulses."""
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    source = np.zeros((nx, ny))

    for x0, y0, t0 in pulse_data:
        r2 = (x - x0) ** 2 + (y - y0) ** 2
        enveloppe_spatiale = np.exp(-r2 / (2 * sigma**2))
        enveloppe_temporelle = np.exp(-((t - t0) ** 2) / (2 * tau**2))
        source += A0 * enveloppe_spatiale * enveloppe_temporelle * np.sin(2 * np.pi * f * (t - t0))

    return source


def waveform_source(temps, pulses, f_source=400, A0_source=5e7, tau_source=0.001):
    """Fonction pour l'affichage dans le temps du signal envoyé."""
    signal = np.zeros_like(temps)

    for _, _, t0 in pulses:
        enveloppe_temporelle = np.exp(-((temps - t0) ** 2) / (2 * tau_source**2))
        signal += A0_source * enveloppe_temporelle * np.sin(2 * np.pi * f_source * (temps - t0))

    return signal


def laplacien_9_points(u, h):
    """Calcul du Laplacien."""
    lap = np.zeros_like(u)
    lap[1:-1, 1:-1] = (
        4
        * (
            u[2:, 1:-1]
            + u[:-2, 1:-1]
            + u[1:-1, 2:]
            + u[1:-1, :-2]
        )
        + (
            u[2:, 2:]
            + u[2:, :-2]
            + u[:-2, 2:]
            + u[:-2, :-2]
        )
        - 20 * u[1:-1, 1:-1]
    ) / (6 * h**2)
    return lap


def conditions_frontieres(u):
    """Conditions frontières de Dirichlet."""
    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0
    return u


def afficher_pml(gamma, L):
    """Affichage du PML"""
    plt.figure(figsize=(6, 5))
    plt.imshow(gamma.T, origin="lower", cmap="inferno", extent=[0, L, 0, L])
    plt.colorbar(label=r"$\gamma(x,y)$")
    plt.title("Couche absorbante")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.tight_layout()
    plt.show()


def afficher_waveform_source(t_total, dt, pulses, f_source=400, A0_source=5e7, tau_source=0.001):
    """Affichage de la source, (amplitude dans le temps de tous les pulses)."""
    temps = np.arange(0, t_total, dt)
    signal = waveform_source(
        temps,
        pulses,
        f_source=f_source,
        A0_source=A0_source,
        tau_source=tau_source
    )

    plt.figure(figsize=(8, 4))
    plt.plot(temps, signal)
    plt.title("Waveform envoyé")
    plt.xlabel("Temps [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def animer_resultats(frames, L, interval=30):
    """Animation du champ de pression."""
    fig, ax = plt.subplots(figsize=(7, 6))

    img = ax.imshow(
        frames[0].T,
        origin="lower",
        cmap="seismic",
        extent=[0, L, 0, L]
    )
    plt.colorbar(img, ax=ax, label="Amplitude")
    ax.set_title("Champ de pression")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    def maj(k):
        img.set_array(frames[k].T)
        return [img]

    ani = FuncAnimation(fig, maj, frames=len(frames), interval=interval, blit=False)
    plt.tight_layout()
    plt.show()
    return ani


def run_simul(
    L=100,
    t_total=0.2,
    nx=500,
    ny=500,
    rho=1000,
    kappa=2.2e9,
    cfl=0.95,
    epaisseur_pml_ratio=0.3,
    puissance_pml=3,
    gamma_max=20000,
    nb_pulses=5,
    t_depart_pulses=0.01,
    dt_pulse=0.03,
    sigma_source=1,
    f_source=400,
    A0_source=5e7,
    tau_source=0.001,
    nb_frames=300,
    afficher_pml_flag=True,
    afficher_waveform_flag=True,
    afficher_animation=True,
    seed=None,
):
    """Lance une simul, peu être aléatoire."""
    if seed is not None:
        random.seed(seed)

    h = L / nx
    c = np.sqrt(kappa / rho)

    dt_max = h / (c * np.sqrt(2))
    dt = cfl * dt_max
    nt = int(t_total / dt)
    pas_sauvegarde = max(1, nt // nb_frames)

    print(f"vitesse de l'onde = {c:.2f} m/s")
    print(f"distance entre les points = {h:.4f} m")
    print(f"Pas de temps = {dt:.2e} s")
    print(f"nt = {nt}")

    epaisseur_pml = int(epaisseur_pml_ratio * nx)
    gamma = creer_pml(nx, ny, epaisseur_pml, puissance_pml, gamma_max)
    
    if afficher_pml_flag:
        afficher_pml(gamma, L)

    pulses = creer_pulses_aleatoires(
        nb_pulses=nb_pulses,
        nx=nx,
        ny=ny,
        epaisseur_pml=epaisseur_pml,
        t_depart=t_depart_pulses,
        dt_pulse=dt_pulse,
    )

    print("Pulses :", pulses)

    if afficher_waveform_flag:
        afficher_waveform_source(
            t_total=t_total,
            dt=dt,
            pulses=pulses,
            f_source=f_source,
            A0_source=A0_source,
            tau_source=tau_source,
        )

    u_nm1 = np.zeros((nx, ny))
    u_n = np.zeros((nx, ny))
    u_np1 = np.zeros((nx, ny))

    frames = [u_n.copy()]
    energie_max = 0.0

    for n in range(nt):
        t_n = n * dt

        lap = laplacien_9_points(u_n, h)
        a = gamma * dt / 2

        source = source_sonar_multi(
            nx, ny, t_n, pulses,
            sigma=sigma_source,
            f=f_source,
            A0=A0_source,
            tau=tau_source
        )

        u_np1[1:-1, 1:-1] = (
            2 * u_n[1:-1, 1:-1]
            - (1 - a[1:-1, 1:-1]) * u_nm1[1:-1, 1:-1]
            + c**2 * dt**2 * lap[1:-1, 1:-1]
            + dt**2 * source[1:-1, 1:-1]
        ) / (1 + a[1:-1, 1:-1])

        u_np1 = conditions_frontieres(u_np1)

        energie_courante = np.sum(u_np1**2)
        energie_max = max(energie_max, energie_courante)

        if n % pas_sauvegarde == 0:
            frames.append(u_np1.copy())

        u_nm1[:, :] = u_n
        u_n[:, :] = u_np1

    energie_finale = np.sum(u_n**2)

    if energie_max > 0:
        pourcentage_residuel = 100 * energie_finale / energie_max
        print(f"Énergie résiduelle dans le domaine : {pourcentage_residuel:.6f} %")
    else:
        pourcentage_residuel = None
        print("Impossible de calculer le pourcentage résiduel : énergie maximale nulle.")

    if afficher_animation:
        animer_resultats(frames, L)

    return {
        "u_final": u_n.copy(),
        "frames": frames,
        "gamma": gamma,
        "pulses": pulses,
        "c": c,
        "h": h,
        "dt": dt,
        "nt": nt,
        "energie_max": energie_max,
        "energie_finale": energie_finale,
        "pourcentage_residuel": pourcentage_residuel,
    }

# Paramètres 

L = 100
t_total = 0.08
nx = 300
ny = 300

rho = 1000
kappa = 2.2e9
cfl = 0.95

epaisseur_pml_ratio = 0.2
puissance_pml = 2
gamma_max = 20000

nb_pulses = 1
t_depart_pulses = 0.01
dt_pulse = 0.03

sigma_source = 1
f_source = 400
A0_source = 5e6
tau_source = 0.003

nb_frames = 100
seed = None

# Simul

resultats = run_simul(
    L=L,
    t_total=t_total,
    nx=nx,
    ny=ny,
    rho=rho,
    kappa=kappa,
    cfl=cfl,
    epaisseur_pml_ratio=epaisseur_pml_ratio,
    puissance_pml=puissance_pml,
    gamma_max=gamma_max,
    nb_pulses=nb_pulses,
    t_depart_pulses=t_depart_pulses,
    dt_pulse=dt_pulse,
    sigma_source=sigma_source,
    f_source=f_source,
    A0_source=A0_source,
    tau_source=tau_source,
    nb_frames=nb_frames,
    afficher_pml_flag=True,
    afficher_waveform_flag=True,
    afficher_animation=True,
    seed=seed,
)