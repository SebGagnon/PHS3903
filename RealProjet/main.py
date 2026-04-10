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


def laplacien_9_points(u, h):
    """Calcul du Laplacien à 9 points."""
    lap = np.zeros_like(u)
    lap[1:-1, 1:-1] = (
        4 * (
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


def couleurs_capteurs(nb_capteurs):
    """Retourne une liste de couleurs bien distinctes."""
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(nb_capteurs)]


def afficher_pml(gamma, L):
    """Affichage du PML."""
    plt.figure(figsize=(6, 5))
    plt.imshow(gamma.T, origin="lower", cmap="inferno", extent=[0, L, 0, L])
    plt.colorbar(label=r"$\gamma(x,y)$")
    plt.title("Couche absorbante")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.tight_layout()
    plt.show()


def creer_capteurs_aleatoires(nb_capteurs, nx, ny, epaisseur_pml):
    """
    Crée des capteurs aléatoires dans le domaine physique.
    Chaque capteur est un dict :
    {
        'nom': 'Capteur 1',
        'position': (x, y),
        'emissions': [t1, t2, ...]
    }
    """
    capteurs = []
    positions_utilisees = set()
    marge = epaisseur_pml + 2

    while len(capteurs) < nb_capteurs:
        x0 = random.randint(marge, nx - marge - 1)
        y0 = random.randint(marge, ny - marge - 1)

        if (x0, y0) not in positions_utilisees:
            positions_utilisees.add((x0, y0))
            capteurs.append({
                "nom": f"Capteur {len(capteurs) + 1}",
                "position": (x0, y0),
                "emissions": []
            })

    return capteurs


def ajouter_emissions_par_defaut(capteurs, t_depart=0.01, dt_pulse=0.03):
    """
    Si on veut un comportement simple par défaut :
    un pulse par capteur, décalé dans le temps.
    """
    for i, capteur in enumerate(capteurs):
        capteur["emissions"] = [t_depart + i * dt_pulse]


def verifier_capteurs(capteurs):
    """Validation minimale de la structure des capteurs."""
    if not isinstance(capteurs, list):
        raise ValueError("capteurs doit être une liste.")

    for i, capteur in enumerate(capteurs):
        if not isinstance(capteur, dict):
            raise ValueError(f"Le capteur d'indice {i} doit être un dictionnaire.")

        if "position" not in capteur:
            raise ValueError(f"Le capteur d'indice {i} doit contenir une clé 'position'.")

        if "emissions" not in capteur:
            capteur["emissions"] = []

        if "nom" not in capteur:
            capteur["nom"] = f"Capteur {i+1}"

        position = capteur["position"]
        if not (isinstance(position, tuple) and len(position) == 2):
            raise ValueError(f"La position du capteur {i} doit être un tuple (x, y).")

        if not isinstance(capteur["emissions"], list):
            raise ValueError(f"'emissions' du capteur {i} doit être une liste.")


def construire_pulses_depuis_capteurs(capteurs):
    """
    Transforme les capteurs en liste aplatie de pulses :
    [(x0, y0, t0), ...]
    """
    pulses = []
    for capteur in capteurs:
        x0, y0 = capteur["position"]
        for t0 in capteur["emissions"]:
            pulses.append((x0, y0, t0))
    return pulses


def source_sonar_multi(nx, ny, t, pulse_data, sigma=1, f=400, A0=5e7, tau=0.001):
    """Source totale en fonction des pulses."""
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    source = np.zeros((nx, ny))

    for x0, y0, t0 in pulse_data:
        r2 = (x - x0) ** 2 + (y - y0) ** 2
        enveloppe_spatiale = np.exp(-r2 / (2 * sigma**2))
        enveloppe_temporelle = np.exp(-((t - t0) ** 2) / (2 * tau**2))
        source += A0 * enveloppe_spatiale * enveloppe_temporelle * np.sin(2 * np.pi * f * (t - t0))

    return source


def waveform_source_par_capteur(temps, capteurs, f_source=400, A0_source=5e7, tau_source=0.001):
    """
    Retourne un dict de signaux émis par capteur.
    Chaque capteur peut avoir plusieurs pulses, ou aucun.
    """
    signaux = {}

    for capteur in capteurs:
        signal = np.zeros_like(temps)
        for t0 in capteur["emissions"]:
            enveloppe_temporelle = np.exp(-((temps - t0) ** 2) / (2 * tau_source**2))
            signal += A0_source * enveloppe_temporelle * np.sin(2 * np.pi * f_source * (temps - t0))
        signaux[capteur["nom"]] = signal

    return signaux


def afficher_waveform_source(t_total, dt, capteurs, f_source=400, A0_source=5e7, tau_source=0.001):
    """Affiche les signaux émis par les capteurs."""
    temps = np.arange(0, t_total, dt)
    signaux = waveform_source_par_capteur(
        temps,
        capteurs,
        f_source=f_source,
        A0_source=A0_source,
        tau_source=tau_source
    )

    plt.figure(figsize=(10, 5))
    couleurs = couleurs_capteurs(len(capteurs))

    for capteur, couleur in zip(capteurs, couleurs):
        nom = capteur["nom"]
        signal = signaux[nom]

        if len(capteur["emissions"]) > 0:
            plt.plot(temps, signal, color=couleur, label=nom)
        else:
            plt.plot([], [], color=couleur, label=f"{nom} (silencieux)")

    plt.title("Waveforms envoyés par capteur")
    plt.xlabel("Temps [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def animer_resultats(frames, capteurs, L, interval=30):
    """Animation du champ de pression avec capteurs colorés et numérotés."""
    fig, ax = plt.subplots(figsize=(7, 6))

    nx, ny = frames[0].shape
    hx = L / nx
    hy = L / ny

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

    couleurs = couleurs_capteurs(len(capteurs))

    for i, (capteur, couleur) in enumerate(zip(capteurs, couleurs)):
        xc, yc = capteur["position"]
        x_m = (xc + 0.5) * hx
        y_m = (yc + 0.5) * hy

        ax.scatter(x_m, y_m, color=couleur, s=50, marker="o")
        ax.text(
            x_m, y_m, str(i + 1),
            color=couleur, fontsize=10,
            ha="left", va="bottom", weight="bold"
        )

    def maj(k):
        img.set_array(frames[k].T)
        return [img]

    ani = FuncAnimation(fig, maj, frames=len(frames), interval=interval, blit=False)
    plt.tight_layout()
    plt.show()
    return ani


def afficher_signaux_capteurs(temps, signaux_capteurs, capteurs):
    """Affiche les signaux mesurés à chaque capteur."""
    plt.figure(figsize=(10, 5))
    couleurs = couleurs_capteurs(len(capteurs))

    for capteur, signal, couleur in zip(capteurs, signaux_capteurs, couleurs):
        plt.plot(temps, signal, color=couleur, label=capteur["nom"])

    plt.title("Pression mesurée aux capteurs")
    plt.xlabel("Temps [s]")
    plt.ylabel("Pression")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


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
    nb_capteurs=5,
    capteurs=None,
    ajouter_emission_defaut_si_vide=True,
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
    afficher_capteurs_flag=True,
    seed=None,
):
    """Lance une simulation acoustique 2D avec capteurs sous forme de dictionnaires."""
    if seed is not None:
        random.seed(seed)

    h = L / nx
    c = np.sqrt(kappa / rho)

    dt_max = h / (c * np.sqrt(2))
    dt = cfl * dt_max
    nt = int(t_total / dt)
    pas_sauvegarde = max(1, nt // nb_frames)

    epaisseur_pml = int(epaisseur_pml_ratio * nx)
    gamma = creer_pml(nx, ny, epaisseur_pml, puissance_pml, gamma_max)

    if afficher_pml_flag:
        afficher_pml(gamma, L)

    if capteurs is None:
        capteurs = creer_capteurs_aleatoires(
            nb_capteurs=nb_capteurs,
            nx=nx,
            ny=ny,
            epaisseur_pml=epaisseur_pml
        )

        if ajouter_emission_defaut_si_vide:
            ajouter_emissions_par_defaut(
                capteurs,
                t_depart=t_depart_pulses,
                dt_pulse=dt_pulse
            )

    verifier_capteurs(capteurs)
    pulses = construire_pulses_depuis_capteurs(capteurs)

    if afficher_waveform_flag:
        afficher_waveform_source(
            t_total=t_total,
            dt=dt,
            capteurs=capteurs,
            f_source=f_source,
            A0_source=A0_source,
            tau_source=tau_source,
        )

    u_nm1 = np.zeros((nx, ny))
    u_n = np.zeros((nx, ny))
    u_np1 = np.zeros((nx, ny))

    frames = [u_n.copy()]
    energie_max = 0.0

    temps = np.arange(nt) * dt
    signaux_capteurs = [np.zeros(nt) for _ in capteurs]

    for n in range(nt):
        t_n = n * dt

        lap = laplacien_9_points(u_n, h)
        a = gamma * dt / 2

        source = source_sonar_multi(
            nx=nx,
            ny=ny,
            t=t_n,
            pulse_data=pulses,
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

        for i, capteur in enumerate(capteurs):
            xc, yc = capteur["position"]
            signaux_capteurs[i][n] = u_np1[xc, yc]

        energie_courante = np.sum(u_np1**2)
        energie_max = max(energie_max, energie_courante)

        if n % pas_sauvegarde == 0:
            frames.append(u_np1.copy())

        u_nm1[:, :] = u_n
        u_n[:, :] = u_np1

    energie_finale = np.sum(u_n**2)

    if energie_max > 0:
        pourcentage_residuel = 100 * energie_finale / energie_max
    else:
        pourcentage_residuel = None

    if afficher_capteurs_flag:
        afficher_signaux_capteurs(temps, signaux_capteurs, capteurs)

    if afficher_animation:
        animer_resultats(frames, capteurs, L)

    return {
        "u_final": u_n.copy(),
        "frames": frames,
        "gamma": gamma,
        "capteurs": capteurs,
        "pulses": pulses,
        "temps": temps,
        "signaux_capteurs": signaux_capteurs,
        "c": c,
        "h": h,
        "dt": dt,
        "nt": nt,
        "energie_max": energie_max,
        "energie_finale": energie_finale,
        "pourcentage_residuel": pourcentage_residuel,
    }


# =========================
# EXEMPLE D'UTILISATION
# =========================

L = 50
t_total = 0.05
nx = 200
ny = 200

rho = 1000
kappa = 2.2e9
cfl = 1.0

epaisseur_pml_ratio = 0.2
puissance_pml = 2
gamma_max = 3000

sigma_source = 1
f_source = 1111
A0_source = 8e6
tau_source = 0.002

nb_frames = 100
seed = 42

capteurs = [
    {
        "nom": "Capteur 1",
        "position": (100, 100),
        "emissions": [0.01]
    },
    {
        "nom": "Capteur 2",
        "position": (100, 150),
        "emissions": []
    },
    {
        "nom": "Capteur 3",
        "position": (100,180),
        "emissions": []
    },
]

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
    capteurs=capteurs,
    sigma_source=sigma_source,
    f_source=f_source,
    A0_source=A0_source,
    tau_source=tau_source,
    nb_frames=nb_frames,
    afficher_pml_flag=True,
    afficher_waveform_flag=True,
    afficher_animation=True,
    afficher_capteurs_flag=True,
    seed=seed,
)
