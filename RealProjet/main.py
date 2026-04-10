import matplotlib
matplotlib.use("TkAgg")   # enlève cette ligne si ça marche déjà chez toi

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
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
    """Couleurs distinctes pour les capteurs."""
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(nb_capteurs)]


def afficher_pml(gamma, L):
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
        'position': (ix, iy),
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
    """Un pulse par capteur, décalé dans le temps."""
    for i, capteur in enumerate(capteurs):
        capteur["emissions"] = [t_depart + i * dt_pulse]


def verifier_capteurs(capteurs):
    """Validation minimale des capteurs."""
    if not isinstance(capteurs, list):
        raise ValueError("capteurs doit être une liste.")

    for i, capteur in enumerate(capteurs):
        if not isinstance(capteur, dict):
            raise ValueError(f"Le capteur {i} doit être un dictionnaire.")

        if "nom" not in capteur:
            capteur["nom"] = f"Capteur {i+1}"

        if "position" not in capteur:
            raise ValueError(f"Le capteur {i} doit avoir une clé 'position'.")

        if "emissions" not in capteur:
            capteur["emissions"] = []

        pos = capteur["position"]
        if not (isinstance(pos, tuple) and len(pos) == 2):
            raise ValueError(f"La position du capteur {i} doit être un tuple (ix, iy).")

        if not isinstance(capteur["emissions"], list):
            raise ValueError(f"'emissions' du capteur {i} doit être une liste.")


def construire_pulses_depuis_capteurs(capteurs):
    """Transforme les capteurs en liste de pulses (ix, iy, t0)."""
    pulses = []
    for capteur in capteurs:
        ix, iy = capteur["position"]
        for t0 in capteur["emissions"]:
            pulses.append((ix, iy, t0))
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
    """Retourne le signal émis par chaque capteur."""
    signaux = {}

    for capteur in capteurs:
        signal = np.zeros_like(temps)
        for t0 in capteur["emissions"]:
            enveloppe_temporelle = np.exp(-((temps - t0) ** 2) / (2 * tau_source**2))
            signal += A0_source * enveloppe_temporelle * np.sin(2 * np.pi * f_source * (temps - t0))
        signaux[capteur["nom"]] = signal

    return signaux


def afficher_waveform_source(t_total, dt, capteurs, f_source=400, A0_source=5e7, tau_source=0.001):
    """Affiche les waveforms envoyés par les capteurs."""
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


def creer_maillage_physique(nx, ny, h):
    """Maillage en mètres au centre des cellules."""
    x = (np.arange(nx) + 0.5) * h
    y = (np.arange(ny) + 0.5) * h
    return np.meshgrid(x, y, indexing="ij")


def verifier_objet(objet):
    """Validation minimale de l'objet mobile."""
    if objet is None:
        return

    if "forme" not in objet:
        raise ValueError("L'objet doit contenir 'forme'.")

    if "centre_init_m" not in objet:
        raise ValueError("L'objet doit contenir 'centre_init_m'.")

    if "vitesse_m_s" not in objet:
        objet["vitesse_m_s"] = (0.0, 0.0)

    if "c_objet" not in objet:
        raise ValueError("L'objet doit contenir 'c_objet'.")

    if "gamma_objet" not in objet:
        objet["gamma_objet"] = 0.0

    if objet["forme"] == "cercle":
        if "rayon_m" not in objet:
            raise ValueError("Un objet cercle doit contenir 'rayon_m'.")
    elif objet["forme"] == "rectangle":
        if "taille_m" not in objet:
            raise ValueError("Un objet rectangle doit contenir 'taille_m'.")
    else:
        raise ValueError("forme doit être 'cercle' ou 'rectangle'.")

    if "actif" not in objet:
        objet["actif"] = True


def position_objet(objet, t):
    """Position du centre de l'objet à l'instant t."""
    x0, y0 = objet["centre_init_m"]
    vx, vy = objet.get("vitesse_m_s", (0.0, 0.0))
    return x0 + vx * t, y0 + vy * t


def construire_cgrid_et_gamma_objet(X, Y, c_eau, t, objet=None):
    """
    Construit :
    - c_grid(x,y,t)
    - gamma_objet_grid(x,y,t)
    - masque_objet
    - centre de l'objet
    """
    c_grid = c_eau * np.ones_like(X)
    gamma_objet_grid = np.zeros_like(X)
    masque_objet = np.zeros_like(X, dtype=bool)
    centre = None

    if objet is None or not objet.get("actif", True):
        return c_grid, gamma_objet_grid, masque_objet, centre

    xc, yc = position_objet(objet, t)
    centre = (xc, yc)

    if objet["forme"] == "cercle":
        r = objet["rayon_m"]
        masque_objet = (X - xc) ** 2 + (Y - yc) ** 2 <= r ** 2

    elif objet["forme"] == "rectangle":
        lx, ly = objet["taille_m"]
        masque_objet = (
            (np.abs(X - xc) <= lx / 2)
            & (np.abs(Y - yc) <= ly / 2)
        )

    c_grid[masque_objet] = objet["c_objet"]
    gamma_objet_grid[masque_objet] = objet["gamma_objet"]

    return c_grid, gamma_objet_grid, masque_objet, centre


def dessiner_objet_initial(ax, objet):
    """Ajoute un patch matplotlib pour l'objet."""
    if objet is None or not objet.get("actif", True):
        return None

    xc, yc = objet["centre_init_m"]

    if objet["forme"] == "cercle":
        patch = Circle(
            (xc, yc),
            objet["rayon_m"],
            fill=False,
            edgecolor="black",
            linewidth=2
        )
    else:
        lx, ly = objet["taille_m"]
        patch = Rectangle(
            (xc - lx / 2, yc - ly / 2),
            lx,
            ly,
            fill=False,
            edgecolor="black",
            linewidth=2
        )

    ax.add_patch(patch)
    return patch


def animer_resultats(
    frames,
    capteurs,
    L,
    signaux_capteurs,
    objet=None,
    trajectoire_objet=None,
    interval=30
):
    """Animation du champ de pression avec échelle fixée au max mesuré par les capteurs."""
    fig, ax = plt.subplots(figsize=(7, 6))

    nx, ny = frames[0].shape
    hx = L / nx
    hy = L / ny

    # maximum absolu parmi tous les signaux des capteurs
    if len(signaux_capteurs) > 0:
        amp_max = 0.2*max(np.max(np.abs(signal)) for signal in signaux_capteurs)
    else:
        amp_max = 0.0

    # sécurité si tout est nul
    if amp_max == 0:
        amp_max = 1.0

    img = ax.imshow(
        frames[0].T,
        origin="lower",
        cmap="seismic",
        extent=[0, L, 0, L],
        vmin=-amp_max,
        vmax=amp_max
    )

    plt.colorbar(img, ax=ax, label="Amplitude")
    ax.set_title("Champ de pression")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    couleurs = couleurs_capteurs(len(capteurs))

    for i, (capteur, couleur) in enumerate(zip(capteurs, couleurs)):
        ix, iy = capteur["position"]
        x_m = (ix + 0.5) * hx
        y_m = (iy + 0.5) * hy

        ax.scatter(x_m, y_m, color=couleur, s=50, marker="o")
        ax.text(
            x_m, y_m, str(i + 1),
            color=couleur, fontsize=10,
            ha="left", va="bottom", weight="bold"
        )

    patch_objet = dessiner_objet_initial(ax, objet)

    def maj(k):
        artistes = [img]
        img.set_array(frames[k].T)

        if patch_objet is not None and trajectoire_objet is not None and k < len(trajectoire_objet):
            centre = trajectoire_objet[k]
            if centre is not None:
                xc, yc = centre
                if objet["forme"] == "cercle":
                    patch_objet.center = (xc, yc)
                else:
                    lx, ly = objet["taille_m"]
                    patch_objet.set_xy((xc - lx / 2, yc - ly / 2))
                artistes.append(patch_objet)

        return artistes

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
    cfl=0.6,
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
    objet=None,
    nb_frames=300,
    afficher_pml_flag=False,
    afficher_waveform_flag=True,
    afficher_animation=True,
    afficher_capteurs_flag=True,
    seed=None,
):
    """Simulation acoustique 2D avec objet mobile via c_grid et gamma_objet."""
    if seed is not None:
        random.seed(seed)

    verifier_objet(objet)

    h = L / nx
    c_eau = np.sqrt(kappa / rho)

    if objet is None:
        c_max = c_eau
    else:
        c_max = max(c_eau, objet["c_objet"])

    dt_max = h / (c_max * np.sqrt(2))
    dt = cfl * dt_max
    nt = int(t_total / dt)
    pas_sauvegarde = max(1, nt // nb_frames)

    epaisseur_pml = int(epaisseur_pml_ratio * nx)
    gamma_pml = creer_pml(nx, ny, epaisseur_pml, puissance_pml, gamma_max)

    if afficher_pml_flag:
        afficher_pml(gamma_pml, L)

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

    X, Y = creer_maillage_physique(nx, ny, h)

    u_nm1 = np.zeros((nx, ny))
    u_n = np.zeros((nx, ny))
    u_np1 = np.zeros((nx, ny))

    frames = [u_n.copy()]
    trajectoire_objet = []

    _, _, _, centre_init = construire_cgrid_et_gamma_objet(X, Y, c_eau, 0.0, objet)
    trajectoire_objet.append(centre_init)

    energie_max = 0.0
    temps = np.arange(nt) * dt
    signaux_capteurs = [np.zeros(nt) for _ in capteurs]

    for n in range(nt):
        t_n = n * dt

        c_grid, gamma_objet_grid, masque_objet, centre_objet = construire_cgrid_et_gamma_objet(
            X, Y, c_eau, t_n, objet
        )

        gamma_total = gamma_pml + gamma_objet_grid

        lap = laplacien_9_points(u_n, h)
        a = gamma_total * dt / 2

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
            + (dt ** 2) * (c_grid[1:-1, 1:-1] ** 2) * lap[1:-1, 1:-1]
            + dt ** 2 * source[1:-1, 1:-1]
        ) / (1 + a[1:-1, 1:-1])

        u_np1 = conditions_frontieres(u_np1)

        for i, capteur in enumerate(capteurs):
            ix, iy = capteur["position"]
            signaux_capteurs[i][n] = u_np1[ix, iy]

        energie_courante = np.sum(u_np1 ** 2)
        energie_max = max(energie_max, energie_courante)

        if n % pas_sauvegarde == 0:
            frames.append(u_np1.copy())
            trajectoire_objet.append(centre_objet)

        u_nm1[:, :] = u_n
        u_n[:, :] = u_np1

    energie_finale = np.sum(u_n ** 2)

    if energie_max > 0:
        pourcentage_residuel = 100 * energie_finale / energie_max
    else:
        pourcentage_residuel = None

    if afficher_capteurs_flag:
        afficher_signaux_capteurs(temps, signaux_capteurs, capteurs)

    ani = None
    if afficher_animation:
        ani = animer_resultats(
        frames,
        capteurs,
        L,
        signaux_capteurs=signaux_capteurs,
        objet=objet,
        trajectoire_objet=trajectoire_objet
        )

    return {
        "u_final": u_n.copy(),
        "frames": frames,
        "gamma_pml": gamma_pml,
        "capteurs": capteurs,
        "pulses": pulses,
        "temps": temps,
        "signaux_capteurs": signaux_capteurs,
        "c_eau": c_eau,
        "dt": dt,
        "nt": nt,
        "h": h,
        "energie_max": energie_max,
        "energie_finale": energie_finale,
        "pourcentage_residuel": pourcentage_residuel,
        "objet": objet,
        "trajectoire_objet": trajectoire_objet,
        "animation": ani,
    }


# =========================================================
# EXEMPLE D'UTILISATION
# =========================================================

L = 500
t_total = 0.5
nx = 200
ny = 200

rho = 1000
kappa = 2.2e9
cfl = 0.05

epaisseur_pml_ratio = 0.2
puissance_pml = 2
gamma_max = 3000

sigma_source = 1
f_source = 0.5
A0_source = 8e10
tau_source = 0.001
nb_frames = 100
seed = 42

# capteurs aléatoires
capteurs = [
    {
        "nom": "Capteur 1",
        "position": (110, 110),
        "emissions": [0.010]
    },
    {
        "nom": "Capteur 2",
        "position": (90, 90),
        "emissions": []
    },
    {
        "nom": "Capteur 3",
        "position": (75, 75),
        "emissions": []
    },
   
]


# objet
objet = {
    "forme": "cercle",              # "cercle" ou "rectangle"
    "centre_init_m": (600.0, 500.0),  # position initiale en mètres
    "vitesse_m_s": (300.0, -300.0),     # vitesse (vx, vy) en m/s
     "rayon_m": 10.0,                 # pour un cercle
    # "taille_m": (10.0, 2.0),       # pour un rectangle
    "c_objet": 5200.0,              # vitesse du son dans l'objet
    "gamma_objet": 4000.0,          # amortissement dans l'objet
    "actif": True,
}

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
    # nb_capteurs=nb_capteurs,
    capteurs=capteurs,
    ajouter_emission_defaut_si_vide=True,
    t_depart_pulses=0.01,
    dt_pulse=0.01,
    sigma_source=sigma_source,
    f_source=f_source,
    A0_source=A0_source,
    tau_source=tau_source,
    objet=objet,
    nb_frames=nb_frames,
    afficher_pml_flag=False,
    afficher_waveform_flag=True,
    afficher_animation=True,
    afficher_capteurs_flag=True,
    seed=seed,
)

print("Simulation terminée")
print("dt =", resultats["dt"])
print("nt =", resultats["nt"])
print("Nombre de pulses =", len(resultats["pulses"]))
print("Énergie résiduelle [%] =", resultats["pourcentage_residuel"])