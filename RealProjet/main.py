import matplotlib
matplotlib.use("TkAgg")   # enlève cette ligne si ça marche déjà chez toi

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
import random
from scipy.optimize import least_squares
from scipy.signal import hilbert
import time


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


def methode_multilateration():


    objet_loin = {
    "forme": "cercle",              # "cercle" ou "rectangle"
    "centre_init_m": (0, 0),  # position initiale en mètres
    "vitesse_m_s": (0, 0),     # vitesse (vx, vy) en m/s
     "rayon_m": 0.0,                 # pour un cercle
    # "taille_m": (10.0, 2.0),       # pour un rectangle
    "c_objet": 5200.0,              # vitesse du son dans l'objet
    "gamma_objet": 4000.0,          # amortissement dans l'objet
    "actif": True,
}
    
    resultats_sans_bateau = run_simul(
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
    objet=objet_loin,
    nb_frames=nb_frames,
    afficher_pml_flag=False,
    afficher_waveform_flag=False,
    afficher_animation=False,
    afficher_capteurs_flag=False,
    seed=seed,
)
    resultats_avec_bateau = run_simul(
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
    afficher_waveform_flag=False,
    afficher_animation=True,
    afficher_capteurs_flag=True,
    seed=seed,
)
    
    # --- Extraction au format de ton ancien code ---
    t_hist                = resultats_avec_bateau["temps"]
    signaux_avec_bateau   = resultats_avec_bateau["signaux_capteurs"]   # liste de N arrays
    signaux_sans_bateau   = resultats_sans_bateau["signaux_capteurs"]
    c_eau                 = resultats_sans_bateau["c_eau"]

    # intensité_echo[i] = signal de l'écho du capteur i
    intensite_echo = np.array([
        signaux_avec_bateau[i] - signaux_sans_bateau[i]
        for i in range(len(signaux_avec_bateau))
    ])  # shape [N_capteurs, N_temps]
    
    show_echo(intensite_echo, t_hist, capteurs, t_total)

    plages = get_plages_pulses(capteurs, t_total)

    positions_estimees       = []
    distances_par_pulse      = []
    t0_par_pulse             = []
    positions_reelles_impact = []
    t_impacts                = []

    for k, (idx_emit, t_debut_plage, t_fin_plage) in enumerate(plages):

        print(f"\n--- Pulse {k+1} | Émetteur C{idx_emit+1} | [{t_debut_plage:.3f}s, {t_fin_plage:.3f}s] ---")

        try:
            pos_m, temps_echo, distances, idx_emit_retour = multilateration_un_pulse(
                intensite_echo, t_hist, capteurs, c_eau,
                idx_emetteur  = idx_emit,
                t_debut_plage = t_debut_plage,
                t_fin_plage   = t_fin_plage
            )

            position_impact_m, t_impact = calc_position_reelle_impact(
                objet      = objet,
                t0_emit    = t_debut_plage,
                d_emetteur = distances[idx_emit_retour],
                c_eau      = c_eau
            )

            # Accumulation des résultats
            positions_estimees.append(pos_m)
            distances_par_pulse.append(distances)
            t0_par_pulse.append(t_debut_plage)
            positions_reelles_impact.append(position_impact_m)
            t_impacts.append(t_impact)

            erreur = np.linalg.norm(pos_m - position_impact_m)
            print(f"  Position réelle à l'impact : {position_impact_m} m  (t={t_impact:.3f}s)")
            print(f"  Position estimée           : {pos_m} m")
            print(f"  Erreur                     : {erreur:.2f} m")

            plot_localisation_echo(
                capteurs           = capteurs,
                position_estimee_m = pos_m,
                objet              = objet,
                distances_m        = distances,
                t0_emit            = t_debut_plage,
                d_emetteur         = distances[idx_emit_retour],
                c_eau              = c_eau,
                L                  = L
            )

        except ValueError as e:
            print(f"  ⚠ Ignoré : {e}")

    # --- Graphiques finaux ---
    if len(positions_estimees) >= 1:
        plot_positions_estimees(
            capteurs                 = capteurs,
            positions_estimees_m     = positions_estimees,
            positions_reelles_impact = positions_reelles_impact,
            t_impacts                = t_impacts,
            t0_par_pulse             = t0_par_pulse,
            objet                    = objet,
            L                        = L
        )

    if len(positions_estimees) >= 2:
        vitesse_estimee, pos_initiale = estimer_vitesse(positions_estimees, t_impacts, objet)

def get_plages_pulses(capteurs, t_total):
    """
    Extrait les plages temporelles depuis la structure capteurs.
    Compatible avec emissions sous forme de floats [t0, ...] 
    ou de dicts [{"t0": t0, ...}, ...]
    """

    tous_les_pulses = []
    for idx_capteur, capteur in enumerate(capteurs):
        for emission in capteur.get("emissions", []):

            # --- Compatibilité float ou dict ---
            if isinstance(emission, dict):
                t0 = emission["t0"]
            else:
                t0 = float(emission)

            tous_les_pulses.append((idx_capteur, t0))

    if len(tous_les_pulses) == 0:
        print("⚠ Aucun pulse détecté dans les capteurs.")
        return []

    # --- Trier par temps d'émission ---
    tous_les_pulses.sort(key=lambda x: x[1])

    # --- Construire les plages ---
    t0_list = [p[1] for p in tous_les_pulses]
    plages  = []

    for i, (idx_capteur, t0) in enumerate(tous_les_pulses):
        t_debut = t0
        t_fin   = t0_list[i + 1] if i + 1 < len(tous_les_pulses) else t_total
        plages.append((idx_capteur, t_debut, t_fin))

    return plages


def construire_pulse_data(capteurs):
    """
    Convertit la liste capteurs en liste (x0, y0, t0) pour source_sonar_multi.
    Compatible avec emissions sous forme de floats ou de dicts.
    """
    pulse_data = []
    for capteur in capteurs:
        x0, y0 = capteur["position"]
        for emission in capteur.get("emissions", []):
            if isinstance(emission, dict):
                t0 = emission["t0"]
            else:
                t0 = float(emission)
            pulse_data.append((x0, y0, t0))
    return pulse_data


def max_enveloppe_plage(signal, t_hist, t_debut_plage, t_fin_plage, seuil_montee=1):
    mask = (t_hist >= t_debut_plage) & (t_hist < t_fin_plage)

    if not np.any(mask):
        return None, None, np.abs(hilbert(signal))

    signal_plage    = signal[mask]
    t_hist_plage    = t_hist[mask]
    enveloppe_plage = np.abs(hilbert(signal_plage))

    idx_pic_local = np.argmax(enveloppe_plage)
    amp_pic       = enveloppe_plage[idx_pic_local]

    seuil_debut = seuil_montee * amp_pic
    idx_debut   = idx_pic_local
    while idx_debut > 0 and enveloppe_plage[idx_debut] > seuil_debut:
        idx_debut -= 1

    enveloppe_full = np.abs(hilbert(signal))
    return t_hist_plage[idx_debut], enveloppe_plage[idx_debut], enveloppe_full


def show_echo(intensite_echo, t_hist, capteurs, t_total, Echo = False):
    """
    Affiche les échos par plage temporelle (un sous-graphique par pulse).

    intensite_echo : array [N_capteurs, N_temps]
    capteurs       : liste de dicts avec champ "emissions" (nouveau format)
    t_total        : durée totale de la simulation [s]
    """
    if not Echo:
        return

    plages   = get_plages_pulses(capteurs, t_total)
    n_pulses = len(plages)
    n_capteurs = len(capteurs)

    # Couleurs dynamiques selon le nombre de capteurs
    cmap     = plt.cm.tab10
    couleurs = [cmap(i % 10) for i in range(n_capteurs)]
    labels   = [f"Capteur {i+1}" for i in range(n_capteurs)]

    fig, axes = plt.subplots(n_pulses, 1, figsize=(10, 4 * n_pulses), sharex=False)
    if n_pulses == 1:
        axes = [axes]

    for k, (idx_emit, t_debut, t_fin) in enumerate(plages):
        ax   = axes[k]
        mask = (t_hist >= t_debut) & (t_hist < t_fin)

        for i, signal in enumerate(intensite_echo):
            ax.plot(t_hist[mask], signal[mask], color=couleurs[i], alpha=0.4)

            t_max, amp_max, enveloppe_full = max_enveloppe_plage(
                signal, t_hist, t_debut, t_fin
            )
            ax.plot(t_hist[mask], enveloppe_full[mask],
                    color=couleurs[i], linestyle="--", label=labels[i])

            if t_max is not None:
                ax.scatter(t_max, amp_max, color=couleurs[i], s=80, zorder=5)
                ax.axvline(t_max, color=couleurs[i], linestyle=":", alpha=0.7)

        ax.axvline(t_debut, color="black", linestyle="-", linewidth=1.5,
                   label=f"Émission C{idx_emit+1} (t={t_debut:.2f}s)")
        ax.set_xlim(t_debut, t_fin)
        ax.set_xlabel("Temps [s]")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Pulse {k+1} — Émetteur C{idx_emit+1} | [{t_debut:.2f}s, {t_fin:.2f}s]")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.4)

    plt.suptitle("Échos par plage de pulse", fontsize=13)
    plt.tight_layout()
    plt.show()


def multilateration_un_pulse(intensite_echos, t_hist, capteurs, c_eau,
                              idx_emetteur=0, t_debut_plage=None, t_fin_plage=None):
    """
    intensite_echos : array [N_capteurs, N_temps]
    capteurs        : liste de dicts avec champ "position" en noeuds
    c_eau           : vitesse du son dans l'eau [m/s]
    idx_emetteur    : index du capteur émetteur
    """

    intensite_echos = np.array(intensite_echos)
    n_capteurs      = len(capteurs)
    h               = L / nx  # taille d'un noeud [m]

    # --- Positions des capteurs en mètres ---
    positions_m = np.array([
        [capteur["position"][0] * h, capteur["position"][1] * h]
        for capteur in capteurs
    ])

    # -------- DETECTION DES TEMPS D'ECHO --------
    temps_echo = []
    for i, signal in enumerate(intensite_echos):
        if t_debut_plage is not None and t_fin_plage is not None:
            t_max, amp_max, _ = max_enveloppe_plage(signal, t_hist, t_debut_plage, t_fin_plage)
        else:
            raise ValueError("t_debut_plage et t_fin_plage doivent être fournis.")

        if t_max is None:
            raise ValueError(f"Pas de pic détecté pour capteur {i+1}")
        temps_echo.append(t_max)

    temps_echo = np.array(temps_echo)
    t0_emit    = t_debut_plage if t_debut_plage is not None else 0.0

    # -------- DISTANCES en mètres --------
    # Émetteur : aller-retour
    d_emetteur = c_eau * (temps_echo[idx_emetteur] - t0_emit) / 2  # [m]

    # Récepteurs : trajet total - aller
    indices_recepteurs = [i for i in range(n_capteurs) if i != idx_emetteur]
    d_recepteurs = {
        i: c_eau * (temps_echo[i] - t0_emit) - d_emetteur
        for i in indices_recepteurs
    }

    # Assemblage distances en mètres
    distances               = np.zeros(n_capteurs)
    distances[idx_emetteur] = d_emetteur
    for i in indices_recepteurs:
        distances[i] = d_recepteurs[i]

    # -------- RÉSIDUS en mètres --------
    def residuals(x):
        res = [np.linalg.norm(x - positions_m[idx_emetteur]) - d_emetteur]
        for i in indices_recepteurs:
            res.append(np.linalg.norm(x - positions_m[i]) - d_recepteurs[i])
        return res

    x0  = np.mean(positions_m, axis=0)
    sol = least_squares(residuals, x0)

    # position estimée en mètres
    return sol.x, temps_echo, distances, idx_emetteur


def calc_position_reelle_impact(objet, t0_emit, d_emetteur, c_eau):
    """
    objet      : dict avec "centre_init_m" et "vitesse_m_s"
    t0_emit    : temps d'émission du pulse [s]
    d_emetteur : distance émetteur → objet [m]
    """
    vx, vy   = objet["vitesse_m_s"]
    x0, y0   = objet["centre_init_m"]

    t_aller  = d_emetteur / c_eau        # [s]
    t_impact = t0_emit + t_aller

    x_impact = x0 + vx * t_impact       # [m]
    y_impact = y0 + vy * t_impact       # [m]

    return np.array([x_impact, y_impact]), t_impact


def plot_localisation_echo(capteurs, position_estimee_m, objet, distances_m,
                           t0_emit, d_emetteur, c_eau, L, plot_loc_echo = True):
    """
    capteurs           : liste de dicts avec "position" en noeuds et "nom"
    position_estimee_m : position estimée [m]
    objet              : dict avec "centre_init_m", "vitesse_m_s", "rayon_m" ou "taille_m"
    distances_m        : array[N_capteurs] en mètres
    t0_emit            : temps d'émission [s]
    d_emetteur         : distance émetteur → objet [m]
    """
    h = L / nx
    # --- Position réelle à l'impact ---
    position_impact_m, t_impact = calc_position_reelle_impact(objet, t0_emit, d_emetteur, c_eau)

    if not plot_loc_echo:
        return position_impact_m, t_impact



    fig, ax   = plt.subplots(figsize=(8, 8))
    n_capteurs = len(capteurs)
    cmap       = plt.cm.tab10
    couleurs_c = [cmap(i % 10) for i in range(n_capteurs)]


    # --- Position initiale de l'objet ---
    x0_m, y0_m = objet["centre_init_m"]

    # --- Taille de l'objet ---
    if objet["forme"] == "cercle":
        rayon_m = objet.get("rayon_m", 5.0)
        cercle_reel_init = plt.Circle((x0_m, y0_m), rayon_m,
                                       color="grey", alpha=0.3,
                                       label="Position initiale (t=0)", zorder=3)
        ax.add_patch(cercle_reel_init)

        x_i, y_i = position_impact_m
        cercle_impact = plt.Circle((x_i, y_i), rayon_m,
                                    color="black", alpha=0.6,
                                    label=f"Position à l'impact (t={t_impact:.3f}s)", zorder=4)
        ax.add_patch(cercle_impact)

    elif objet["forme"] == "rectangle":
        lx_m, ly_m = objet.get("taille_m", (5.0, 5.0))

        ax.add_patch(plt.Rectangle(
            (x0_m - lx_m/2, y0_m - ly_m/2), lx_m, ly_m,
            color="grey", alpha=0.3, label="Position initiale (t=0)", zorder=3
        ))
        x_i, y_i = position_impact_m
        ax.add_patch(plt.Rectangle(
            (x_i - lx_m/2, y_i - ly_m/2), lx_m, ly_m,
            color="black", alpha=0.6,
            label=f"Position à l'impact (t={t_impact:.3f}s)", zorder=4
        ))

    # --- Flèche trajectoire ---
    ax.annotate("", xy=(x_i, y_i), xytext=(x0_m, y0_m),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # --- Position estimée ---
    x_e, y_e = position_estimee_m
    ax.scatter(x_e, y_e, marker="*", color="gold", s=250, zorder=5,
               edgecolors="black", linewidths=0.8, label="Position estimée (MC)")

    # --- Capteurs + cercles + lignes ---
    for i, capteur in enumerate(capteurs):
        ix, iy  = capteur["position"]
        x_c, y_c = ix * h, iy * h
        couleur  = couleurs_c[i]
        nom      = capteur.get("nom", f"C{i+1}")

        ax.scatter(x_c, y_c, color=couleur, s=100, zorder=4, marker="^")
        ax.text(x_c + 0.5, y_c + 0.5, nom, color=couleur, fontsize=9, fontweight="bold")

        ax.add_patch(plt.Circle(
            (x_c, y_c), radius=distances_m[i],
            color=couleur, fill=False, linestyle="--", linewidth=1.2, alpha=0.7,
            label=f"{nom} — d={distances_m[i]:.1f}m"
        ))

        ax.plot([x_c, x_e], [y_c, y_e],
                color=couleur, linestyle="-", linewidth=1.2, alpha=0.6)

        mid_x = (x_c + x_e) / 2
        mid_y = (y_c + y_e) / 2
        ax.text(mid_x, mid_y, f"{distances_m[i]:.1f}m", color=couleur,
                fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6))

    # --- Erreur ---
    erreur = np.linalg.norm(position_estimee_m - position_impact_m)
    ax.set_title(f"Localisation par multilatération\n"
                 f"Erreur : {erreur:.2f} m  |  t_impact = {t_impact:.3f}s", fontsize=13)

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.show()
    return position_impact_m, t_impact


def plot_positions_estimees(capteurs, positions_estimees_m, positions_reelles_impact,
                            t_impacts, t0_par_pulse, objet, L):
    """
    capteurs                : liste de dicts avec "position" en noeuds et "nom"
    positions_estimees_m    : array[N_pulses, 2] en mètres
    positions_reelles_impact: liste de array[2] en mètres
    objet                   : dict avec "forme", "rayon_m" ou "taille_m", "vitesse_m_s"
    """
    h = L / nx

    fig, ax            = plt.subplots(figsize=(9, 9))
    n_capteurs         = len(capteurs)
    cmap_cap           = plt.cm.tab10
    couleurs_c         = [cmap_cap(i % 10) for i in range(n_capteurs)]
    positions_estimees_m = np.array(positions_estimees_m)

    # --- Taille de l'objet ---
    if objet["forme"] == "cercle":
        rayon_m = objet.get("rayon_m", 5.0)
    else:
        lx_m, ly_m = objet.get("taille_m", (5.0, 5.0))

    # --- Trajectoire réelle ---
    traj_x = [r[0] for r in positions_reelles_impact]
    traj_y = [r[1] for r in positions_reelles_impact]
    ax.plot(traj_x, traj_y, color="black", linestyle="--", linewidth=1.2,
            alpha=0.5, label="Trajectoire réelle", zorder=2)

    # --- Position réelle à chaque impact ---
    for k, (pos_r, t_imp) in enumerate(zip(positions_reelles_impact, t_impacts)):
        x_r, y_r = pos_r
        label_r = "Position réelle à l'impact" if k == 0 else "_nolegend_"

        if objet["forme"] == "cercle":
            ax.add_patch(plt.Circle(
                (x_r, y_r), rayon_m,
                color="black", alpha=0.15, zorder=3, label=label_r
            ))
        else:
            ax.add_patch(plt.Rectangle(
                (x_r - lx_m/2, y_r - ly_m/2), lx_m, ly_m,
                color="black", alpha=0.15, zorder=3, label=label_r
            ))
        ax.text(x_r + 0.5, y_r + 0.5, f"R{k+1}\nt={t_imp:.2f}s",
                color="black", fontsize=7, alpha=0.7)

    # --- Capteurs ---
    for i, capteur in enumerate(capteurs):
        ix, iy  = capteur["position"]
        x_c, y_c = ix * h, iy * h
        nom      = capteur.get("nom", f"C{i+1}")
        ax.scatter(x_c, y_c, color=couleurs_c[i], s=100, zorder=5, marker="^")
        ax.text(x_c + 0.5, y_c + 0.5, nom, color=couleurs_c[i], fontsize=10, fontweight="bold")

    # --- Positions estimées ---
    cmap   = plt.cm.plasma
    colors = [cmap(k / max(len(positions_estimees_m) - 1, 1)) for k in range(len(positions_estimees_m))]
    erreurs = []

    for k, (pos_m, pos_r, couleur, t0) in enumerate(zip(positions_estimees_m, positions_reelles_impact, colors, t0_par_pulse)):
        x_e, y_e = pos_m
        x_r, y_r = pos_r
        erreur    = np.linalg.norm(pos_m - np.array(pos_r))
        erreurs.append(erreur)

        ax.scatter(x_e, y_e, marker="*", color=couleur, s=250, zorder=6,
                   edgecolors="black", linewidths=0.8,
                   label=f"P{k+1} | t={t0:.2f}s | err={erreur:.1f}m")
        ax.text(x_e + 0.5, y_e + 0.5, f"P{k+1}", color=couleur, fontsize=9, fontweight="bold")
        ax.plot([x_r, x_e], [y_r, y_e], color=couleur, linestyle=":", linewidth=1.0, alpha=0.7)

    # --- Trajectoire estimée ---
    if len(positions_estimees_m) >= 2:
        ax.plot(positions_estimees_m[:, 0], positions_estimees_m[:, 1],
                color="orange", linestyle="-", linewidth=1.5,
                alpha=0.8, label="Trajectoire estimée", zorder=4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(t0_par_pulse), vmax=max(t0_par_pulse)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Temps d'émission [s]", shrink=0.6)

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.set_title(f"Évolution des positions estimées\nErreur moyenne : {np.mean(erreurs):.2f} m", fontsize=13)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


def estimer_vitesse(positions_estimees_m, t_impacts, objet, show_vitesse = True):
    """
    positions_estimees_m : array[N_pulses, 2] déjà en mètres
    t_impacts            : liste des temps d'impact [s]
    objet                : dict avec "vitesse_m_s" pour comparaison
    """
    positions_estimees_m = np.array(positions_estimees_m)
    t_impacts            = np.array(t_impacts)

    if len(positions_estimees_m) < 2:
        print("⚠ Pas assez de positions pour estimer la vitesse.")
        return None, None

    A = np.column_stack([t_impacts, np.ones(len(t_impacts))])

    sol_x, _, _, _ = np.linalg.lstsq(A, positions_estimees_m[:, 0], rcond=None)
    sol_y, _, _, _ = np.linalg.lstsq(A, positions_estimees_m[:, 1], rcond=None)

    vx_estime, x0_estime = sol_x
    vy_estime, y0_estime = sol_y
    vitesse_estimee       = np.array([vx_estime, vy_estime])

    print(f"\n{'='*45}")
    print(f"  Estimation de la vitesse")
    print(f"{'='*45}")
    print(f"  vx estimé : {vx_estime:.2f} m/s")
    print(f"  vy estimé : {vy_estime:.2f} m/s")
    print(f"  ||v||     : {np.linalg.norm(vitesse_estimee):.2f} m/s")

    # --- Comparaison avec vitesse réelle ---
    if objet is not None and "vitesse_m_s" in objet:
        vx_reel, vy_reel = objet["vitesse_m_s"]
        vrai_v   = np.array([vx_reel, vy_reel])
        erreur_v = np.linalg.norm(vitesse_estimee - vrai_v)
        print(f"  vx réel   : {vx_reel:.2f} m/s")
        print(f"  vy réel   : {vy_reel:.2f} m/s")
        print(f"  Erreur    : {erreur_v:.2f} m/s")
    print(f"{'='*45}")
    if not show_vitesse:
        return vitesse_estimee, (x0_estime, y0_estime)
    # --- Graphique ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    t_fine    = np.linspace(t_impacts[0], t_impacts[-1], 100)

    for ax, coord, sol, label in zip(
        axes,
        [positions_estimees_m[:, 0], positions_estimees_m[:, 1]],
        [sol_x, sol_y],
        ["x [m]", "y [m]"]
    ):
        v_est, p0_est = sol
        ax.scatter(t_impacts, coord, color="blue", zorder=5, label="Positions estimées")
        ax.plot(t_fine, v_est * t_fine + p0_est, color="red", linestyle="--",
                label=f"Régression : v = {v_est:.2f} m/s")
        ax.set_xlabel("Temps d'impact [s]")
        ax.set_ylabel(label)
        ax.set_title(f"Régression linéaire — {label}")
        ax.legend()
        ax.grid(True, alpha=0.4)

    plt.suptitle("Estimation de la vitesse", fontsize=13)
    plt.tight_layout()
    plt.show()

    return vitesse_estimee, (x0_estime, y0_estime)
# =========================================================
# EXEMPLE D'UTILISATION
# =========================================================

L = 1000
t_total = 1.6
nx = 300
ny = 300

rho = 1000
kappa = 2.2e9
cfl = 1

epaisseur_pml_ratio = 0.2
puissance_pml = 2
gamma_max = 3000

sigma_source = 1
f_source = 1
A0_source = 8e10
tau_source = 0.001
nb_frames = 100
seed = 42



# capteurs aléatoires
capteurs = [
    {
        "nom": "Capteur 1",
        "position": (200, 50),
        "emissions": [0.01]
    },
    {
        "nom": "Capteur 2",
        "position": (200, 150),
        "emissions": [0.5]
    },
    {
        "nom": "Capteur 3",
        "position": (75, 150),
        "emissions": [1]    },
   
]


# objet
objet = {
    "forme": "cercle",              # "cercle" ou "rectangle"
    "centre_init_m": (500.0, 400.0),  # position initiale en mètres
    "vitesse_m_s": (50.0, 0.0),     # vitesse (vx, vy) en m/s
     "rayon_m": 10.0,                 # pour un cercle
    # "taille_m": (10.0, 2.0),       # pour un rectangle
    "c_objet": 5200.0,              # vitesse du son dans l'objet
    "gamma_objet": 4000.0,          # amortissement dans l'objet
    "actif": True,
}



run_methode_multilateration = True

if run_methode_multilateration:
    methode_multilateration() # Modifier les paramètres d'affichage dans la fonction methode_multilateration

else: 
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

