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
import copy
from matplotlib.patches import Circle
from scipy.signal import hilbert
from scipy.optimize import least_squares
from skopt import gp_minimize
from skopt.space import Real

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
    """Source totale avec wavelet de Ricker défini par la fréquence centrale f_c."""
    nx = params["nx"]
    ny = params["ny"]
    sigma = params["sigma"]
    A0 = params["A0"]
    f_c = params["f_c"]   # fréquence centrale en Hz

    pulses = construction_pulses(params)

    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    source = np.zeros((nx, ny))

    for x0, y0, t0 in pulses:
        enveloppe_spatiale = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

        a = np.pi * f_c * (t - t0)
        enveloppe_temporelle = (1 - 2 * a**2) * np.exp(-a**2)

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

def positions_capteurs_m(params):
    """
    Retourne les positions des capteurs en mètres.
    """
    hx = params["L"] / params["nx"]
    hy = params["L"] / params["ny"]

    positions = []
    for capteur in params["capteurs"]:
        ix, iy = capteur["position"]
        x_m = (ix + 0.5) * hx
        y_m = (iy + 0.5) * hy
        positions.append((x_m, y_m))

    return np.array(positions, dtype=float)

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
    """Construit le tuple des pulses (ix, iy, t0) à partir de params['capteurs']."""
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

def animer_resultats_db(params, resultats, duree_animation_ms=3000, fraction_db_min=0.785):
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

    db_max = max(np.max(frame_db) for frame_db in frames_db)
    db_min = fraction_db_min * db_max

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

def plot_signaux_db(params, resultats, p_ref=1e-6, epsilon=1e-12):
    """Plot des mesures de chaque capteur en dB."""
    temps = resultats["temps"]
    signaux = resultats["signaux_capteurs"]
    couleurs = couleurs_capteurs(params)

    plt.figure(figsize=(8, 5))

    for i, (signal, couleur) in enumerate(zip(signaux, couleurs)):
        signal = np.asarray(signal)
        signal_db = 20 * np.log10(np.maximum(np.abs(signal), epsilon) / p_ref)
        plt.plot(temps, signal_db, color=couleur, label=f"Capteur {i+1}")

    plt.xlabel("Temps [s]")
    plt.ylabel("Niveau de pression [dB]")
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
        source = sourcetot(params, t)

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
        "trajectoire_objet": trajectoire_objet,
    }

def calcule_echos(params):
    """
    Fait deux simulations :
    - avec l'objet
    - sans objet
    puis retourne les échos = signal_avec - signal_sans
    """

    params_avec = copy.deepcopy(params)
    params_sans = copy.deepcopy(params)

    # sans objet
    params_sans["objet"] = None

    resultats_avec = run_simulation(params_avec)
    resultats_sans = run_simulation(params_sans)

    signaux_avec = resultats_avec["signaux_capteurs"]
    signaux_sans = resultats_sans["signaux_capteurs"]

    intensite_echo = [
        signal_avec - signal_sans
        for signal_avec, signal_sans in zip(signaux_avec, signaux_sans)
    ]

    return {
        "temps": resultats_avec["temps"],
        "signaux_avec": signaux_avec,
        "signaux_sans": signaux_sans,
        "intensite_echo": [np.array(signal) for signal in intensite_echo],
        "resultats_avec": resultats_avec,
        "resultats_sans": resultats_sans,
    }

def get_plages_pulses(params):
    """
    Construit les plages temporelles à partir des émissions,

    """
    capteurs = params["capteurs"]
    t_total = params["t_total"]

    pulses = []

    for idx_capteur, capteur in enumerate(capteurs):
        for emission in capteur.get("emissions", []):
            if isinstance(emission, dict):
                t0 = emission["t0"]
            else:
                t0 = float(emission)

            pulses.append((idx_capteur, t0))

    if not pulses:
        print("⚠ Aucun pulse détecté dans les capteurs.")
        return []

    pulses.sort(key=lambda x: x[1])
    temps_emission = [pulse[1] for pulse in pulses]

    plages = []
    for i, (idx_capteur, t0) in enumerate(pulses):
        t_fin = temps_emission[i + 1] if i + 1 < len(pulses) else t_total
        plages.append((idx_capteur, t0, t_fin))

    return plages

def max_enveloppe_plage(signal, temps, t_debut, t_fin, seuil_montee=1):
    """
    Trouve le max d'une enveloppe avec la TF d'Hilbert, et calcul le temps associé à ce max
    """
    masque = (temps >= t_debut) & (temps < t_fin)

    if not np.any(masque):
        return None, None, np.abs(hilbert(signal))

    signal_plage = signal[masque]
    temps_plage = temps[masque]
    enveloppe_plage = np.abs(hilbert(signal_plage))

    idx_pic = np.argmax(enveloppe_plage)
    amp_pic = enveloppe_plage[idx_pic]

    seuil = seuil_montee * amp_pic
    idx_debut = idx_pic

    while idx_debut > 0 and enveloppe_plage[idx_debut] > seuil:
        idx_debut -= 1

    enveloppe = np.abs(hilbert(signal))
    return temps_plage[idx_debut], enveloppe_plage[idx_debut], enveloppe

def calcule_distances_pulses(params, detections):
    """
    Convertit les temps d'écho détectés en distances.
    Retourne une liste de dictionnaires, un par pulse.
    """
    c_eau = np.sqrt(params["kappa"] / params["rho"])
    n_capteurs = len(params["capteurs"])

    resultats_distances = []

    for detection in detections:
        idx_emit = detection["capteur_emetteur"]
        t_emission = detection["t_emission"]
        temps_echos = detection["temps_echos"]

        if temps_echos[idx_emit] is None:
            raise ValueError(f"Aucun temps d'écho détecté pour le capteur émetteur C{idx_emit+1}")

        d_emit = c_eau * (temps_echos[idx_emit] - t_emission) / 2

        distances = np.zeros(n_capteurs)
        distances[idx_emit] = d_emit

        for i in range(n_capteurs):
            if i == idx_emit:
                continue

            if temps_echos[i] is None:
                distances[i] = np.nan
            else:
                distances[i] = c_eau * (temps_echos[i] - t_emission) - d_emit

        resultats_distances.append({
            "pulse_id": detection["pulse_id"],
            "capteur_emetteur": idx_emit,
            "t_emission": t_emission,
            "temps_echos": detection["temps_echos"],
            "distances": distances,
        })

    return resultats_distances

def detecte_temps_echos(params, echos, seuil_montee=1):
    """
    Détecte les temps d'écho pour chaque pulse et chaque capteur.
    Retourne une liste de dictionnaires, un par pulse.
    """
    temps = echos["temps"]
    intensite_echo = echos["intensite_echo"]
    plages = get_plages_pulses(params)

    detections = []

    for k, (idx_emit, t_debut, t_fin) in enumerate(plages):
        detection_pulse = {
            "pulse_id": k,
            "capteur_emetteur": idx_emit,
            "t_emission": t_debut,
            "t_fin": t_fin,
            "temps_echos": [],
            "amplitudes": [],
        }

        for i, signal in enumerate(intensite_echo):
            t_echo, amp_echo, _ = max_enveloppe_plage(
                signal,
                temps,
                t_debut,
                t_fin,
                seuil_montee=seuil_montee
            )

            detection_pulse["temps_echos"].append(t_echo)
            detection_pulse["amplitudes"].append(amp_echo)

        detections.append(detection_pulse)

    return detections

def affiche_detections(detections):
    """
    Affiche les temps détectés pulse par pulse.
    """
    for detection in detections:
        print(f"\nPulse {detection['pulse_id'] + 1}")
        print(f"  Emetteur : C{detection['capteur_emetteur'] + 1}")
        print(f"  t_emission = {detection['t_emission']:.6f} s")
        print(f"  t_fin      = {detection['t_fin']:.6f} s")

        for i, (t_echo, amp) in enumerate(zip(detection["temps_echos"], detection["amplitudes"])):
            if t_echo is None:
                print(f"  Capteur {i+1} : aucun écho détecté")
            else:
                print(f"  Capteur {i+1} : t_echo = {t_echo:.6f} s | amplitude = {amp:.6f}")

def erreur_bord_pulse(params, position_estimee, resultat_pulse):
    """
    Erreur entre la position estimee et le bord de l'objet.
    """
    hx = params["L"] / params["nx"]
    hy = params["L"] / params["ny"]

    x_est, y_est = position_estimee["position_estimee"]

    t_reel = resultat_pulse["t_emission"]
    x_obj_idx, y_obj_idx = position_objet_t(params, t_reel)

    x_reel = (x_obj_idx + 0.5) * hx
    y_reel = (y_obj_idx + 0.5) * hy

    rayon_objet_m = params["objet"]["rayon"] * hx

    distance_centre = np.sqrt((x_est - x_reel)**2 + (y_est - y_reel)**2)
    erreur = abs(distance_centre - rayon_objet_m)

    return erreur

def erreur_bord_pulse(params, position_estimee, resultat_pulse):
    """
    Erreur entre la position estimee et le bord de l'objet.
    """
    hx = params["L"] / params["nx"]
    hy = params["L"] / params["ny"]

    x_est, y_est = position_estimee["position_estimee"]

    t_reel = resultat_pulse["t_emission"]
    x_obj_idx, y_obj_idx = position_objet_t(params, t_reel)

    x_reel = (x_obj_idx + 0.5) * hx
    y_reel = (y_obj_idx + 0.5) * hy

    rayon_objet_m = params["objet"]["rayon"] * hx

    distance_centre = np.sqrt((x_est - x_reel)**2 + (y_est - y_reel)**2)
    erreur = abs(distance_centre - rayon_objet_m)

    return erreur

def calcule_c_max(params):
    c_eau = np.sqrt(params["kappa"] / params["rho"])

    if params.get("objet") is not None and "rho" in params["objet"] and "kappa" in params["objet"]:
        c_objet = np.sqrt(params["objet"]["kappa"] / params["objet"]["rho"])
        return max(c_eau, c_objet)

    return c_eau

def calcule_cfl(params):
    h = params["L"] / params["nx"]
    c_max = calcule_c_max(params)
    return c_max * params["dt"] / h

def liste_cfl_to_liste_nx(params, liste_cfl):
    """
    Convertit une liste de CFL en liste de nx, en gardant dt fixe.
    """
    c_max = calcule_c_max(params)
    L = params["L"]
    dt = params["dt"]

    liste_nx = []
    for cfl in liste_cfl:
        nx = int(round(cfl * L / (c_max * dt)))
        nx = max(nx, 5)
        liste_nx.append(nx)

    return liste_nx

def liste_cfl_to_liste_dt(params, liste_cfl):
    """
    Convertit une liste de CFL en liste de dt, en gardant nx fixe.
    """
    c_max = calcule_c_max(params)
    h = params["L"] / params["nx"]

    liste_dt = []
    for cfl in liste_cfl:
        dt = cfl * h / c_max
        liste_dt.append(dt)

    return liste_dt

def calcule_erreurs_multilateration(params):
    """
    Lance toute la chaine et retourne les erreurs pour les pulses.
    """
    echos = calcule_echos(params)
    detections = detecte_temps_echos(params, echos)
    distances_pulses = calcule_distances_pulses(params, detections)
    positions_estimees = estime_positions(params, distances_pulses)

    erreurs = []
    details = []

    n_pulses = min(3, len(distances_pulses))

    for i in range(n_pulses):
        err = erreur_bord_pulse(params, positions_estimees[i], distances_pulses[i])
        erreurs.append(err)

        details.append({
            "pulse_id": i,
            "erreur": err,
            "position_estimee": positions_estimees[i]["position_estimee"],
            "t_emission": distances_pulses[i]["t_emission"],
            "distances": distances_pulses[i]["distances"],
        })

    erreur_moyenne = np.mean(erreurs) if len(erreurs) > 0 else np.nan

    return {
        "erreurs_pulses": erreurs,
        "erreur_moyenne": erreur_moyenne,
        "details": details,
    }

def etude_erreur_maillage(params_base, liste_nx, nom_pickle="etude_maillage.pkl"):
    resultats = {}

    for nx in liste_nx:
        params = copy.deepcopy(params_base)
        params["nx"] = nx
        params["ny"] = nx

        params["capteurs"][0]["position"] = m_to_grid(params["L"], params["nx"], params["ny"], position_c1_m)
        params["capteurs"][1]["position"] = m_to_grid(params["L"], params["nx"], params["ny"], position_c2_m)
        params["capteurs"][2]["position"] = m_to_grid(params["L"], params["nx"], params["ny"], position_c3_m)

        params["objet"]["position"] = m_to_grid(params["L"], params["nx"], params["ny"], position_objet_m)
        params["objet"]["rayon"] = rayon_m_to_grid(params["L"], params["nx"], rayon_objet_m)

        print(f"Test maillage nx=ny={nx}, dt={params['dt']}")

        bilan = calcule_erreurs_multilateration(params)

        resultats[nx] = {
            "nx": nx,
            "ny": nx,
            "dt": params["dt"],
            "cfl": calcule_cfl(params),
            "erreurs_pulses": bilan["erreurs_pulses"],
            "erreur_moyenne": bilan["erreur_moyenne"],
            "details": bilan["details"],
        }

        with open(nom_pickle, "wb") as f:
            pickle.dump(resultats, f)

    return resultats

def etude_erreur_dt(params_base, liste_dt, nom_pickle="etude_dt.pkl"):
    """
    Fonction pour etude de l'erreur en fonction de dt
    """
    resultats = {}

    for dt in liste_dt:
        params = copy.deepcopy(params_base)
        params["dt"] = dt

        print(f"Test dt={dt}, nx=ny={params['nx']}")

        try:
            bilan = calcule_erreurs_multilateration(params)

            resultats[dt] = {
                "nx": params["nx"],
                "ny": params["ny"],
                "dt": dt,
                "cfl": calcule_cfl(params),
                "erreurs_pulses": bilan["erreurs_pulses"],
                "erreur_moyenne": bilan["erreur_moyenne"],
                "details": bilan["details"],
            }

            print(f"  erreur moyenne = {bilan['erreur_moyenne']:.6f} m")

        except Exception as e:
            resultats[dt] = {
                "nx": params["nx"],
                "ny": params["ny"],
                "dt": dt,
                "erreurs_pulses": None,
                "erreur_moyenne": np.nan,
                "details": None,
                "erreur_message": str(e),
            }

            print(f"  ECHEC : {e}")

        with open(nom_pickle, "wb") as f:
            pickle.dump(resultats, f)

    return resultats

def affiche_echos(params, echos):
    """
    Affiche les échos mesurés aux capteurs
    """
    temps = echos["temps"]
    intensite_echo = echos["intensite_echo"]

    plt.figure(figsize=(10, 5))

    for i, signal in enumerate(intensite_echo):
        plt.plot(temps, signal, label=f"Capteur {i+1}")

    plt.xlabel("Temps [s]")
    plt.ylabel("Écho [kPa]")
    plt.title("Échos = signal avec objet - signal sans objet")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def affiche_echos_avec_enveloppe(params, echos):
    """
    Affiche les échos capteur par capteur dans chaque plage,
    avec l'enveloppe et le temps détecté.
    """
    temps = echos["temps"]
    intensite_echo = echos["intensite_echo"]
    plages = get_plages_pulses(params)

    if not plages:
        return

    n_pulses = len(plages)
    n_capteurs = len(intensite_echo)

    fig, axes = plt.subplots(n_pulses, 1, figsize=(10, 4 * n_pulses), sharex=False)

    if n_pulses == 1:
        axes = [axes]

    couleurs = plt.cm.tab10.colors
    labels = [f"Capteur {i+1}" for i in range(n_capteurs)]

    for k, (idx_emit, t_debut, t_fin) in enumerate(plages):
        ax = axes[k]
        masque = (temps >= t_debut) & (temps < t_fin)

        for i, signal in enumerate(intensite_echo):
            couleur = couleurs[i % len(couleurs)]

            ax.plot(
                temps[masque],
                signal[masque],
                color=couleur,
                alpha=0.4
            )

            t_max, amp_max, enveloppe = max_enveloppe_plage(
                signal, temps, t_debut, t_fin
            )

            ax.plot(
                temps[masque],
                enveloppe[masque],
                color=couleur,
                linestyle="--",
                label=labels[i]
            )

            if t_max is not None:
                ax.scatter(t_max, amp_max, color=couleur, s=80, zorder=5)
                ax.axvline(t_max, color=couleur, linestyle=":", alpha=0.7)

        ax.axvline(
            t_debut,
            color="black",
            linestyle="-",
            linewidth=1.5,
            label=f"Émission C{idx_emit+1} (t={t_debut:.3f}s)"
        )

        ax.set_xlim(t_debut, t_fin)
        ax.set_xlabel("Temps [s]")
        ax.set_ylabel("Écho [kPa]")
        ax.set_title(f"Pulse {k+1} — Émetteur C{idx_emit+1} | [{t_debut:.3f}s, {t_fin:.3f}s]")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.4)

    plt.suptitle("Échos par plage de pulse avec enveloppe", fontsize=13)
    plt.tight_layout()
    plt.show()

def affiche_distances_pulses(distances_pulses):
    """
    Affiche les distances estimées pulse par pulse.
    """
    for resultat in distances_pulses:
        print(f"\nPulse {resultat['pulse_id'] + 1}")
        print(f"  Emetteur : C{resultat['capteur_emetteur'] + 1}")
        print(f"  t_emission = {resultat['t_emission']:.6f} s")

        for i, d in enumerate(resultat["distances"]):
            if np.isnan(d):
                print(f"  Capteur {i+1} : distance non disponible")
            else:
                print(f"  Capteur {i+1} : distance estimee = {d:.4f} m")
def estime_position_un_pulse(params, resultat_pulse):
    """
    Même multilateration qu'avant, mais avec plusieurs points de départ
    pour éviter de converger vers un mauvais minimum local.
    """
    capteurs_m = positions_capteurs_m(params)
    distances = np.array(resultat_pulse["distances"], dtype=float)

    masque_valide = np.isfinite(distances) & (distances > 0)
    capteurs_valides = capteurs_m[masque_valide]
    distances_valides = distances[masque_valide]

    if len(distances_valides) < 3:
        raise ValueError("Pas assez de distances valides pour estimer une position.")

    def residus(pos):
        x, y = pos
        d_calc = np.sqrt((capteurs_valides[:, 0] - x)**2 + (capteurs_valides[:, 1] - y)**2)
        return d_calc - distances_valides

    L = params["L"]

    points_init = []

    # centre des capteurs
    points_init.append(np.mean(capteurs_valides, axis=0))

    # chaque capteur
    for p in capteurs_valides:
        points_init.append(np.array(p, dtype=float))

    # # milieux entre paires de capteurs
    # for i in range(len(capteurs_valides)):
    #     for j in range(i + 1, len(capteurs_valides)):
    #         points_init.append(0.5 * (capteurs_valides[i] + capteurs_valides[j]))



    meilleur_sol = None
    meilleur_score = np.inf

    for x0 in points_init:
        try:
            sol = least_squares(
                residus,
                x0=np.clip(x0, 0.0, L),
                bounds=([0.0, 0.0], [L, L]),
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
                max_nfev=5000
            )

            score = np.sum(sol.fun**2)

            if score < meilleur_score:
                meilleur_score = score
                meilleur_sol = sol

        except Exception:
            pass

    if meilleur_sol is None:
        raise RuntimeError("Échec de la multilateration.")

    return {
        "pulse_id": resultat_pulse["pulse_id"],
        "capteur_emetteur": resultat_pulse["capteur_emetteur"],
        "position_estimee": meilleur_sol.x,
        "cout": meilleur_sol.cost,
        "succes": meilleur_sol.success,
        "message": meilleur_sol.message,
    }

def estime_positions(params, distances_pulses):
    """
    Estime la position pour tous les pulses.
    """
    positions_estimees = []

    for resultat_pulse in distances_pulses:
        position_estimee = estime_position_un_pulse(params, resultat_pulse)
        positions_estimees.append(position_estimee)

    return positions_estimees

def affiche_position_un_pulse(params, position_estimee, t_reel):
    """
    Affiche la position estimee, la position reelle et l'erreur pour un pulse.
    """
    x_est, y_est = position_estimee["position_estimee"]

    x_obj_idx, y_obj_idx = position_objet_t(params, t_reel)

    hx = params["L"] / params["nx"]
    hy = params["L"] / params["ny"]

    x_reel = (x_obj_idx + 0.5) * hx
    y_reel = (y_obj_idx + 0.5) * hy

    distance_au_centre = np.sqrt((x_est - x_reel)**2 + (y_est - y_reel)**2)
    rayon_objet_m = None
    if params.get("objet") is not None and "rayon" in params["objet"]:
        rayon_objet_m = params["objet"]["rayon"] * hx
    if rayon_objet_m is not None:
        erreur = abs(distance_au_centre - rayon_objet_m)
    else:
        erreur = distance_au_centre

    print(f"Position estimee : x = {x_est:.3f} m, y = {y_est:.3f} m")
    print(f"Position reelle  : x = {x_reel:.3f} m, y = {y_reel:.3f} m")
    print(f"Erreur           : {erreur:.3f} m")

def affiche_positions(params, positions_estimees, distances_pulses):
    """
    Affiche les positions estimees, reelles et les erreurs pour tous les pulses.
    """
    hx = params["L"] / params["nx"]
    hy = params["L"] / params["ny"]


    for position_estimee, resultat_pulse in zip(positions_estimees, distances_pulses):
        x_est, y_est = position_estimee["position_estimee"]
        t_reel = resultat_pulse["t_emission"]

        x_obj_idx, y_obj_idx = position_objet_t(params, t_reel)
        x_reel = (x_obj_idx + 0.5) * hx
        y_reel = (y_obj_idx + 0.5) * hy

        distance_au_centre = np.sqrt((x_est - x_reel)**2 + (y_est - y_reel)**2)
        rayon_objet_m = None
        if params.get("objet") is not None and "rayon" in params["objet"]:
            rayon_objet_m = params["objet"]["rayon"] * hx
        if rayon_objet_m is not None:
            erreur = abs(distance_au_centre - rayon_objet_m)
        else:
            erreur = distance_au_centre

        print(f"\nPulse {position_estimee['pulse_id'] + 1}")
        print(f"Position estimee : x = {x_est:.3f} m, y = {y_est:.3f} m")
        print(f"Position reelle  : x = {x_reel:.3f} m, y = {y_reel:.3f} m")
        print(f"Erreur           : {erreur:.3f} m")

def tracer_positions_estimees_et_reelles(params, positions_estimees, distances_pulses, resultats=None):
    """
    Trace les positions estimees et les positions reelles de l'objet, et sa trajectoire.
    
    """
    hx = params["L"] / params["nx"]
    hy = params["L"] / params["ny"]

    # Capteurs en metres
    capteurs_x = []
    capteurs_y = []
    for capteur in params["capteurs"]:
        ix, iy = capteur["position"]
        capteurs_x.append((ix + 0.5) * hx)
        capteurs_y.append((iy + 0.5) * hy)

    # Positions estimees
    x_est = []
    y_est = []

    # Positions reelles aux instants des pulses
    x_reel = []
    y_reel = []

    for position_estimee, resultat_pulse in zip(positions_estimees, distances_pulses):
        x_e, y_e = position_estimee["position_estimee"]
        x_est.append(x_e)
        y_est.append(y_e)

        t_reel = resultat_pulse["t_emission"]
        x_obj_idx, y_obj_idx = position_objet_t(params, t_reel)

        x_r = (x_obj_idx + 0.5) * hx
        y_r = (y_obj_idx + 0.5) * hy

        x_reel.append(x_r)
        y_reel.append(y_r)

    plt.figure(figsize=(8, 8))

    # Trajectoire 
    if resultats is not None and resultats.get("trajectoire_objet_m") is not None:
        traj = resultats["trajectoire_objet_m"]
        plt.plot(traj[:, 0], traj[:, 1], "k-", alpha=0.5, label="Trajectoire reelle")

    # Points reels aux pulses
    plt.plot(x_reel, y_reel, "go--", label="Positions reelles aux pulses")

    # Points estimes
    plt.plot(x_est, y_est, "ro--", label="Positions estimees")

    # Capteurs
    plt.scatter(capteurs_x, capteurs_y, c="blue", s=80, marker="^", label="Capteurs")

    for i, (x_c, y_c) in enumerate(zip(capteurs_x, capteurs_y)):
        plt.text(x_c + 5, y_c + 5, f"C{i+1}")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Positions estimees vs positions reelles")
    plt.xlim(0, params["L"])
    plt.ylim(0, params["L"])
    plt.axis("equal")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

def tracer_localisation_un_pulse(params, position_estimee, resultat_pulse):
    """
    Trace la localisation pour un seul pulse avec les cercles de distance
    et l'objet reel.
    """
    fig, ax = plt.subplots(figsize=(9, 9))

    hx = params["L"] / params["nx"]
    hy = params["L"] / params["ny"]

    capteurs_m = positions_capteurs_m(params)
    distances = np.array(resultat_pulse["distances"], dtype=float)

    x_est, y_est = position_estimee["position_estimee"]

    t_reel = resultat_pulse["t_emission"]
    x_obj_idx, y_obj_idx = position_objet_t(params, t_reel)
    x_reel = (x_obj_idx + 0.5) * hx
    y_reel = (y_obj_idx + 0.5) * hy

    rayon_objet_m = None
    if params.get("objet") is not None and "rayon" in params["objet"]:
        rayon_objet_m = params["objet"]["rayon"] * hx

    erreur_centre = np.sqrt((x_est - x_reel)**2 + (y_est - y_reel)**2)

    if rayon_objet_m is not None:
        erreur_bord = abs(erreur_centre - rayon_objet_m)
    else:
        erreur_bord = erreur_centre

    couleurs = plt.cm.tab10.colors

    for i, ((x_c, y_c), d) in enumerate(zip(capteurs_m, distances)):
        couleur = couleurs[i % len(couleurs)]

        ax.scatter(x_c, y_c, color=couleur, s=120, marker=".", zorder=4)
        ax.text(x_c + 5, y_c + 5, f"C{i+1}", color=couleur, fontsize=10, fontweight="bold")

        if np.isfinite(d) and d > 0:
            cercle = plt.Circle(
                (x_c, y_c),
                radius=d,
                fill=False,
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                color=couleur
            )
            ax.add_patch(cercle)

            ax.plot([x_c, x_est], [y_c, y_est], color=couleur, linewidth=1.2, alpha=0.6)

            # xm = 0.5 * (x_c + x_est)
            # ym = 0.5 * (y_c + y_est)
            # ax.text(
            #     xm,
            #     ym,
            #     f"{d:.1f} m",
            #     color=couleur,
            #     fontsize=8,
            #     ha="center",
            #     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7)
            # )

    if rayon_objet_m is not None:
        objet_patch = plt.Circle(
            (x_reel, y_reel),
            radius=rayon_objet_m,
            facecolor="gray",
            edgecolor="black",
            alpha=0.25,
            linewidth=2,
            zorder=2
        )
        ax.add_patch(objet_patch)

    ax.scatter(
        x_est, y_est,
        s=9,
        color="crimson",
        zorder=6,
        label="Position estimee"
    )


    ax.set_title(
        f"Localisation par multilateration\n"
        f"Pulse {resultat_pulse['pulse_id'] + 1} | erreur bord = {erreur_bord:.2f} m"
    )
    ax.set_xlim(0, params["L"])
    ax.set_ylim(0, params["L"])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()

def m_to_grid(L, nx, ny, position_m):
    x_m, y_m = position_m

    hx = L / nx
    hy = L / ny

    ix = int(round(x_m / hx - 0.5))
    iy = int(round(y_m / hy - 0.5))

    return (ix, iy)

def rayon_m_to_grid(L, nx, rayon_m):
    h = L / nx
    return rayon_m / h

def fonction_cout_pml(x, params_normal_base, signaux_ref):
    gamma_max, puissance_pml = x

    params_test = copy.deepcopy(params_normal_base)
    params_test["gamma_max_pml"] = gamma_max
    params_test["puissance_pml"] = puissance_pml

    resultats = run_simulation(params_test)
    signaux_test = resultats["signaux_capteurs"]

    cout = 0.0
    for sig_ref, sig_test in zip(signaux_ref, signaux_test):
        if len(sig_ref) != len(sig_test):
            raise ValueError(
                f"Longueur differente entre reference ({len(sig_ref)}) "
                f"et test ({len(sig_test)}). Verifie dt et t_total."
            )
        cout += np.sum(np.abs(sig_test - sig_ref))

    print(
        f"gamma_max = {gamma_max:.3f}, "
        f"puissance_pml = {puissance_pml:.3f}, "
        f"cout = {cout:.6f}"
    )

    return cout

def zone_valide_capteurs_m(params, marge_interne_m=30.0):
    """
    Retourne les bornes valides en mètres pour placer les capteurs hors du pml.
    """
    L = params["L"]
    epaisseur_pml_m = params["epaisseur_pml_ratio"] * L

    borne_min = epaisseur_pml_m + marge_interne_m
    borne_max = L - epaisseur_pml_m - marge_interne_m

    if borne_min >= borne_max:
        raise ValueError("Zone valide trop petite pour placer les capteurs.")

    return borne_min, borne_max

def positions_capteurs_trop_proches(capteurs_m, distance_min_m=80.0):
    """
    Vérifie si au moins deux capteurs sont trop proches.
    """
    for i in range(len(capteurs_m)):
        for j in range(i + 1, len(capteurs_m)):
            d = np.sqrt(
                (capteurs_m[i][0] - capteurs_m[j][0])**2 +
                (capteurs_m[i][1] - capteurs_m[j][1])**2
            )
            if d < distance_min_m:
                return True
    return False

def applique_positions_capteurs(params, capteurs_m):
    """
    Remplace les positions des 3 capteurs dans params à partir de positions en mètres.
    """
    params_new = copy.deepcopy(params)

    for i, (x_m, y_m) in enumerate(capteurs_m):
        params_new["capteurs"][i]["position"] = m_to_grid(
            params_new["L"],
            params_new["nx"],
            params_new["ny"],
            (x_m, y_m)
        )

    return params_new

def trajectoire_valide(x0, y0, vx, vy, params, marge_bord_m=0.0):
    """
    Vérifie que la trajectoire complète reste hors du PML
    entre t=0 et t=t_total.
    """
    L = params["L"]
    t_total = params["t_total"]

    epaisseur_pml_m = params["epaisseur_pml_ratio"] * L
    borne_min = epaisseur_pml_m + marge_bord_m
    borne_max = L - epaisseur_pml_m - marge_bord_m

    x_fin = x0 + vx * t_total
    y_fin = y0 + vy * t_total

    return (
        borne_min <= x0 <= borne_max and
        borne_min <= y0 <= borne_max and
        borne_min <= x_fin <= borne_max and
        borne_min <= y_fin <= borne_max
    )

def genere_objet_aleatoire(params, rng, marge_bord_m=120.0,
                           vmin=0.0, vmax=20.0, max_essais=1000):
    """
    position initiale et une vitesse aléatoires pour l'objet
    """
    params_new = copy.deepcopy(params)
    L = params_new["L"]

    epaisseur_pml_m = params_new["epaisseur_pml_ratio"] * L
    borne_min = epaisseur_pml_m + marge_bord_m
    borne_max = L - epaisseur_pml_m - marge_bord_m

    if borne_min >= borne_max:
        raise ValueError("Zone valide trop petite pour générer un objet hors du PML.")

    for _ in range(max_essais):
        x0 = rng.uniform(borne_min, borne_max)
        y0 = rng.uniform(borne_min, borne_max)

        angle = rng.uniform(0, 2 * np.pi)
        vitesse = rng.uniform(vmin, vmax)

        vx = vitesse * np.cos(angle)
        vy = vitesse * np.sin(angle)

        if trajectoire_valide(x0, y0, vx, vy, params_new, marge_bord_m=marge_bord_m):
            params_new["objet"]["position"] = m_to_grid(
                params_new["L"],
                params_new["nx"],
                params_new["ny"],
                (x0, y0)
            )
            params_new["objet"]["vitesse_m_s"] = (vx, vy)

            return params_new, {
                "position_m": (x0, y0),
                "vitesse_m_s": (vx, vy),
                "vitesse_norme": vitesse,
                "angle_rad": angle,
            }

    raise RuntimeError("Impossible de générer une trajectoire valide hors du PML.")

def capteurs_valides(capteurs_m, params, marge_bord_m=20.0, distance_min_m=80.0):
    L = params["L"]
    epaisseur_pml_m = params["epaisseur_pml_ratio"] * L
    borne_min = epaisseur_pml_m + marge_bord_m
    borne_max = L - epaisseur_pml_m - marge_bord_m

    for x, y in capteurs_m:
        if not (borne_min <= x <= borne_max and borne_min <= y <= borne_max):
            return False

    for i in range(len(capteurs_m)):
        for j in range(i + 1, len(capteurs_m)):
            xi, yi = capteurs_m[i]
            xj, yj = capteurs_m[j]
            d = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if d < distance_min_m:
                return False

    return True

def cout_capteurs_bo(x, params_base, n_essais=8, seed=42,
                     marge_bord_m=20.0, distance_min_m=80.0,
                     penalite=1e6, verbose=True):
    """
    x = [x1, y1, x2, y2, x3, y3] en mètres
    """
    capteurs_m = [
        (x[0], x[1]),
        (x[2], x[3]),
        (x[4], x[5]),
    ]

    if not capteurs_valides(
        capteurs_m,
        params_base,
        marge_bord_m=marge_bord_m,
        distance_min_m=distance_min_m
    ):
        if verbose:
            print(f"Capteurs invalides -> pénalité {penalite}")
        return penalite

    rng = np.random.default_rng(seed)
    erreurs = []

    params_capteurs = copy.deepcopy(params_base)

    for i, (xc, yc) in enumerate(capteurs_m):
        params_capteurs["capteurs"][i]["position"] = m_to_grid(
            params_capteurs["L"],
            params_capteurs["nx"],
            params_capteurs["ny"],
            (xc, yc)
        )

    for k in range(n_essais):
        try:
            params_test, info = genere_objet_aleatoire(
                params_capteurs,
                rng,
                marge_bord_m=marge_bord_m,
                vmin=0.0,
                vmax=20.0
            )

            bilan = calcule_erreurs_multilateration(params_test)
            err = bilan["erreur_moyenne"]

            if not np.isfinite(err):
                err = penalite

            erreurs.append(err)

        except Exception as e:
            if verbose:
                print(f"Essai {k+1} échoué : {e}")
            erreurs.append(penalite)

    cout = float(np.mean(erreurs))

    if verbose:
        print("\nConfiguration testée")
        print(f"C1 = ({x[0]:.1f}, {x[1]:.1f}) m")
        print(f"C2 = ({x[2]:.1f}, {x[3]:.1f}) m")
        print(f"C3 = ({x[4]:.1f}, {x[5]:.1f}) m")
        print(f"Coût moyen = {cout:.4f} m")

    return cout

def cree_scenarios_fixes(params, n_scenarios=20, seed=140,
                         marge_bord_m=20.0, vmin=0.0, vmax=75.0):
    """
    Génère une liste fixe de scénarios bateau.
    """
    rng = np.random.default_rng(seed)
    scenarios = []

    for _ in range(n_scenarios):
        _, info = genere_objet_aleatoire(
            params,
            rng,
            marge_bord_m=marge_bord_m,
            vmin=vmin,
            vmax=vmax
        )
        scenarios.append({
            "position_m": info["position_m"],
            "vitesse_m_s": info["vitesse_m_s"],
        })

    return scenarios

def applique_scenario_objet(params, scenario):
    """
    Applique une position/vitesse d'objet déjà fixée.
    """
    params_new = copy.deepcopy(params)

    params_new["objet"]["position"] = m_to_grid(
        params_new["L"],
        params_new["nx"],
        params_new["ny"],
        scenario["position_m"]
    )
    params_new["objet"]["vitesse_m_s"] = scenario["vitesse_m_s"]

    return params_new

def construit_objectif_bo(params_base, scenarios, historique=None,
                          marge_bord_m=20.0, distance_min_m=80.0,
                          penalite=0, verbose=True):
    """
    Retourne une fonction f(x) pour l'optimisation bayésienne.
    x = [x1, y1, x2, y2, x3, y3] en mètres
    """
    if historique is None:
        historique = []

    def objectif(x):
        capteurs_m = [
            (x[0], x[1]),
            (x[2], x[3]),
            (x[4], x[5]),
        ]

        if not capteurs_valides(
            capteurs_m,
            params_base,
            marge_bord_m=marge_bord_m,
            distance_min_m=distance_min_m
        ):
            if verbose:
                print(f"Capteurs invalides -> pénalité {penalite}")
            historique.append({
                "x": list(x),
                "cout": penalite,
                "valide": False,
            })
            return penalite

        params_capteurs = copy.deepcopy(params_base)

        for i, (xc, yc) in enumerate(capteurs_m):
            params_capteurs["capteurs"][i]["position"] = m_to_grid(
                params_capteurs["L"],
                params_capteurs["nx"],
                params_capteurs["ny"],
                (xc, yc)
            )

        erreurs = []

        for scenario in scenarios:
            try:
                params_test = applique_scenario_objet(params_capteurs, scenario)
                bilan = calcule_erreurs_multilateration(params_test)
                err = bilan["erreur_moyenne"]

                if not np.isfinite(err):
                    err = penalite

            except Exception as e:
                if verbose:
                    print("Essai échoué :", e)
                err = penalite

            erreurs.append(err)

        cout = float(np.mean(erreurs))

        historique.append({
            "x": list(x),
            "cout": cout,
            "valide": True,
            "erreurs": erreurs,
        })

        if verbose:
            print("\nÉvaluation BO")
            print(f"C1 = ({x[0]:.1f}, {x[1]:.1f}) m")
            print(f"C2 = ({x[2]:.1f}, {x[3]:.1f}) m")
            print(f"C3 = ({x[4]:.1f}, {x[5]:.1f}) m")
            print(f"Coût moyen = {cout:.4f} m")

        return cout

    return objectif

def creer_params_essai(params_base, position_m, gamma_max, puissance_pml):
    params_test = copy.deepcopy(params_base)
    params_test.update({
        "L": 1000.0,
        "nx": 250,
        "ny": 250,
        "epaisseur_pml_ratio": 0.2,
        "gamma_max_pml": gamma_max,
        "puissance_pml": puissance_pml,
        "capteurs": [
            {
                "nom": "Capteur 1",
                "position": m_to_grid(1000.0, 250, 250, position_m),
                "emissions": [0.01],
            }
        ],
    })
    return params_test

def genere_essais_aleatoires(nb_essais, demi_cote=300.0, marge=20.0, seed=23):
    rng = np.random.default_rng(seed)
    essais = []

    for _ in range(nb_essais):
        dx = rng.uniform(-demi_cote + marge, demi_cote - marge)
        dy = rng.uniform(-demi_cote + marge, demi_cote - marge)
        position_m = (500.0 + dx, 500.0 + dy)
        essais.append(position_m)

    return essais

def cout_signal_apres_impulsion(params_test, resultats, facteur_tau=4.0):
    temps = np.array(resultats["temps"])
    signal = np.array(resultats["signaux_capteurs"][0])

    t0 = params_test["capteurs"][0]["emissions"][0]
    tau = params_test["tau"]

    t_debut = t0 + facteur_tau * tau
    masque = temps >= t_debut

    return np.sum(np.abs(signal[masque]))

def fonction_cout_pml(x, params_base, essais):
    gamma_max, puissance_pml = x
    cout_total = 0.0

    for position_m in essais:
        params_test = creer_params_essai(
            params_base=params_base,
            position_m=position_m,
            gamma_max=gamma_max,
            puissance_pml=puissance_pml,
        )

        resultats = run_simulation(params_test)
        cout_total += cout_signal_apres_impulsion(params_test, resultats)

    cout_moyen = cout_total / len(essais)

    print(
        f"gamma_max = {gamma_max:.3f}, "
        f"puissance_pml = {puissance_pml:.3f}, "
        f"cout = {cout_moyen:.6f}"
    )

    return cout_moyen

L = 1000.0
nx = 250
ny = 250
rayon_objet_m = 15.0
ratio_pml = 0.2
gamma_max = 0
puissance = 0.0
position_objet_m = (500.42, 432.77)
dt = 1e-4

position_c1_m = (500, 500)
position_c2_m = (647, 597)
position_c3_m = (663, 300)

params = {
    "L": L,
    "nx": nx,
    "ny": ny,
    "dt": dt,
    "t_total": 0.5,
    "rho": 1000,
    "kappa": 2.2e9,
    "sigma": 1,                                                         
    "A0": 15000,
    "f_c": 10,
    "epaisseur_pml_ratio": ratio_pml,
    "puissance_pml" : puissance,
    "gamma_max_pml": gamma_max, 
    "gamma_eau" : 0.3,
    "capteurs": [

        {"nom": "Capteur 1", "position": m_to_grid(L, nx, ny, position_c1_m), "emissions": [2500*dt]},
        # {"nom": "Capteur 2", "position": m_to_grid(L, nx, ny, position_c2_m), "emissions": [1]},
        # {"nom": "Capteur 3", "position": m_to_grid(L, nx, ny, position_c3_m), "emissions": [2]},
    ],

    # "objet": {
    #     "position": m_to_grid(L, nx, ny, position_objet_m),
    #     "rayon": rayon_m_to_grid(L, nx, rayon_objet_m),
    #     "rho": 7800,
    #     "kappa": 1.6e11,
    #     "gamma": 2,
    #     "vitesse_m_s": (8.80, -10.820),
    # },
}

MODE = 1 # 1 -- simul simple, 2 -- multilateration , 3 -- boucle pour tests

if MODE == 1 :
    ix, iy = params["capteurs"][0]["position"]
    temps = np.arange(0, params["t_total"], params["dt"])

    s = np.array([sourcetot(params, t)[ix, iy] for t in temps])

    plt.plot(temps, s)
    plt.xlabel("Temps [s]")
    plt.ylabel("Amplitude [Pa]")
    plt.grid(True)
    plt.show()
    resultats = run_simulation(params)
    # ani =animer_resultats(params, resultats)
    
    plot_signaux(params, resultats)
    

    
if MODE == 2 :
    # resultats = run_simulation(params)
    # animer_resultats(params, resultats)
    echos = calcule_echos(params)
    detections = detecte_temps_echos(params, echos)
    distances_pulses = calcule_distances_pulses(params, detections)
    positions_estimees = estime_positions(params, distances_pulses)

    for i in range(len(distances_pulses)):
        t_reel = distances_pulses[i]["t_emission"]
        affiche_position_un_pulse(params, positions_estimees[i], t_reel)
        tracer_localisation_un_pulse(params, positions_estimees[i], distances_pulses[i])

if MODE == 3 :
    liste_cfl = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1,1.15,1.2,1.25,1.3]

    liste_nx = liste_cfl_to_liste_nx(params, liste_cfl)
    liste_dt = liste_cfl_to_liste_dt(params, liste_cfl)

    resultats_maillage = etude_erreur_maillage(
        params,
        liste_nx,
        nom_pickle="etude_maillage_cfl.pkl"
    )

    resultats_dt = etude_erreur_dt(
        params,
        liste_dt,
        nom_pickle="etude_dt_cfl.pkl"
    )

if MODE == 4:
    nb_essais = 20
    essais = genere_essais_aleatoires(
        nb_essais=nb_essais,
        demi_cote=300.0,
        marge=5.0,
        seed=23,
    )

    espace = [
        Real(0.01, 150, name="gamma_max"),
        Real(0.01, 2, name="puissance_pml"),
    ]

    resultat_bo = gp_minimize(
        func=lambda x: fonction_cout_pml(x, params, essais),
        dimensions=espace,
        x0=[[60.0, 0.5]],
        n_calls=35,
        n_initial_points=7,
        acq_func="EI",
        random_state=12,
    )

    gamma_best, puissance_best = resultat_bo.x
    cout_verif = fonction_cout_pml(resultat_bo.x, params, essais)

    print("Meilleur cout :", resultat_bo.fun)
    print("Meilleurs parametres :", resultat_bo.x)
    print("Cout reverifie :", cout_verif)

    historique_bo = {
        "positions_essais_m": essais,
        "essais_bo": [
            {
                "gamma_max": x[0],
                "puissance_pml": x[1],
                "cout": y
            }
            for x, y in zip(resultat_bo.x_iters, resultat_bo.func_vals)
        ],
        "x_best": resultat_bo.x,
        "fun_best": resultat_bo.fun,
        "cout_verifie": cout_verif,
    }

    with open("optimisation_pml_residuel.pkl", "wb") as f:
        pickle.dump(historique_bo, f)
if MODE == 5:
    x_test = [
        0.35 * params["L"], 0.35 * params["L"],
        0.70 * params["L"], 0.70 * params["L"],
        0.55 * params["L"], 0.45 * params["L"],
    ]

    cout = cout_capteurs_bo(
        x_test,
        params,
        n_essais=8,
        seed=42,
        marge_bord_m=20.0,
        distance_min_m=80.0,
        verbose=True
    )

    print(f"\nCoût retourné = {cout:.4f} m")

if MODE == 6:
    from skopt import gp_minimize
    from skopt.space import Real
    import pickle
    import time

    marge_bord_m = 20.0
    distance_min_m = 80.0
    n_scenarios_train = 12

    L = params["L"]
    epaisseur_pml_m = params["epaisseur_pml_ratio"] * L
    borne_min = epaisseur_pml_m + marge_bord_m
    borne_max = L - epaisseur_pml_m - marge_bord_m

    print("\n==============================")
    print("LANCEMENT OPTIMISATION BO")
    print(f"L = {L:.2f} m")
    print(f"Épaisseur PML = {epaisseur_pml_m:.2f} m")
    print(f"Borne min = {borne_min:.2f} m")
    print(f"Borne max = {borne_max:.2f} m")
    print(f"Distance minimale entre capteurs = {distance_min_m:.2f} m")
    print(f"Nombre de scénarios train = {n_scenarios_train}")

    espace = [
        Real(borne_min, borne_max, name="x1"),
        Real(borne_min, borne_max, name="y1"),
        Real(borne_min, borne_max, name="x2"),
        Real(borne_min, borne_max, name="y2"),
        Real(borne_min, borne_max, name="x3"),
        Real(borne_min, borne_max, name="y3"),
    ]

    scenarios_train = cree_scenarios_fixes(
        params,
        n_scenarios=n_scenarios_train,
        seed=123,
        marge_bord_m=marge_bord_m,
        vmin=0.0,
        vmax=20.0
    )

    print("\nScénarios d'entraînement générés :", len(scenarios_train))

    historique = []

    objectif_brut = construit_objectif_bo(
        params,
        scenarios_train,
        historique=historique,
        marge_bord_m=marge_bord_m,
        distance_min_m=distance_min_m,
        penalite=1e6,
        verbose=True
    )

    compteur = {"n": 0, "best": np.inf, "best_x": None}
    t0_global = time.time()

    def objectif(x):
        compteur["n"] += 1
        print("\n----------------------------------------")
        print(f"ÉVALUATION {compteur['n']}")
        print(f"x = {x}")

        t0_eval = time.time()
        cout = objectif_brut(x)
        dt_eval = time.time() - t0_eval

        if cout < compteur["best"]:
            compteur["best"] = cout
            compteur["best_x"] = list(x)
            print(">>> NOUVEAU MEILLEUR COÛT <<<")

        print(f"Coût obtenu = {cout:.6f}")
        print(f"Meilleur coût à date = {compteur['best']:.6f}")
        print(f"Temps pour cette évaluation = {dt_eval:.2f} s")

        if compteur["best_x"] is not None:
            xb = compteur["best_x"]
            print("Meilleurs capteurs à date :")
            print(f"  C1 = ({xb[0]:.2f}, {xb[1]:.2f}) m")
            print(f"  C2 = ({xb[2]:.2f}, {xb[3]:.2f}) m")
            print(f"  C3 = ({xb[4]:.2f}, {xb[5]:.2f}) m")

        print("----------------------------------------")
        return cout

    res = gp_minimize(
        func=objectif,
        dimensions=espace,
        n_calls=50,
        n_initial_points=10,
        acq_func="LCB",
        random_state=42
    )

    temps_total = time.time() - t0_global

    print("\n==============================")
    print("OPTIMISATION TERMINÉE")
    print(f"Temps total = {temps_total:.2f} s")
    print("Nombre total d'évaluations :", compteur["n"])
    print("Meilleur coût train :", res.fun)
    print("Meilleure solution x :", res.x)

    capteurs_best = [
        (res.x[0], res.x[1]),
        (res.x[2], res.x[3]),
        (res.x[4], res.x[5]),
    ]

    print("\nMeilleurs capteurs finaux :")
    for i, (x, y) in enumerate(capteurs_best):
        print(f"  C{i+1} = ({x:.2f}, {y:.2f}) m")

    couts_train = [entry["cout"] for entry in historique if "cout" in entry]

    print("\nRésumé des coûts train :")
    print(f"  Nombre d'essais = {len(couts_train)}")
    print(f"  Coût min = {np.min(couts_train):.6f}")
    print(f"  Coût max = {np.max(couts_train):.6f}")
    print(f"  Coût moyen = {np.mean(couts_train):.6f}")

    with open("bo_capteurs_new.pkl", "wb") as f:
        pickle.dump({
            "res_x": res.x,
            "res_fun": res.fun,
            "capteurs_best": capteurs_best,
            "historique": historique,
            "couts_train": couts_train,
            "scenarios_train": scenarios_train,
        }, f)

    print("\nRésultats sauvés dans bo_capteurs_new.pkl")