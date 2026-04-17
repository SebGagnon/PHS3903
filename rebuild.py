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
    Estime la position de l'objet pour un pulse à partir
    des distances estimées des 3 capteurs.
    """
    capteurs_m = positions_capteurs_m(params)
    distances = np.array(resultat_pulse["distances"], dtype=float)

    masque_valide = np.isfinite(distances)
    capteurs_valides = capteurs_m[masque_valide]
    distances_valides = distances[masque_valide]

    if len(distances_valides) < 2:
        raise ValueError("Pas assez de distances valides pour estimer une position.")

    def residus(pos):
        x, y = pos
        d_calc = np.sqrt((capteurs_valides[:, 0] - x)**2 + (capteurs_valides[:, 1] - y)**2)
        return d_calc - distances_valides

    x0 = np.mean(capteurs_valides[:, 0])
    y0 = np.mean(capteurs_valides[:, 1])

    sol = least_squares(
        residus,
        x0=np.array([x0, y0]),
        bounds=([0.0, 0.0], [params["L"], params["L"]])
    )

    return {
        "pulse_id": resultat_pulse["pulse_id"],
        "capteur_emetteur": resultat_pulse["capteur_emetteur"],
        "position_estimee": sol.x,
        "cout": sol.cost,
        "succes": sol.success,
        "message": sol.message,
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

L = 1000.0
nx = 400
ny = 400
rayon_objet_m = 12.0
ratio_pml = 0.2

position_objet_m = (600.0, 720.0)
position_c1_m = (350.0, 350.0)
position_c2_m = (300.0, 700.0)
position_c3_m = (700.0, 350.0)

params = {
    "L": L,
    "nx": nx,
    "ny": ny,
    "dt": 3e-4,
    "t_total": 4,
    "rho": 1000,
    "kappa": 2.2e9,
    "sigma": 1,                                                         
    "A0": 1000,
    "tau": 0.005,
    "epaisseur_pml_ratio": ratio_pml,
    "puissance_pml" : 0.5,
    "gamma_max_pml": 60, 
    "gamma_eau" : 0.3,
    "capteurs": [
        {"nom": "Capteur 1", "position": m_to_grid(L, nx, ny, position_c1_m), "emissions": [0.020]},
        {"nom": "Capteur 2", "position": m_to_grid(L, nx, ny, position_c2_m), "emissions": [1]},
        {"nom": "Capteur 3", "position": m_to_grid(L, nx, ny, position_c3_m), "emissions": [1.75]},
    ],

    "objet": {
        "position": m_to_grid(L, nx, ny, position_objet_m),
        "rayon": rayon_m_to_grid(L, nx, rayon_objet_m),
        "rho": 7800,
        "kappa": 1.6e11,
        "gamma": 2,
        "vitesse_m_s": (0.0, 0.0),
    },
}

MODE = 3 # 1 -- simul simple, 2 -- multilateration , 3 -- boucle pour tests

if MODE == 1 :
    resultats = run_simulation(params)
    animer_resultats(params, resultats)

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

    liste_nx = [10,20,30,40,50,60,70,80,90,100,125,150,175,200,225,250,275,300,325,350,375,400]
    liste_dt = [1e-4,1.5e-4,2e-4,2.5e-4,3e-4,3.5e-4,4e-4,4.5e-4, 5e-4,5.5e-4,6e-4,6.5e-4, 7e-4,7.5e-4,8e-4]

    resultats_maillage = etude_erreur_maillage(
        params,
        liste_nx,
        nom_pickle="etude_maillage2.pkl"
    )

    resultats_dt = etude_erreur_dt(
        params, 
        liste_dt,
        nom_pickle="etude_dt2.pkl"
    )
