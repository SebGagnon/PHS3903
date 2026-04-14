import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
from copy import deepcopy


def laplacien_9_points(u, h):
    """Calcule le laplacien 2D à 9 points."""
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


def source_gaussienne(nx, ny, t, x0, y0, sigma, A0, t0, tau):
    """Construit la source gaussienne spatiale et temporelle."""
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")

    enveloppe_spatiale = np.exp(
        -((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2)
    )
    enveloppe_temporelle = np.exp(
        -((t - t0) ** 2) / (2 * tau**2)
    )

    return A0 * enveloppe_spatiale * enveloppe_temporelle


def conditions_frontieres(u):
    """Conditions de Dirichlet nulles sur les bords."""
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    return u


def creer_pml(params):
    """Crée une matrice gamma_pml(x, y) à partir des paramètres."""
    nx = params["nx"]
    ny = params["ny"]
    epaisseur_ratio = params["epaisseur_pml_ratio"]
    puissance = params["puissance_pml"]
    gamma_max = params["gamma_max_pml"]

    epaisseur = int(epaisseur_ratio * nx)
    epaisseur = max(epaisseur, 1)

    gamma_pml = np.zeros((nx, ny))

    lignes, cols = np.indices((nx, ny))

    dist = np.minimum.reduce([
        lignes,
        cols,
        nx - 1 - lignes,
        ny - 1 - cols
    ])

    masque = dist < epaisseur
    s = (epaisseur - dist[masque]) / epaisseur

    gamma_pml[masque] = gamma_max * s**puissance

    return gamma_pml


def metre_vers_indice(x_m, y_m, L, nx, ny):
    """Convertit une position physique (m) en indices de grille."""
    ix = int(np.clip(x_m / L * nx, 0, nx - 1))
    iy = int(np.clip(y_m / L * ny, 0, ny - 1))
    return ix, iy


def ajouter_positions_physiques_si_absentes(params):
    """
    Ajoute source_pos_m et capteurs_m à partir des indices actuels
    si ces champs n'existent pas encore.
    """
    if "source_pos_m" not in params:
        params["source_pos_m"] = (
            (params["x0"] + 0.5) * params["L"] / params["nx"],
            (params["y0"] + 0.5) * params["L"] / params["ny"],
        )

    if "capteurs_m" not in params:
        params["capteurs_m"] = [
            (
                (ix + 0.5) * params["L"] / params["nx"],
                (iy + 0.5) * params["L"] / params["ny"],
            )
            for (ix, iy) in params["capteurs"]
        ]

    return params


def extraire_source_et_capteurs(params):
    """
    Récupère les positions de la source et des capteurs en indices.
    Si des positions physiques existent, elles sont converties pour la grille courante.
    """
    L = params["L"]
    nx = params["nx"]
    ny = params["ny"]

    if "source_pos_m" in params:
        x0, y0 = metre_vers_indice(
            params["source_pos_m"][0],
            params["source_pos_m"][1],
            L, nx, ny
        )
    else:
        x0 = params["x0"]
        y0 = params["y0"]

    if "capteurs_m" in params:
        capteurs = [
            metre_vers_indice(x_m, y_m, L, nx, ny)
            for (x_m, y_m) in params["capteurs_m"]
        ]
    else:
        capteurs = params["capteurs"]

    return x0, y0, capteurs


def run_simulation(params):
    """
    Lance une simulation de l'onde amortie avec PML.

    Paramètres requis:
    L, nx, ny, dt, t_total, rho, kappa, sigma, A0, t0, tau,
    gamma_eau, epaisseur_pml_ratio, puissance_pml, gamma_max_pml
    et soit:
    - x0, y0, capteurs
    ou
    - source_pos_m, capteurs_m
    """
    L = params["L"]
    nx = params["nx"]
    ny = params["ny"]
    dt = params["dt"]
    t_total = params["t_total"]
    rho = params["rho"]
    kappa = params["kappa"]

    sigma = params["sigma"]
    A0 = params["A0"]
    t0 = params["t0"]
    tau = params["tau"]
    gamma_eau = params["gamma_eau"]

    save_frames = params.get("save_frames", True)
    pas_sauvegarde = params.get("pas_sauvegarde", 20)

    x0, y0, capteurs = extraire_source_et_capteurs(params)

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

        if save_frames and n % pas_sauvegarde == 0:
            frames.append(u_np1.copy())
            temps_frames.append(t)

        u_nm1[:, :] = u_n
        u_n[:, :] = u_np1

    return {
        "u_final": u_n.copy(),
        "frames": frames,
        "temps_frames": np.array(temps_frames),
        "c": c,
        "h": h,
        "nt": nt,
        "temps": np.array(temps),
        "signaux_capteurs": [np.array(signal) for signal in signaux_capteurs],
    }


def animer_resultats(frames, temps_frames, L, duree_animation_ms=3000):
    """Animation en kPa."""
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


def animer_resultats_db(frames, temps_frames, L, duree_animation_ms=3000):
    """Animation en dB re 1 µPa à partir de |p|."""
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


def interpole_signal(temps_src, signal_src, temps_ref):
    """Interpole un signal sur la base de temps de référence."""
    return np.interp(temps_ref, temps_src, signal_src)


def cross_correlation_normalisee(signal_ref, signal_test, dt):
    """
    Calcule la cross-correlation normalisée entre deux signaux.
    Retourne:
    - corr_max
    - lag_temps correspondant
    - vecteur complet de corrélation
    """
    a = np.asarray(signal_ref) - np.mean(signal_ref)
    b = np.asarray(signal_test) - np.mean(signal_test)

    norme = np.linalg.norm(a) * np.linalg.norm(b)
    if norme == 0:
        return np.nan, np.nan, None

    corr = np.correlate(a, b, mode="full") / norme

    indice_max = np.argmax(corr)
    corr_max = corr[indice_max]

    lags = np.arange(-len(a) + 1, len(a))
    lag_echantillons = lags[indice_max]
    lag_temps = lag_echantillons * dt

    return corr_max, lag_temps, corr


def analyse_convergence(params_base, liste_dt, liste_nx, nom_fichier="convergence_nuit.pkl"):
    """
    Analyse de convergence:
    - référence = plus petit dt et plus grand nx
    - comparaison par cross-correlation normalisée aux 3 capteurs
    - sauvegarde progressive dans un fichier pickle
    """
    params_base = ajouter_positions_physiques_si_absentes(deepcopy(params_base))

    resultats = {
        "reference": None,
        "tests_dt": [],
        "tests_nx": [],
    }

    dt_ref = min(liste_dt)
    nx_ref = max(liste_nx)

    params_ref = deepcopy(params_base)
    params_ref["dt"] = dt_ref
    params_ref["nx"] = nx_ref
    params_ref["ny"] = nx_ref
    params_ref["save_frames"] = False

    print(f"[REFERENCE] dt = {dt_ref}, nx = ny = {nx_ref}")
    ref = run_simulation(params_ref)

    resultats["reference"] = {
        "params": deepcopy(params_ref),
        "temps": ref["temps"],
        "signaux_capteurs": ref["signaux_capteurs"],
    }

    with open(nom_fichier, "wb") as f:
        pickle.dump(resultats, f)

    temps_ref = ref["temps"]
    signaux_ref = ref["signaux_capteurs"]

    if len(temps_ref) < 2:
        raise ValueError("La simulation de référence ne contient pas assez de points temporels.")

    dt_ref_signal = temps_ref[1] - temps_ref[0]

    # Convergence en dt
    for dt_test in sorted(set(liste_dt), reverse=True):
        params_test = deepcopy(params_base)
        params_test["dt"] = dt_test
        params_test["nx"] = nx_ref
        params_test["ny"] = nx_ref
        params_test["save_frames"] = False

        print(f"[DT] dt = {dt_test}, nx = ny = {nx_ref}")
        test = run_simulation(params_test)

        xcorr_capteurs = []
        lag_temps_capteurs = []
        signaux_interpoles = []

        for i in range(len(signaux_ref)):
            signal_interp = interpole_signal(
                test["temps"],
                test["signaux_capteurs"][i],
                temps_ref
            )
            signaux_interpoles.append(signal_interp)

            xcorr_max, lag_temps, _ = cross_correlation_normalisee(
                signaux_ref[i],
                signal_interp,
                dt_ref_signal
            )

            xcorr_capteurs.append(xcorr_max)
            lag_temps_capteurs.append(lag_temps)

        entree = {
            "params": deepcopy(params_test),
            "temps_bruts": test["temps"],
            "signaux_capteurs_bruts": test["signaux_capteurs"],
            "temps_interpoles": temps_ref.copy(),
            "signaux_capteurs_interpoles": signaux_interpoles,
            "xcorr_max_capteurs": xcorr_capteurs,
            "lag_temps_capteurs": lag_temps_capteurs,
            "xcorr_moyenne": float(np.nanmean(xcorr_capteurs)),
        }

        resultats["tests_dt"].append(entree)

        with open(nom_fichier, "wb") as f:
            pickle.dump(resultats, f)

    # Convergence en nx = ny
    for nx_test in sorted(set(liste_nx)):
        params_test = deepcopy(params_base)
        params_test["dt"] = dt_ref
        params_test["nx"] = nx_test
        params_test["ny"] = nx_test
        params_test["save_frames"] = False

        print(f"[NX] dt = {dt_ref}, nx = ny = {nx_test}")
        test = run_simulation(params_test)

        xcorr_capteurs = []
        lag_temps_capteurs = []
        signaux_interpoles = []
        fenetres_capteurs = []

        for i in range(len(signaux_ref)):
            signal_interp = interpole_signal(
                test["temps"],
                test["signaux_capteurs"][i],
                temps_ref
            )
            signaux_interpoles.append(signal_interp)

            xcorr_max, lag_temps, fenetre, _ = cross_correlation_fenetree(
    signaux_ref[i],
    signal_interp,
    dt_ref_signal,
    frac=0.05,
    marge=200
)                 

            xcorr_capteurs.append(xcorr_max)
            lag_temps_capteurs.append(lag_temps)
            fenetres_capteurs.append(fenetre)

        entree = {
            "params": deepcopy(params_test),
            "temps_bruts": test["temps"],
            "signaux_capteurs_bruts": test["signaux_capteurs"],
            "temps_interpoles": temps_ref.copy(),
            "signaux_capteurs_interpoles": signaux_interpoles,
            "xcorr_max_capteurs": xcorr_capteurs,
            "lag_temps_capteurs": lag_temps_capteurs,
            "xcorr_moyenne": float(np.nanmean(xcorr_capteurs)),
            "fenetres_capteurs": fenetres_capteurs,
        }

        resultats["tests_nx"].append(entree)

        with open(nom_fichier, "wb") as f:
            pickle.dump(resultats, f)

    print(f"Analyse terminée. Résultats sauvegardés dans {nom_fichier}")
    return resultats


def charger_convergence(nom_fichier="convergence_nuit.pkl"):
    """Charge les résultats d'analyse de convergence."""
    with open(nom_fichier, "rb") as f:
        return pickle.load(f)


def tracer_resume_convergence(data):
    """Trace un résumé des cross-correlations moyennes."""
    if len(data["tests_dt"]) > 0:
        dts = [entree["params"]["dt"] for entree in data["tests_dt"]]
        xcorrs_dt = [entree["xcorr_moyenne"] for entree in data["tests_dt"]]

        plt.figure(figsize=(7, 4))
        plt.plot(dts, xcorrs_dt, "o-")
        plt.xlabel("dt [s]")
        plt.ylabel("Cross-correlation moyenne")
        plt.title("Convergence en dt")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if len(data["tests_nx"]) > 0:
        nxs = [entree["params"]["nx"] for entree in data["tests_nx"]]
        xcorrs_nx = [entree["xcorr_moyenne"] for entree in data["tests_nx"]]

        plt.figure(figsize=(7, 4))
        plt.plot(nxs, xcorrs_nx, "o-")
        plt.xlabel("nx = ny")
        plt.ylabel("Cross-correlation moyenne")
        plt.title("Convergence en nx = ny")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def tracer_signaux_convergence(data, groupe="tests_dt", index=0, capteur=0):
    """
    Compare un signal interpolé d'un cas de convergence avec la référence.
    groupe = "tests_dt" ou "tests_nx"
    """
    ref_t = data["reference"]["temps"]
    ref_s = data["reference"]["signaux_capteurs"][capteur]

    entree = data[groupe][index]
    test_t = entree["temps_interpoles"]
    test_s = entree["signaux_capteurs_interpoles"][capteur]

    plt.figure(figsize=(8, 5))
    plt.plot(ref_t, ref_s, label="Référence")
    plt.plot(test_t, test_s, "--", label="Test interpolé")
    plt.xlabel("Temps [s]")
    plt.ylabel("Pression [Pa]")
    plt.title(
        f"{groupe}, cas {index}, capteur {capteur + 1}\n"
        f"xcorr = {entree['xcorr_max_capteurs'][capteur]:.5f}, "
        f"lag = {entree['lag_temps_capteurs'][capteur]:.5e} s"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()
def fenetre_signal_utile(signal_ref, frac=0.05, marge=200):
    """
    Détecte la zone utile du signal à partir de l'amplitude absolue.
    frac = fraction du max utilisée comme seuil
    marge = nombre d'échantillons ajoutés avant/après
    """
    s = np.abs(signal_ref)
    smax = np.max(s)

    if smax == 0:
        return 0, len(signal_ref)

    seuil = frac * smax
    indices = np.where(s >= seuil)[0]

    if len(indices) == 0:
        return 0, len(signal_ref)

    i0 = max(indices[0] - marge, 0)
    i1 = min(indices[-1] + marge + 1, len(signal_ref))

    return i0, i1
def etude_dt(params_base, liste_dt, nom_fichier="etude_dt.pkl"):
    """
    Lance plusieurs simulations pour différents pas de temps
    et sauvegarde les signaux capteurs après chaque run.
    """
    params_base = ajouter_positions_physiques_si_absentes(deepcopy(params_base))

    resultats = []

    for dt in liste_dt:
        params_test = deepcopy(params_base)
        params_test["dt"] = dt
        params_test["save_frames"] = False

        print(f"[DT] dt = {dt}")
        sim = run_simulation(params_test)

        entree = {
            "dt": dt,
            "temps": sim["temps"],
            "signaux_capteurs": sim["signaux_capteurs"],
        }

        resultats.append(entree)

        with open(nom_fichier, "wb") as f:
            pickle.dump(resultats, f)

    print(f"Étude terminée. Résultats sauvegardés dans {nom_fichier}")
    return resultats


def charger_etude_dt(nom_fichier="etude_dt.pkl"):
    with open(nom_fichier, "rb") as f:
        return pickle.load(f)


def tracer_etude_dt(data, capteur=0, normaliser=False, t_min=None, t_max=None):
    """
    Superpose les signaux d'un même capteur pour plusieurs dt.
    """
    plt.figure(figsize=(9, 5))

    for entree in data:
        t = entree["temps"]
        s = entree["signaux_capteurs"][capteur].copy()

        if normaliser:
            smax = np.max(np.abs(s))
            if smax > 0:
                s = s / smax

        if t_min is not None or t_max is not None:
            masque = np.ones_like(t, dtype=bool)
            if t_min is not None:
                masque &= (t >= t_min)
            if t_max is not None:
                masque &= (t <= t_max)

            t = t[masque]
            s = s[masque]

        plt.plot(t, s, label=f"dt = {entree['dt']:.5e} s")

    plt.xlabel("Temps [s]")
    plt.ylabel("Pression [Pa]")
    plt.title(f"Comparaison des signaux — capteur {capteur + 1}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def tracer_etude_dt_3capteurs(data, normaliser=False, t_min=None, t_max=None):
    """
    Trace les 3 capteurs séparément.
    """
    for capteur in range(3):
        tracer_etude_dt(
            data,
            capteur=capteur,
            normaliser=normaliser,
            t_min=t_min,
            t_max=t_max
        )

def cross_correlation_fenetree(signal_ref, signal_test, dt, frac=0.05, marge=200):
    """
    Cross-correlation normalisée sur la fenêtre utile du signal,
    en utilisant l'enveloppe |signal|.
    """
    ref = np.abs(np.asarray(signal_ref))
    test = np.abs(np.asarray(signal_test))

    i0, i1 = fenetre_signal_utile(ref, frac=frac, marge=marge)

    a = ref[i0:i1] - np.mean(ref[i0:i1])
    b = test[i0:i1] - np.mean(test[i0:i1])

    norme = np.linalg.norm(a) * np.linalg.norm(b)
    if norme == 0:
        return np.nan, np.nan, (i0, i1), None

    corr = np.correlate(a, b, mode="full") / norme

    indice_max = np.argmax(corr)
    corr_max = corr[indice_max]

    lags = np.arange(-len(a) + 1, len(a))
    lag_temps = lags[indice_max] * dt

    return corr_max, lag_temps, (i0, i1), corr

# =========================
# PARAMÈTRES DE BASE
# =========================

params = {
    "L": 1000.0,
    "nx": 300,
    "ny": 300,
    "dt": 1e-3,
    "t_total": 1.0,
    "rho": 1000.0,
    "kappa": 2.2e9,

    "x0": 150,
    "y0": 150,
    "sigma": 1.0,
    "A0": 1000.0,
    "t0": 0.02,
    "tau": 0.015,

    "gamma_eau": 0.3,
    "epaisseur_pml_ratio": 0.2,
    "puissance_pml": 0.5,
    "gamma_max_pml": 50.0,

    "capteurs": [
        (175, 175),
        (175, 225),
        (175, 275),
    ],

    "save_frames": True,
    "pas_sauvegarde": 20,
}

# Ajoute automatiquement les positions physiques si absentes.
# C'est important pour l'analyse de convergence quand nx change.
params = ajouter_positions_physiques_si_absentes(params)


# =========================
# CHOIX DU MODE
# =========================
# "simulation"        -> simulation normale
# "etude_dt"          -> lance plusieurs simulations pour différents dt
# "visualisation_dt"  -> recharge et affiche les signaux
MODE = "visualisation_dt"
# MODE = "etude_dt"


if MODE == "simulation":
    resultats = run_simulation(params)

    ani = animer_resultats(
        resultats["frames"],
        resultats["temps_frames"],
        params["L"]
    )

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


elif MODE == "etude_dt":
    liste_dt = np.linspace(1.94625e-3,1.94641e-3,3)
    liste_dt = np.append(liste_dt, 1e-4)

    data = etude_dt(
        params,
        liste_dt=liste_dt,
        nom_fichier="etude_dt.pkl"
    )


elif MODE == "visualisation_dt":
    data = charger_etude_dt("etude_dt.pkl")


    # Signal normalisé pour comparer la forme seulement
    tracer_etude_dt_3capteurs(
        data,
        normaliser=True,
        t_min=0.0,
        t_max=0.8
    )