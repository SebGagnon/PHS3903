import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from sko.PSO import PSO
import pickle
import optuna
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


def source_sonar_multi(nx, ny, t, pulse_data, sigma=1, A0=5e7, tau=0.001):
    """Source totale avec un simple pulse gaussien."""
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    source = np.zeros((nx, ny))

    for x0, y0, t0 in pulse_data:
        r2 = (x - x0) ** 2 + (y - y0) ** 2
        enveloppe_spatiale = np.exp(-r2 / (2 * sigma**2))
        enveloppe_temporelle = np.exp(-((t - t0) ** 2) / (2 * tau**2))
        source += A0 * enveloppe_spatiale * enveloppe_temporelle

    return source


def waveform_source(temps, pulses, A0_source=5e7, tau_source=0.001):
    """Signal temporel total envoyé : somme de pulses gaussiens."""
    signal = np.zeros_like(temps)

    for _, _, t0 in pulses:
        enveloppe_temporelle = np.exp(-((temps - t0) ** 2) / (2 * tau_source**2))
        signal += A0_source * enveloppe_temporelle

    return signal


def laplacien_9_points(u, h):
    """Calcul du Laplacien."""
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
    """Retourne des couleurs bien distinctes pour les capteurs."""
    couleurs_fixes = [
        "red",
        "blue",
        "lime",
        "magenta",
        "orange",
        "cyan",
        "yellow",
        "white",
        "black",
        "purple",
    ]
    return couleurs_fixes[:nb_capteurs]


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


def creer_capteurs_depuis_pulses(pulses):
    """Crée un capteur à la position de chaque pulse."""
    return [(x0, y0) for x0, y0, _ in pulses]


def afficher_waveform_source(t_total, dt, pulses, A0_source=5e7, tau_source=0.001):
    """Affichage du signal temporel total envoyé."""
    temps = np.arange(0, t_total, dt)
    signal = waveform_source(
        temps,
        pulses,
        A0_source=A0_source,
        tau_source=tau_source
    )

    plt.figure(figsize=(8, 4))
    plt.plot(temps, signal)
    plt.title("Pulse envoyé")
    plt.xlabel("Temps [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def animer_resultats(frames, capteurs, L, interval=30):
    """Animation du champ de pression avec échelle de couleurs correcte."""
    fig, ax = plt.subplots(figsize=(7, 6))

    nx_local, ny_local = frames[0].shape
    hx = L / nx_local
    hy = L / ny_local

    amp_max = max(np.max(np.abs(frame)) for frame in frames)
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

    for i, ((xc, yc), couleur) in enumerate(zip(capteurs, couleurs)):
        x_m = (xc + 0.5) * hx
        y_m = (yc + 0.5) * hy

        ax.scatter(x_m, y_m, color=couleur, s=50, marker="o")
        ax.text(
            x_m, y_m, str(i + 1),
            color=couleur,
            fontsize=10,
            ha="left",
            va="bottom",
            weight="bold"
        )

    def maj(k):
        img.set_array(frames[k].T)
        return [img]

    ani = FuncAnimation(fig, maj, frames=len(frames), interval=interval, blit=False)
    plt.tight_layout()
    plt.show()
    return ani


def afficher_signaux_capteurs(temps, signaux_capteurs):
    """Affiche tous les signaux des capteurs sur un seul graphe."""
    plt.figure(figsize=(10, 5))

    couleurs = couleurs_capteurs(len(signaux_capteurs))

    for i, (signal, couleur) in enumerate(zip(signaux_capteurs, couleurs)):
        plt.plot(temps, signal, color=couleur, label=f"Capteur {i+1}")

    plt.title("Pression aux capteurs")
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
    nb_pulses=5,
    t_depart_pulses=0.01,
    dt_pulse=0.03,
    sigma_source=1,
    A0_source=5e7,
    tau_source=0.001,
    nb_frames=300,
    afficher_pml_flag=True,
    afficher_waveform_flag=True,
    afficher_animation=True,
    afficher_capteurs_flag=True,
    seed=None,
):
    """Lance une simulation acoustique avec source impulsionnelle gaussienne."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

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

    pulses = creer_pulses_aleatoires(
        nb_pulses=nb_pulses,
        nx=nx,
        ny=ny,
        epaisseur_pml=epaisseur_pml,
        t_depart=t_depart_pulses,
        dt_pulse=dt_pulse,
    )

    capteurs = creer_capteurs_depuis_pulses(pulses)

    if afficher_waveform_flag:
        afficher_waveform_source(
            t_total=t_total,
            dt=dt,
            pulses=pulses,
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

    if epaisseur_pml > 0:
        zone_physique = (
            slice(epaisseur_pml, nx - epaisseur_pml),
            slice(epaisseur_pml, ny - epaisseur_pml)
        )
    else:
        zone_physique = (slice(None), slice(None))

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

        for i, (xc, yc) in enumerate(capteurs):
            signaux_capteurs[i][n] = u_np1[xc, yc]

        energie_courante = np.sum(u_np1[zone_physique]**2)
        energie_max = max(energie_max, energie_courante)

        if n % pas_sauvegarde == 0:
            frames.append(u_np1.copy())

        u_nm1[:, :] = u_n
        u_n[:, :] = u_np1

    energie_finale = np.sum(u_n[zone_physique]**2)

    if energie_max > 0:
        pourcentage_residuel = 100 * energie_finale / energie_max
    else:
        pourcentage_residuel = None

    if afficher_capteurs_flag:
        afficher_signaux_capteurs(temps, signaux_capteurs)

    if afficher_animation:
        animer_resultats(frames, capteurs, L)

    return {
        "u_final": u_n.copy(),
        "frames": frames,
        "gamma": gamma,
        "pulses": pulses,
        "capteurs": capteurs,
        "temps": temps,
        "signaux_capteurs": signaux_capteurs,
        "c": c,
        "h": h,
        "dt": dt,
        "nt": nt,
        "energie_max": energie_max,
        "energie_finale": energie_finale,
        "pourcentage_residuel": pourcentage_residuel,
        "zone_physique": zone_physique,
    }

nom_fichier_sauvegarde = "pso_pml_sauvegarde.pkl"
historique_evaluations = []
compteur_evaluations = 0


def sauvegarder_etat_pso(nom_fichier, contenu):
    with open(nom_fichier, "wb") as f:
        pickle.dump(contenu, f)

def objectif_pml_moyen(x):
    """
    x = [epaisseur_pml_ratio, gamma_max, puissance_pml]
    Retourne la moyenne du pourcentage résiduel sur plusieurs seeds
    et sauvegarde l'historique des évaluations.
    """
    global historique_evaluations, compteur_evaluations

    epaisseur_ratio, gamma_max_local, puissance_pml_local = x

    seeds = [0, 1, 2, 3, 4]
    scores = []

    for seed in seeds:
        resultats = run_simul(
            L=L,
            t_total=t_total,
            nx=nx,
            ny=ny,
            rho=rho,
            kappa=kappa,
            cfl=cfl,
            epaisseur_pml_ratio=float(epaisseur_ratio),
            puissance_pml=float(puissance_pml_local),
            gamma_max=float(gamma_max_local),
            nb_pulses=nb_pulses,
            t_depart_pulses=t_depart_pulses,
            dt_pulse=dt_pulse,
            sigma_source=sigma_source,
            A0_source=A0_source,
            tau_source=tau_source,
            nb_frames=1,
            afficher_pml_flag=False,
            afficher_waveform_flag=False,
            afficher_animation=False,
            afficher_capteurs_flag=False,
            seed=seed,
        )

        score = resultats["pourcentage_residuel"]

        if score is None or not np.isfinite(score):
            score = 1e12

        scores.append(float(score))

    score_moyen = float(np.mean(scores))
    compteur_evaluations += 1

    entree = {
        "evaluation": compteur_evaluations,
        "timestamp": time.time(),
        "epaisseur_pml_ratio": float(epaisseur_ratio),
        "epaisseur_pml_noeuds": int(float(epaisseur_ratio) * nx),
        "gamma_max": float(gamma_max_local),
        "puissance_pml": float(puissance_pml_local),
        "scores_par_seed": scores,
        "score_moyen": score_moyen,
    }

    historique_evaluations.append(entree)

    # sauvegarde régulière
    if compteur_evaluations % 5 == 0:
        sauvegarder_etat_pso(
            nom_fichier_sauvegarde,
            {
                "historique_evaluations": historique_evaluations,
                "compteur_evaluations": compteur_evaluations,
            }
        )
        print(f"Sauvegarde intermédiaire : {compteur_evaluations} évaluations")

    return score_moyen

def afficher_convergence_pso(historique):
    plt.figure(figsize=(7, 4))
    plt.plot(historique, marker="o")
    plt.xlabel("Itération")
    plt.ylabel("Meilleur pourcentage résiduel moyen [%]")
    plt.title("Convergence du PSO")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Paramètres de simulation

L = 1000
t_total = 0.8
nx = 200
ny = 200

rho = 1000
kappa = 2.2e9
cfl = 0.9

epaisseur_pml_ratio = 0.25
puissance_pml = 1.2
gamma_max = 300

nb_pulses = 5
t_depart_pulses = 0.05
dt_pulse = 0.001

sigma_source = 1
A0_source = 8e7
tau_source = 0.005

nb_frames = 100
seed = None



# Bornes optimisation bayésienne

seed_bo = 123
n_trials = 5

base_nom = f"bo_pml_gp_trials{n_trials}_seed{seed_bo}"
nom_fichier_checkpoint = base_nom + "_checkpoint.pkl"
nom_fichier_sauvegarde = base_nom + "_final.pkl"
nom_etude = base_nom


def objectif_pml_moyen_optuna(trial):
    """
    Version Optuna de la fonction objectif.
    """
    global historique_evaluations, compteur_evaluations

    epaisseur_ratio = trial.suggest_float("epaisseur_pml_ratio", 0.05, 0.25)
    gamma_max_local = trial.suggest_float("gamma_max", 50.0, 10000.0, log=True)
    puissance_pml_local = trial.suggest_float("puissance_pml", 0.5, 6.0)

    seeds = [0, 1, 2, 3, 4,5,6,7,8,9]
    scores = []

    for seed_local in seeds:
        resultats = run_simul(
            L=L,
            t_total=t_total,
            nx=nx,
            ny=ny,
            rho=rho,
            kappa=kappa,
            cfl=cfl,
            epaisseur_pml_ratio=float(epaisseur_ratio),
            puissance_pml=float(puissance_pml_local),
            gamma_max=float(gamma_max_local),
            nb_pulses=nb_pulses,
            t_depart_pulses=t_depart_pulses,
            dt_pulse=dt_pulse,
            sigma_source=sigma_source,
            A0_source=A0_source,
            tau_source=tau_source,
            nb_frames=1,
            afficher_pml_flag=False,
            afficher_waveform_flag=False,
            afficher_animation=False,
            afficher_capteurs_flag=False,
            seed=seed_local,
        )

        score = resultats["pourcentage_residuel"]

        if score is None or not np.isfinite(score):
            score = 1e12

        scores.append(float(score))

    score_moyen = float(np.mean(scores))
    compteur_evaluations += 1

    entree = {
        "evaluation": compteur_evaluations,
        "timestamp": time.time(),
        "epaisseur_pml_ratio": float(epaisseur_ratio),
        "epaisseur_pml_noeuds": int(float(epaisseur_ratio) * nx),
        "gamma_max": float(gamma_max_local),
        "puissance_pml": float(puissance_pml_local),
        "scores_par_seed": scores,
        "score_moyen": score_moyen,
        "trial_number": trial.number,
    }

    historique_evaluations.append(entree)

    if compteur_evaluations % 5 == 0:
        sauvegarder_etat_pso(
            nom_fichier_checkpoint,
            {
                "type": "checkpoint",
                "historique_evaluations": historique_evaluations,
                "compteur_evaluations": compteur_evaluations,
            }
        )
        print(f"Sauvegarde intermédiaire : {compteur_evaluations} évaluations")

    return score_moyen


sampler = optuna.samplers.GPSampler(
    seed=seed_bo,
    n_startup_trials=10,
    deterministic_objective=True,
)

study = optuna.create_study(
    study_name=nom_etude,
    direction="minimize",
    sampler=sampler,
    storage=f"sqlite:///{nom_etude}.db",
    load_if_exists=True,
)

study.optimize(objectif_pml_moyen_optuna, n_trials=n_trials)

best_params = study.best_params
best_value = float(study.best_value)

best_epaisseur_ratio = float(best_params["epaisseur_pml_ratio"])
best_gamma_max = float(best_params["gamma_max"])
best_puissance = float(best_params["puissance_pml"])
best_score = best_value

# historique du meilleur score au fil des essais
gbest_y_hist = []
best_so_far = np.inf
for tr in study.trials:
    if tr.value is not None and np.isfinite(tr.value):
        best_so_far = min(best_so_far, float(tr.value))
        gbest_y_hist.append(best_so_far)

# sauvegarde finale complète
sauvegarder_etat_pso(
    nom_fichier_sauvegarde,
    {
        "type": "final",
        "historique_evaluations": historique_evaluations,
        "compteur_evaluations": compteur_evaluations,
        "best_params": best_params,
        "best_epaisseur_ratio": best_epaisseur_ratio,
        "best_epaisseur_noeuds": int(best_epaisseur_ratio * nx),
        "best_gamma_max": best_gamma_max,
        "best_puissance": best_puissance,
        "best_score": best_score,
        "gbest_y_hist": gbest_y_hist,
        "study_name": nom_etude,
        "storage": f"sqlite:///{nom_etude}.db",
        "sampler": "GPSampler",
        "n_trials": n_trials,
        "seed_bo": seed_bo,
    }
)

print("\nSauvegarde finale écrite dans :", nom_fichier_sauvegarde)

afficher_convergence_pso(gbest_y_hist)


# Validation finale avec affichage

resultats_best = run_simul(
    L=L,
    t_total=t_total,
    nx=nx,
    ny=ny,
    rho=rho,
    kappa=kappa,
    cfl=cfl,
    epaisseur_pml_ratio=best_epaisseur_ratio,
    puissance_pml=best_puissance,
    gamma_max=best_gamma_max,
    nb_pulses=nb_pulses,
    t_depart_pulses=t_depart_pulses,
    dt_pulse=dt_pulse,
    sigma_source=sigma_source,
    A0_source=A0_source,
    tau_source=tau_source,
    nb_frames=nb_frames,
    afficher_pml_flag=True,
    afficher_waveform_flag=True,
    afficher_animation=True,
    afficher_capteurs_flag=True,
    seed=0,
)

print("Pourcentage résiduel validation =", resultats_best["pourcentage_residuel"])