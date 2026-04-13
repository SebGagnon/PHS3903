import random
import numpy as np
import matplotlib.pyplot as plt

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import pickle

def creer_pml(nx, ny, epaisseur, puissance, gamma_max):
    '''Crée la couche absorbante près des bords.'''
    gamma = np.zeros((nx, ny))
    lignes, colonnes = np.indices((nx, ny))
    dist_bord = np.minimum.reduce([
        lignes,
        colonnes,
        nx - 1 - lignes,
        ny - 1 - colonnes,
    ])

    masque = dist_bord < epaisseur
    profil = (epaisseur - dist_bord[masque]) / epaisseur
    gamma[masque] = gamma_max * profil ** puissance
    return gamma


def laplacien_9_points(u, h):
    '''Calcule le laplacien 2D avec un stencil à 9 points.'''
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
    ) / (6 * h ** 2)
    return lap


def impose_bords_dirichlet(u):
    '''Impose u = 0 sur les frontières.'''
    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0
    return u


def position_capteur_m(capteur, h=None):
    '''Retourne la position d un capteur en mètres.'''
    if 'position_m' in capteur:
        return capteur['position_m']

    if h is None:
        raise ValueError('h est requis pour convertir une ancienne position en mètres.')

    ix, iy = capteur['position']
    return (ix + 0.5) * h, (iy + 0.5) * h


def position_capteur_noeud(capteur, h, nx, ny):
    '''Retourne l indice du noeud le plus proche du capteur.'''
    if 'position' in capteur and 'position_m' not in capteur:
        return capteur['position']

    x_m, y_m = position_capteur_m(capteur)
    ix = int(round(x_m / h - 0.5))
    iy = int(round(y_m / h - 0.5))
    ix = int(np.clip(ix, 0, nx - 1))
    iy = int(np.clip(iy, 0, ny - 1))
    return ix, iy


def pulses_depuis_capteurs(capteurs, h=None):
    '''Transforme les émissions des capteurs en liste de pulses en mètres.'''
    pulses = []
    for capteur in capteurs:
        x_m, y_m = position_capteur_m(capteur, h=h)
        for t0 in capteur['emissions']:
            pulses.append((x_m, y_m, t0))
    return pulses


def source_sonar(x_grid, y_grid, t, pulses, sigma=1, a0=5e7, tau=0.001):
    '''Construit la source gaussienne totale dans l espace et le temps.'''
    source = np.zeros_like(x_grid)

    for x0_m, y0_m, t0 in pulses:
        r2 = (x_grid - x0_m) ** 2 + (y_grid - y0_m) ** 2
        env_spatiale = np.exp(-r2 / (2 * sigma ** 2))
        env_temporelle = np.exp(-((t - t0) ** 2) / (2 * tau ** 2))
        source += a0 * env_spatiale * env_temporelle

    return source


def creer_maillage_physique(nx, ny, h):

    '''Crée le maillage en mètres au centre des cellules.'''
    x = (np.arange(nx) + 0.5) * h
    y = (np.arange(ny) + 0.5) * h
    return np.meshgrid(x, y, indexing='ij')

def run_simul(
        
    taille=1000,
    t_total=1.2,
    nx=250,
    ny=250,
    rho=1000,
    kappa=2.2e9,
    cfl=0.4,
    ratio_pml=0.2,
    puissance_pml=3.0,
    gamma_max=20000.0,
    capteurs=None,
    sigma_source=8.0,
    a0_source=5e7,
    tau_source=0.003,
    seed=None,
):
    '''Simulation acoustique 2D minimale pour tester le PML.'''
    if seed is not None:
        random.seed(seed)

    if capteurs is None:
        raise ValueError('capteurs doit être fourni.')

    h = taille / nx
    c_eau = np.sqrt(kappa / rho)
    dt = cfl * h / (c_eau * np.sqrt(2))
    nt = int(t_total / dt)

    epaisseur_pml = int(ratio_pml * nx)
    gamma_pml = creer_pml(nx, ny, epaisseur_pml, puissance_pml, gamma_max)

    pulses = pulses_depuis_capteurs(capteurs, h=h)
    x_grid, y_grid = creer_maillage_physique(nx, ny, h)

    u_prev = np.zeros((nx, ny))
    u_now = np.zeros((nx, ny))
    u_next = np.zeros((nx, ny))

    temps = np.arange(nt) * dt
    signaux_capteurs = [np.zeros(nt) for _ in capteurs]

    energie_max = 0.0

    for n in range(nt):
        t_n = n * dt

        lap = laplacien_9_points(u_now, h)
        a = gamma_pml * dt / 2

        source = source_sonar(
            x_grid,
            y_grid,
            t_n,
            pulses,
            sigma=sigma_source,
            a0=a0_source,
            tau=tau_source,
        )

        u_next[1:-1, 1:-1] = (
            2 * u_now[1:-1, 1:-1]
            - (1 - a[1:-1, 1:-1]) * u_prev[1:-1, 1:-1]
            + dt ** 2 * c_eau ** 2 * lap[1:-1, 1:-1]
            + dt ** 2 * source[1:-1, 1:-1]
        ) / (1 + a[1:-1, 1:-1])

        impose_bords_dirichlet(u_next)

        for i, capteur in enumerate(capteurs):
            ix, iy = position_capteur_noeud(capteur, h, nx, ny)
            signaux_capteurs[i][n] = u_next[ix, iy]

        energie = np.sum(u_next ** 2)
        energie_max = max(energie_max, energie)

        u_prev[:, :] = u_now
        u_now[:, :] = u_next

    energie_finale = np.sum(u_now ** 2)
    pourcentage_residuel = None if energie_max == 0 else 100 * energie_finale / energie_max

    return {

        'u_final': u_now.copy(),
        'gamma_pml': gamma_pml,
        'capteurs': capteurs,
        'pulses': pulses,
        'temps': temps,
        'signaux_capteurs': signaux_capteurs,
        'c_eau': c_eau,
        'dt': dt,
        'nt': nt,
        'h': h,
        'energie_max': energie_max,
        'energie_finale': energie_finale,
        'pourcentage_residuel': pourcentage_residuel,
    }

def creer_capteur_central(taille, t_emit=0.01):
    '''Capteur unique au centre, qui émet aussi.'''
    centre = taille / 2
    return [{
        'nom': 'Capteur centre',
        'position_m': (centre, centre),
        'emissions': [t_emit],
    }]

def mesure_residuelle_pml(params, gamma_max=None, puissance_pml=None, show=False):
    '''Mesure l efficacité du PML avec l énergie résiduelle globale.'''
    cfg = dict(params)

    if gamma_max is not None:
        cfg['gamma_max'] = gamma_max
    if puissance_pml is not None:
        cfg['puissance_pml'] = puissance_pml

    # une source fixe, sans dépendre d une mesure au capteur
    t_emit = cfg.pop('t_emit', 0.01)
    cfg['capteurs'] = creer_capteur_central(cfg['taille'], t_emit=t_emit)

    res = run_simul(**cfg)

    score = res['pourcentage_residuel']

    if score is None:
        score = np.inf

    if show:
        print(f"gamma_max         = {cfg['gamma_max']:.6e}")
        print(f"puissance_pml     = {cfg['puissance_pml']:.6f}")
        print(f"energie_max       = {res['energie_max']:.6e}")
        print(f"energie_finale    = {res['energie_finale']:.6e}")
        print(f"pourcentage_residuel = {score:.6e} %")

    return {
        'score': score,
        'energie_max': res['energie_max'],
        'energie_finale': res['energie_finale'],
        'pourcentage_residuel': score,
        'res': res,
    }
def mesure_reflexion_pml(params, gamma_max=None, puissance_pml=None, show=False):
    '''Mesure la réflexion du PML avec une fenêtre centrée sur le pic réel de retour.'''
    cfg = dict(params)

    if gamma_max is not None:
        cfg['gamma_max'] = gamma_max
    if puissance_pml is not None:
        cfg['puissance_pml'] = puissance_pml

    t_emit = cfg.pop('t_emit', 0.01)
    cfg['capteurs'] = creer_capteur_central(cfg['taille'], t_emit=t_emit)

    res = run_simul(**cfg)

    temps = res['temps']
    signal = res['signaux_capteurs'][0]
    c_eau = res['c_eau']

    taille = cfg['taille']
    ratio_pml = cfg['ratio_pml']
    tau_source = cfg['tau_source']

    dist_debut_pml = taille / 2 - ratio_pml * taille
    dist_fin_pml = taille / 2

    # fenêtre incidente autour de l émission
    t0_inc = t_emit - 3 * tau_source
    t1_inc = t_emit + 3 * tau_source

    # grande fenêtre où la réflexion du PML peut physiquement revenir
    t0_search = t_emit + 2 * dist_debut_pml / c_eau
    t1_search = t_emit + 2 * dist_fin_pml / c_eau + 4 * tau_source

    masque_search = (temps >= t0_search) & (temps <= t1_search)
    if not np.any(masque_search):
        raise ValueError('t_total trop petit pour chercher la réflexion du PML.')

    # pic réel de retour dans la fenêtre de recherche
    idx_search = np.where(masque_search)[0]
    idx_peak_local = np.argmax(np.abs(signal[masque_search]))
    idx_peak = idx_search[idx_peak_local]
    t_peak = temps[idx_peak]

    # petite fenêtre centrée sur le pic réel
    demi_largeur = 4 * tau_source
    t0_ref = t_peak - demi_largeur
    t1_ref = t_peak + demi_largeur

    masque_inc = (temps >= t0_inc) & (temps <= t1_inc)
    masque_ref = (temps >= t0_ref) & (temps <= t1_ref)

    if not np.any(masque_inc):
        raise ValueError('Fenêtre incidente vide.')
    if not np.any(masque_ref):
        raise ValueError('Fenêtre réflexion vide.')

    energie_incidente = np.sum(signal[masque_inc] ** 2)
    energie_reflechie = np.sum(signal[masque_ref] ** 2)

    if energie_incidente == 0:
        score = np.inf
    else:
        score = energie_reflechie / energie_incidente

    if show:
        plt.figure(figsize=(10, 4))
        plt.plot(temps, signal, label='signal au centre')

        plt.axvline(t_emit, color='black', linestyle='--', label='émission')
        plt.axvline(t0_inc, color='green', linestyle='--', label='début incident')
        plt.axvline(t1_inc, color='green', linestyle=':')

        plt.axvline(t0_search, color='orange', linestyle='--', label='début recherche')
        plt.axvline(t1_search, color='orange', linestyle=':')

        plt.axvline(t_peak, color='purple', linestyle='--', label='pic retour')
        plt.axvline(t0_ref, color='red', linestyle='--', label='début fenêtre réflexion')
        plt.axvline(t1_ref, color='red', linestyle=':')

        plt.xlabel('Temps [s]')
        plt.ylabel('Pression')
        plt.title(
            f'Score réflexion/incidente = {score:.3e}\n'
            f'E_ref={energie_reflechie:.3e}, E_inc={energie_incidente:.3e}'
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        print(f't_peak = {t_peak:.6e}')
        print(f'energie_incidente = {energie_incidente:.6e}')
        print(f'energie_reflechie = {energie_reflechie:.6e}')
        print(f'score = {score:.6e}')

    return {
        'score': score,
        'energie_incidente': energie_incidente,
        'energie_reflechie': energie_reflechie,
        't_peak': t_peak,
        't0_inc': t0_inc,
        't1_inc': t1_inc,
        't0_ref': t0_ref,
        't1_ref': t1_ref,
        'temps': temps,
        'signal': signal,
        'res': res,
    }
def optimise_pml(
    params,
    n_calls=25,
    n_initial_points=8,
    random_state=42,
    gamma_bounds=(1e2, 1e6),
    puissance_bounds=(0.5, 8.0),
    save_path='bo_pml_result.pkl',
):
    '''Optimise gamma_max et puissance_pml avec BO sur l énergie résiduelle.'''
    espace = [
        Real(gamma_bounds[0], gamma_bounds[1], prior='log-uniform', name='gamma_max'),
        Real(puissance_bounds[0], puissance_bounds[1], prior='uniform', name='puissance_pml'),
    ]

    historique = []

    @use_named_args(espace)
    def objectif(gamma_max, puissance_pml):
        out = mesure_residuelle_pml(
            params,
            gamma_max=gamma_max,
            puissance_pml=puissance_pml,
            show=False,
        )
        score = out['score']

        entree = {
            'gamma_max': gamma_max,
            'puissance_pml': puissance_pml,
            'score': score,
        }
        historique.append(entree)

        with open(save_path, 'wb') as f:
            pickle.dump(
                {
                    'type': 'checkpoint',
                    'metric': 'pourcentage_residuel',
                    'params': params,
                    'gamma_bounds': gamma_bounds,
                    'puissance_bounds': puissance_bounds,
                    'historique': historique,
                },
                f,
            )

        print(
            f'gamma_max={gamma_max:.3e} | '
            f'puissance_pml={puissance_pml:.3f} | '
            f'pourcentage_residuel={score:.6e} %'
        )
        return score

    res_bo = gp_minimize(
        func=objectif,
        dimensions=espace,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=random_state,
        acq_func='EI',
    )

    with open(save_path, 'wb') as f:
        pickle.dump(
            {
                'type': 'final',
                'metric': 'pourcentage_residuel',
                'params': params,
                'gamma_bounds': gamma_bounds,
                'puissance_bounds': puissance_bounds,
                'historique': historique,
                'best_gamma_max': res_bo.x[0],
                'best_puissance_pml': res_bo.x[1],
                'best_score': res_bo.fun,
                'x_iters': res_bo.x_iters,
                'func_vals': res_bo.func_vals,
            },
            f,
        )

    return res_bo
def main():
    params = {
        'taille': 1000,
        't_total': 1.5,
        'nx': 300,
        'ny': 300,
        'rho': 1000,
        'kappa': 2.2e9,
        'cfl': 0.4,
        'ratio_pml': 0.2,
        'puissance_pml': 10,
        'gamma_max': 10000,
        'sigma_source': 2.1,
        'a0_source': 1,
        'tau_source': 0.004,
        'seed': 0,
        't_emit': 0.02,
    }

    mesure0 = mesure_residuelle_pml(params, show=True)
    print(f'\nScore initial : {mesure0["score"]:.6e}')

    res_bo = optimise_pml(
    params,
    n_calls=150,
    n_initial_points=20,
    random_state=42,
    gamma_bounds=(0.01, 1e5),
    puissance_bounds=(0.1, 6.0),
    save_path='bo_pml_test.pkl',
)

    print('\nMeilleur résultat trouvé')
    print(f'gamma_max     = {res_bo.x[0]:.6e}')
    print(f'puissance_pml = {res_bo.x[1]:.6f}')
    print(f'score         = {res_bo.fun:.6e}')

    plt.figure(figsize=(8, 4))
    plt.plot(res_bo.func_vals, 'o-', label='score évalué')
    plt.plot(np.minimum.accumulate(res_bo.func_vals), '-', label='meilleur cumulatif')
    plt.xlabel('Itération')
    plt.ylabel('Score réflexion')
    plt.title('Optimisation bayésienne du PML')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()