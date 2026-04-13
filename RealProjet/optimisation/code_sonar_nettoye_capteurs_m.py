import random
from matplotlib.colors import SymLogNorm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
from scipy.optimize import least_squares
from scipy.signal import hilbert


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


def couleurs_capteurs(nb_capteurs):
    '''Retourne des couleurs distinctes pour les capteurs.'''
    cmap = plt.get_cmap('tab10')
    return [cmap(i % 10) for i in range(nb_capteurs)]


def affiche_pml(gamma, taille):
    '''Affiche la carte du PML.'''
    plt.figure(figsize=(6, 5))
    plt.imshow(gamma.T, origin='lower', cmap='inferno', extent=[0, taille, 0, taille])
    plt.colorbar(label=r'$\gamma(x,y)$')
    plt.title('Couche absorbante')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.tight_layout()
    plt.show()


def creer_capteurs_aleatoires(nb_capteurs, nx, ny, epaisseur_pml, taille):
    '''Crée des capteurs aléatoires en mètres hors du PML.'''
    h = taille / nx
    capteurs = []
    deja_pris = set()
    marge = epaisseur_pml + 2

    while len(capteurs) < nb_capteurs:
        ix = random.randint(marge, nx - marge - 1)
        iy = random.randint(marge, ny - marge - 1)

        if (ix, iy) in deja_pris:
            continue

        deja_pris.add((ix, iy))
        capteurs.append({
            'nom': f'Capteur {len(capteurs) + 1}',
            'position_m': ((ix + 0.5) * h, (iy + 0.5) * h),
            'emissions': [],
        })

    return capteurs


def ajoute_emissions_defaut(capteurs, t_depart=0.01, dt_pulse=0.03):
    '''Ajoute un pulse par capteur avec un décalage temporel.'''
    for i, capteur in enumerate(capteurs):
        capteur['emissions'] = [t_depart + i * dt_pulse]


def valide_capteurs(capteurs):
    '''Vérifie la structure minimale des capteurs.'''
    if not isinstance(capteurs, list):
        raise ValueError('capteurs doit être une liste.')

    for i, capteur in enumerate(capteurs):
        if not isinstance(capteur, dict):
            raise ValueError(f'Le capteur {i} doit être un dictionnaire.')

        capteur.setdefault('nom', f'Capteur {i + 1}')
        capteur.setdefault('emissions', [])

        if 'position_m' not in capteur and 'position' not in capteur:
            raise ValueError(f'Le capteur {i} doit avoir une position.')

        if 'position_m' in capteur:
            pos = capteur['position_m']
            if not (isinstance(pos, tuple) and len(pos) == 2):
                raise ValueError(f'La position_m du capteur {i} doit être un tuple (x_m, y_m).')
        else:
            pos = capteur['position']
            if not (isinstance(pos, tuple) and len(pos) == 2):
                raise ValueError(f'La position du capteur {i} doit être un tuple (ix, iy).')

        if not isinstance(capteur['emissions'], list):
            raise ValueError(f'Les emissions du capteur {i} doivent être une liste.')


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


def signaux_sources(temps, capteurs, a0=5e7, tau=0.001):
    '''Retourne le signal émis par chaque capteur.'''
    signaux = {}

    for capteur in capteurs:
        signal = np.zeros_like(temps)
        for t0 in capteur['emissions']:
            signal += a0 * np.exp(-((temps - t0) ** 2) / (2 * tau ** 2))
        signaux[capteur['nom']] = signal

    return signaux


def affiche_signaux_sources(t_total, dt, capteurs, a0=5e7, tau=0.001):
    '''Affiche les pulses envoyés par les capteurs.'''
    temps = np.arange(0, t_total, dt)
    signaux = signaux_sources(temps, capteurs, a0=a0, tau=tau)
    couleurs = couleurs_capteurs(len(capteurs))

    plt.figure(figsize=(10, 5))

    for capteur, couleur in zip(capteurs, couleurs):
        nom = capteur['nom']
        if capteur['emissions']:
            plt.plot(temps, signaux[nom], color=couleur, label=nom)
        else:
            plt.plot([], [], color=couleur, label=f'{nom} (silencieux)')

    plt.title('Pulses envoyés par capteur')
    plt.xlabel('Temps [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def affiche_signaux_capteurs(temps, signaux_capteurs, capteurs):
    '''Affiche la pression mesurée à chaque capteur.'''
    couleurs = couleurs_capteurs(len(capteurs))
    plt.figure(figsize=(10, 5))

    for capteur, signal, couleur in zip(capteurs, signaux_capteurs, couleurs):
        plt.plot(temps, signal, color=couleur, label=capteur['nom'])

    plt.title('Pression mesurée aux capteurs')
    plt.xlabel('Temps [s]')
    plt.ylabel('Pression')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def creer_maillage_physique(nx, ny, h):
    '''Crée le maillage en mètres au centre des cellules.'''
    x = (np.arange(nx) + 0.5) * h
    y = (np.arange(ny) + 0.5) * h
    return np.meshgrid(x, y, indexing='ij')


def valide_objet(objet):
    '''Vérifie la structure minimale de l objet mobile.'''
    if objet is None:
        return

    if 'forme' not in objet:
        raise ValueError('L\'objet doit contenir forme.')
    if 'centre_init_m' not in objet:
        raise ValueError('L\'objet doit contenir centre_init_m.')
    if 'c_objet' not in objet:
        raise ValueError('L\'objet doit contenir c_objet.')

    objet.setdefault('vitesse_m_s', (0.0, 0.0))
    objet.setdefault('gamma_objet', 0.0)
    objet.setdefault('actif', True)

    if objet['forme'] == 'cercle' and 'rayon_m' not in objet:
        raise ValueError('Un objet cercle doit contenir rayon_m.')
    if objet['forme'] == 'rectangle' and 'taille_m' not in objet:
        raise ValueError('Un objet rectangle doit contenir taille_m.')
    if objet['forme'] not in {'cercle', 'rectangle'}:
        raise ValueError('forme doit être cercle ou rectangle.')


def position_objet(objet, t):
    '''Retourne la position du centre de l objet à l instant t.'''
    x0, y0 = objet['centre_init_m']
    vx, vy = objet.get('vitesse_m_s', (0.0, 0.0))
    return x0 + vx * t, y0 + vy * t


def champs_objet(x_grid, y_grid, c_eau, t, objet=None):
    '''Construit c_grid et gamma_grid pour l objet mobile.'''
    c_grid = c_eau * np.ones_like(x_grid)
    gamma_grid = np.zeros_like(x_grid)
    centre = None

    if objet is None or not objet.get('actif', True):
        return c_grid, gamma_grid, centre

    xc, yc = position_objet(objet, t)
    centre = (xc, yc)

    if objet['forme'] == 'cercle':
        rayon = objet['rayon_m']
        masque = (x_grid - xc) ** 2 + (y_grid - yc) ** 2 <= rayon ** 2
    else:
        lx, ly = objet['taille_m']
        masque = (np.abs(x_grid - xc) <= lx / 2) & (np.abs(y_grid - yc) <= ly / 2)

    c_grid[masque] = objet['c_objet']
    gamma_grid[masque] = objet['gamma_objet']
    return c_grid, gamma_grid, centre


def ajoute_patch_objet(ax, objet):
    '''Dessine l objet initial sur l axe.'''
    if objet is None or not objet.get('actif', True):
        return None

    xc, yc = objet['centre_init_m']

    if objet['forme'] == 'cercle':
        patch = Circle((xc, yc), objet['rayon_m'], fill=False, edgecolor='black', linewidth=2)
    else:
        lx, ly = objet['taille_m']
        patch = Rectangle((xc - lx / 2, yc - ly / 2), lx, ly, fill=False, edgecolor='black', linewidth=2)

    ax.add_patch(patch)
    return patch


def anime_resultats(frames, capteurs, taille, signaux_capteurs, objet=None, trajectoire=None, interval=30):
    '''Anime le champ de pression.'''
    fig, ax = plt.subplots(figsize=(7, 6))

    nx, ny = frames[0].shape
    h = taille / nx

    if signaux_capteurs:
        amp_max = 0.2 * max(np.max(np.abs(signal)) for signal in signaux_capteurs)
    else:
        amp_max = max(np.max(np.abs(frame)) for frame in frames)

    if amp_max == 0:
        amp_max = 1.0

    alpha = 1.2   

    def transforme_pression(u):
        u_norm = u / amp_max
        u_aff = np.sign(u_norm) * np.abs(u_norm) ** alpha
        return np.clip(u_aff, -1, 1)

    img = ax.imshow(
        transforme_pression(frames[0]).T,
        origin='lower',
        cmap='seismic',
        extent=[0, taille, 0, taille],
        vmin=-1,
        vmax=1,
    )

    plt.colorbar(img, ax=ax, label='Pression transformée')
    ax.set_title('Champ de pression')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    for i, (capteur, couleur) in enumerate(zip(capteurs, couleurs_capteurs(len(capteurs)))):
        x_m, y_m = position_capteur_m(capteur, h=h)
        ax.scatter(x_m, y_m, color=couleur, s=50, marker='o')
        ax.text(
            x_m, y_m, str(i + 1),
            color=couleur, fontsize=10,
            ha='left', va='bottom', weight='bold'
        )

    patch_objet = ajoute_patch_objet(ax, objet)

    def maj(i):
        artistes = [img]
        img.set_array(transforme_pression(frames[i]).T)

        if patch_objet is not None and trajectoire is not None and i < len(trajectoire):
            centre = trajectoire[i]
            if centre is not None:
                xc, yc = centre
                if objet['forme'] == 'cercle':
                    patch_objet.center = (xc, yc)
                else:
                    lx, ly = objet['taille_m']
                    patch_objet.set_xy((xc - lx / 2, yc - ly / 2))
                artistes.append(patch_objet)

        return artistes

    ani = FuncAnimation(fig, maj, frames=len(frames), interval=interval, blit=False)
    plt.tight_layout()
    plt.show()
    return ani

def run_simul(
    taille=100,
    t_total=0.2,
    nx=500,
    ny=500,
    rho=1000,
    kappa=2.2e9,
    cfl=0.6,
    ratio_pml=0.3,
    puissance_pml=3,
    gamma_max=20000,
    nb_capteurs=5,
    capteurs=None,
    emission_defaut=True,
    t_depart_pulses=0.01,
    dt_pulse=0.03,
    sigma_source=1,
    a0_source=5e7,
    tau_source=0.001,
    objet=None,
    nb_frames=300,
    afficher_pml=False,
    afficher_sources=True,
    afficher_animation=True,
    afficher_capteurs=True,
    seed=None,
):
    '''Lance la simulation acoustique 2D.'''
    if seed is not None:
        random.seed(seed)

    valide_objet(objet)

    h = taille / nx
    c_eau = np.sqrt(kappa / rho)
    c_max = max(c_eau, objet['c_objet']) if objet is not None else c_eau

    dt = cfl * h / (c_max * np.sqrt(2))
    nt = int(t_total / dt)
    pas_frame = max(1, nt // nb_frames)

    epaisseur_pml = int(ratio_pml * nx)
    gamma_pml = creer_pml(nx, ny, epaisseur_pml, puissance_pml, gamma_max)

    if afficher_pml:
        affiche_pml(gamma_pml, taille)

    if capteurs is None:
        capteurs = creer_capteurs_aleatoires(nb_capteurs, nx, ny, epaisseur_pml, taille)
        if emission_defaut:
            ajoute_emissions_defaut(capteurs, t_depart=t_depart_pulses, dt_pulse=dt_pulse)

    valide_capteurs(capteurs)
    pulses = pulses_depuis_capteurs(capteurs, h=h)

    if afficher_sources:
        affiche_signaux_sources(t_total, dt, capteurs, a0=a0_source, tau=tau_source)

    x_grid, y_grid = creer_maillage_physique(nx, ny, h)

    u_prev = np.zeros((nx, ny))
    u_now = np.zeros((nx, ny))
    u_next = np.zeros((nx, ny))

    frames = [u_now.copy()]
    trajectoire = []
    _, _, centre_init = champs_objet(x_grid, y_grid, c_eau, 0.0, objet)
    trajectoire.append(centre_init)

    energie_max = 0.0
    temps = np.arange(nt) * dt
    signaux_capteurs = [np.zeros(nt) for _ in capteurs]

    for n in range(nt):
        t_n = n * dt

        c_grid, gamma_objet, centre_objet = champs_objet(x_grid, y_grid, c_eau, t_n, objet)
        gamma_tot = gamma_pml + gamma_objet

        lap = laplacien_9_points(u_now, h)
        a = gamma_tot * dt / 2

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
            + dt ** 2 * c_grid[1:-1, 1:-1] ** 2 * lap[1:-1, 1:-1]
            + dt ** 2 * source[1:-1, 1:-1]
        ) / (1 + a[1:-1, 1:-1])

        impose_bords_dirichlet(u_next)

        for i, capteur in enumerate(capteurs):
            ix, iy = position_capteur_noeud(capteur, h, nx, ny)
            signaux_capteurs[i][n] = u_next[ix, iy]

        energie = np.sum(u_next ** 2)
        energie_max = max(energie_max, energie)

        if n % pas_frame == 0:
            frames.append(u_next.copy())
            trajectoire.append(centre_objet)

        u_prev[:, :] = u_now
        u_now[:, :] = u_next

    energie_finale = np.sum(u_now ** 2)
    pourcentage_residuel = None if energie_max == 0 else 100 * energie_finale / energie_max

    if afficher_capteurs:
        affiche_signaux_capteurs(temps, signaux_capteurs, capteurs)

    animation = None
    if afficher_animation:
        animation = anime_resultats(
            frames,
            capteurs,
            taille,
            signaux_capteurs,
            objet=objet,
            trajectoire=trajectoire,
        )

    return {
        'u_final': u_now.copy(),
        'frames': frames,
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
        'objet': objet,
        'trajectoire_objet': trajectoire,
        'animation': animation,
    }


def plages_pulses(capteurs, t_total):
    '''Construit les plages temporelles à partir des émissions.'''
    pulses = []

    for idx_capteur, capteur in enumerate(capteurs):
        for emission in capteur.get('emissions', []):
            if isinstance(emission, dict):
                t0 = emission['t0']
            else:
                t0 = float(emission)
            pulses.append((idx_capteur, t0))

    if not pulses:
        print('⚠ Aucun pulse détecté dans les capteurs.')
        return []

    pulses.sort(key=lambda x: x[1])
    temps_emission = [pulse[1] for pulse in pulses]
    plages = []

    for i, (idx_capteur, t0) in enumerate(pulses):
        t_fin = temps_emission[i + 1] if i + 1 < len(pulses) else t_total
        plages.append((idx_capteur, t0, t_fin))

    return plages


def pic_enveloppe_plage(signal, temps, t_debut, t_fin, seuil_montee=1):
    '''Trouve un pic d enveloppe dans une plage donnée.'''
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


def affiche_echos(intensite_echo, temps, capteurs, t_total, afficher=False):
    '''Affiche les échos capteur par capteur dans chaque plage.'''
    if not afficher:
        return

    plages = plages_pulses(capteurs, t_total)
    if not plages:
        return

    n_pulses = len(plages)
    couleurs = couleurs_capteurs(len(capteurs))
    labels = [f'Capteur {i + 1}' for i in range(len(capteurs))]

    fig, axes = plt.subplots(n_pulses, 1, figsize=(10, 4 * n_pulses), sharex=False)
    if n_pulses == 1:
        axes = [axes]

    for k, (idx_emit, t_debut, t_fin) in enumerate(plages):
        ax = axes[k]
        masque = (temps >= t_debut) & (temps < t_fin)

        for i, signal in enumerate(intensite_echo):
            ax.plot(temps[masque], signal[masque], color=couleurs[i], alpha=0.4)

            t_max, amp_max, enveloppe = pic_enveloppe_plage(signal, temps, t_debut, t_fin)
            ax.plot(temps[masque], enveloppe[masque], color=couleurs[i], linestyle='--', label=labels[i])

            if t_max is not None:
                ax.scatter(t_max, amp_max, color=couleurs[i], s=80, zorder=5)
                ax.axvline(t_max, color=couleurs[i], linestyle=':', alpha=0.7)

        ax.axvline(t_debut, color='black', linestyle='-', linewidth=1.5, label=f'Émission C{idx_emit + 1} (t={t_debut:.2f}s)')
        ax.set_xlim(t_debut, t_fin)
        ax.set_xlabel('Temps [s]')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Pulse {k + 1} — Émetteur C{idx_emit + 1} | [{t_debut:.2f}s, {t_fin:.2f}s]')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.4)

    plt.suptitle('Échos par plage de pulse', fontsize=13)
    plt.tight_layout()
    plt.show()


def positions_capteurs_m(capteurs, h=None):
    '''Retourne les positions des capteurs en mètres.'''
    return np.array([position_capteur_m(capteur, h=h) for capteur in capteurs])


def multilateration_un_pulse(intensite_echos, temps, capteurs, c_eau, h=None, idx_emetteur=0, t_debut=None, t_fin=None):
    '''Estime une position à partir d un seul pulse.'''
    intensite_echos = np.array(intensite_echos)
    n_capteurs = len(capteurs)
    pos_capteurs = positions_capteurs_m(capteurs, h=h)

    if t_debut is None or t_fin is None:
        raise ValueError('t_debut et t_fin doivent être fournis.')

    temps_echo = []
    for i, signal in enumerate(intensite_echos):
        t_max, _, _ = pic_enveloppe_plage(signal, temps, t_debut, t_fin)
        if t_max is None:
            raise ValueError(f'Pas de pic détecté pour capteur {i + 1}')
        temps_echo.append(t_max)

    temps_echo = np.array(temps_echo)
    d_emit = c_eau * (temps_echo[idx_emetteur] - t_debut) / 2

    indices_recepteurs = [i for i in range(n_capteurs) if i != idx_emetteur]
    d_recepteurs = {
        i: c_eau * (temps_echo[i] - t_debut) - d_emit
        for i in indices_recepteurs
    }

    distances = np.zeros(n_capteurs)
    distances[idx_emetteur] = d_emit
    for i in indices_recepteurs:
        distances[i] = d_recepteurs[i]

    def residus(x):
        res = [np.linalg.norm(x - pos_capteurs[idx_emetteur]) - d_emit]
        for i in indices_recepteurs:
            res.append(np.linalg.norm(x - pos_capteurs[i]) - d_recepteurs[i])
        return res

    x0 = np.mean(pos_capteurs, axis=0)
    sol = least_squares(residus, x0)
    return sol.x, temps_echo, distances, idx_emetteur


def position_reelle_impact(objet, t0_emit, d_emetteur, c_eau):
    '''Calcule la position réelle de l objet au moment de l impact.'''
    vx, vy = objet['vitesse_m_s']
    x0, y0 = objet['centre_init_m']
    t_impact = t0_emit + d_emetteur / c_eau
    return np.array([x0 + vx * t_impact, y0 + vy * t_impact]), t_impact


def trace_localisation_echo(capteurs, pos_estimee, objet, distances, t0_emit, d_emetteur, c_eau, taille, h, afficher=True):
    '''Trace la localisation issue d un pulse.'''
    pos_reelle, t_impact = position_reelle_impact(objet, t0_emit, d_emetteur, c_eau)

    if not afficher:
        return pos_reelle, t_impact

    fig, ax = plt.subplots(figsize=(8, 8))
    couleurs = couleurs_capteurs(len(capteurs))

    x0_m, y0_m = objet['centre_init_m']
    x_i, y_i = pos_reelle

    if objet['forme'] == 'cercle':
        rayon = objet.get('rayon_m', 5.0)
        ax.add_patch(plt.Circle((x0_m, y0_m), rayon, color='grey', alpha=0.3, label='Position initiale', zorder=3))
        ax.add_patch(plt.Circle((x_i, y_i), rayon, color='black', alpha=0.6, label=f'Position à l impact (t={t_impact:.3f}s)', zorder=4))
    else:
        lx, ly = objet.get('taille_m', (5.0, 5.0))
        ax.add_patch(plt.Rectangle((x0_m - lx / 2, y0_m - ly / 2), lx, ly, color='grey', alpha=0.3, label='Position initiale', zorder=3))
        ax.add_patch(plt.Rectangle((x_i - lx / 2, y_i - ly / 2), lx, ly, color='black', alpha=0.6, label=f'Position à l impact (t={t_impact:.3f}s)', zorder=4))

    ax.annotate('', xy=(x_i, y_i), xytext=(x0_m, y0_m), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    x_e, y_e = pos_estimee
    ax.scatter(x_e, y_e, marker='*', color='gold', s=250, zorder=5, edgecolors='black', linewidths=0.8, label='Position estimée')

    for i, capteur in enumerate(capteurs):
        x_c, y_c = position_capteur_m(capteur, h=h)
        nom = capteur.get('nom', f'C{i + 1}')
        couleur = couleurs[i]

        ax.scatter(x_c, y_c, color=couleur, s=100, zorder=4, marker='^')
        ax.text(x_c + 0.5, y_c + 0.5, nom, color=couleur, fontsize=9, fontweight='bold')
        ax.add_patch(plt.Circle((x_c, y_c), radius=distances[i], color=couleur, fill=False, linestyle='--', linewidth=1.2, alpha=0.7, label=f'{nom} — d={distances[i]:.1f}m'))
        ax.plot([x_c, x_e], [y_c, y_e], color=couleur, linestyle='-', linewidth=1.2, alpha=0.6)

        xm = (x_c + x_e) / 2
        ym = (y_c + y_e) / 2
        ax.text(xm, ym, f'{distances[i]:.1f}m', color=couleur, fontsize=8, ha='center', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))

    erreur = np.linalg.norm(pos_estimee - pos_reelle)
    ax.set_title(f'Localisation par multilatération\nErreur : {erreur:.2f} m | t_impact = {t_impact:.3f}s', fontsize=13)
    ax.set_xlim(0, taille)
    ax.set_ylim(0, taille)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.show()
    return pos_reelle, t_impact


def trace_positions_estimees(capteurs, positions_estimees, positions_reelles, t_impacts, t0_pulses, objet, taille, h):
    '''Trace l ensemble des positions estimées au fil des pulses.'''
    fig, ax = plt.subplots(figsize=(9, 9))
    couleurs_cap = couleurs_capteurs(len(capteurs))
    positions_estimees = np.array(positions_estimees)

    if objet['forme'] == 'cercle':
        rayon = objet.get('rayon_m', 5.0)
    else:
        lx, ly = objet.get('taille_m', (5.0, 5.0))

    traj_x = [pos[0] for pos in positions_reelles]
    traj_y = [pos[1] for pos in positions_reelles]
    ax.plot(traj_x, traj_y, color='black', linestyle='--', linewidth=1.2, alpha=0.5, label='Trajectoire réelle', zorder=2)

    for k, (pos_reelle, t_impact) in enumerate(zip(positions_reelles, t_impacts)):
        x_r, y_r = pos_reelle
        label = 'Position réelle à l impact' if k == 0 else '_nolegend_'

        if objet['forme'] == 'cercle':
            ax.add_patch(plt.Circle((x_r, y_r), rayon, color='black', alpha=0.15, zorder=3, label=label))
        else:
            ax.add_patch(plt.Rectangle((x_r - lx / 2, y_r - ly / 2), lx, ly, color='black', alpha=0.15, zorder=3, label=label))

        ax.text(x_r + 0.5, y_r + 0.5, f'R{k + 1}\nt={t_impact:.2f}s', color='black', fontsize=7, alpha=0.7)

    for i, capteur in enumerate(capteurs):
        x_c, y_c = position_capteur_m(capteur, h=h)
        nom = capteur.get('nom', f'C{i + 1}')
        ax.scatter(x_c, y_c, color=couleurs_cap[i], s=100, zorder=5, marker='^')
        ax.text(x_c + 0.5, y_c + 0.5, nom, color=couleurs_cap[i], fontsize=10, fontweight='bold')

    cmap = plt.cm.plasma
    couleurs = [cmap(i / max(len(positions_estimees) - 1, 1)) for i in range(len(positions_estimees))]
    erreurs = []

    for i, (pos_est, pos_reelle, couleur, t0) in enumerate(zip(positions_estimees, positions_reelles, couleurs, t0_pulses)):
        erreur = np.linalg.norm(pos_est - np.array(pos_reelle))
        erreurs.append(erreur)

        x_e, y_e = pos_est
        x_r, y_r = pos_reelle
        ax.scatter(x_e, y_e, marker='*', color=couleur, s=250, zorder=6, edgecolors='black', linewidths=0.8, label=f'P{i + 1} | t={t0:.2f}s | err={erreur:.1f}m')
        ax.text(x_e + 0.5, y_e + 0.5, f'P{i + 1}', color=couleur, fontsize=9, fontweight='bold')
        ax.plot([x_r, x_e], [y_r, y_e], color=couleur, linestyle=':', linewidth=1.0, alpha=0.7)

    if len(positions_estimees) >= 2:
        ax.plot(positions_estimees[:, 0], positions_estimees[:, 1], color='orange', linestyle='-', linewidth=1.5, alpha=0.8, label='Trajectoire estimée', zorder=4)

    norm = plt.Normalize(vmin=min(t0_pulses), vmax=max(t0_pulses))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Temps d émission [s]', shrink=0.6)

    ax.set_xlim(0, taille)
    ax.set_ylim(0, taille)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.set_title(f'Évolution des positions estimées\nErreur moyenne : {np.mean(erreurs):.2f} m', fontsize=13)
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.show()


def estime_vitesse(positions_estimees, t_impacts, objet, afficher=True):
    '''Estime une vitesse par régression linéaire.'''
    positions_estimees = np.array(positions_estimees)
    t_impacts = np.array(t_impacts)

    if len(positions_estimees) < 2:
        print('⚠ Pas assez de positions pour estimer la vitesse.')
        return None, None

    a = np.column_stack([t_impacts, np.ones(len(t_impacts))])
    sol_x, _, _, _ = np.linalg.lstsq(a, positions_estimees[:, 0], rcond=None)
    sol_y, _, _, _ = np.linalg.lstsq(a, positions_estimees[:, 1], rcond=None)

    vx_est, x0_est = sol_x
    vy_est, y0_est = sol_y
    vitesse = np.array([vx_est, vy_est])

    print(f'\n{45 * "="}')
    print('  Estimation de la vitesse')
    print(f'{45 * "="}')
    print(f'  vx estimé : {vx_est:.2f} m/s')
    print(f'  vy estimé : {vy_est:.2f} m/s')
    print(f'  ||v||     : {np.linalg.norm(vitesse):.2f} m/s')

    if objet is not None and 'vitesse_m_s' in objet:
        vx_reel, vy_reel = objet['vitesse_m_s']
        vitesse_reelle = np.array([vx_reel, vy_reel])
        err_v = np.linalg.norm(vitesse - vitesse_reelle)
        print(f'  vx réel   : {vx_reel:.2f} m/s')
        print(f'  vy réel   : {vy_reel:.2f} m/s')
        print(f'  Erreur    : {err_v:.2f} m/s')

    print(f'{45 * "="}')

    if not afficher:
        return vitesse, (x0_est, y0_est)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    t_fine = np.linspace(t_impacts[0], t_impacts[-1], 100)

    for ax, coord, sol, label in zip(
        axes,
        [positions_estimees[:, 0], positions_estimees[:, 1]],
        [sol_x, sol_y],
        ['x [m]', 'y [m]'],
    ):
        v_est, p0_est = sol
        ax.scatter(t_impacts, coord, color='blue', zorder=5, label='Positions estimées')
        ax.plot(t_fine, v_est * t_fine + p0_est, color='red', linestyle='--', label=f'Régression : v = {v_est:.2f} m/s')
        ax.set_xlabel('Temps d impact [s]')
        ax.set_ylabel(label)
        ax.set_title(f'Régression linéaire — {label}')
        ax.legend()
        ax.grid(True, alpha=0.4)

    plt.suptitle('Estimation de la vitesse', fontsize=13)
    plt.tight_layout()
    plt.show()
    return vitesse, (x0_est, y0_est)


def methode_multilateration(params, capteurs, objet, afficher_echos=False):
    '''Lance la simulation avec et sans objet, puis compare les échos.'''
    objet_loin = {
        'forme': 'cercle',
        'centre_init_m': (0.0, 0.0),
        'vitesse_m_s': (0.0, 0.0),
        'rayon_m': 0.0,
        'c_objet': 5200.0,
        'gamma_objet': 4000.0,
        'actif': True,
    }

    base = {
        **params,
        'capteurs': capteurs,
        'emission_defaut': True,
        't_depart_pulses': 0.01,
        'dt_pulse': 0.01,
        'afficher_pml': False,
        'afficher_sources': False,
    }

    res_sans_objet = run_simul(
        **base,
        objet=objet_loin,
        afficher_animation=False,
        afficher_capteurs=False,
    )

    res_avec_objet = run_simul(
        **base,
        objet=objet,
        afficher_animation=True,
        afficher_capteurs=True,
    )

    temps = res_avec_objet['temps']
    signaux_avec = res_avec_objet['signaux_capteurs']
    signaux_sans = res_sans_objet['signaux_capteurs']
    c_eau = res_sans_objet['c_eau']
    h = res_sans_objet['h']
    taille = params['taille']
    t_total = params['t_total']

    intensite_echo = np.array([
        signaux_avec[i] - signaux_sans[i]
        for i in range(len(signaux_avec))
    ])

    affiche_echos(intensite_echo, temps, capteurs, t_total, afficher=afficher_echos)

    positions_estimees = []
    distances_pulse = []
    t0_pulses = []
    positions_reelles = []
    t_impacts = []

    for k, (idx_emit, t_debut, t_fin) in enumerate(plages_pulses(capteurs, t_total)):
        print(f'\n--- Pulse {k + 1} | Émetteur C{idx_emit + 1} | [{t_debut:.3f}s, {t_fin:.3f}s] ---')

        try:
            pos_est, temps_echo, distances, idx_retour = multilateration_un_pulse(
                intensite_echo,
                temps,
                capteurs,
                c_eau,
                h=h,
                idx_emetteur=idx_emit,
                t_debut=t_debut,
                t_fin=t_fin,
            )

            pos_reelle, t_impact = position_reelle_impact(
                objet,
                t_debut,
                distances[idx_retour],
                c_eau,
            )

            positions_estimees.append(pos_est)
            distances_pulse.append(distances)
            t0_pulses.append(t_debut)
            positions_reelles.append(pos_reelle)
            t_impacts.append(t_impact)

            erreur = np.linalg.norm(pos_est - pos_reelle)
            print(f'  Position réelle à l impact : {pos_reelle} m (t={t_impact:.3f}s)')
            print(f'  Position estimée           : {pos_est} m')
            print(f'  Erreur                     : {erreur:.2f} m')

            trace_localisation_echo(
                capteurs,
                pos_est,
                objet,
                distances,
                t_debut,
                distances[idx_retour],
                c_eau,
                taille,
                h,
                afficher=True,
            )

        except ValueError as err:
            print(f'  ⚠ Ignoré : {err}')

    if positions_estimees:
        trace_positions_estimees(
            capteurs,
            positions_estimees,
            positions_reelles,
            t_impacts,
            t0_pulses,
            objet,
            taille,
            h,
        )

    if len(positions_estimees) >= 2:
        estime_vitesse(positions_estimees, t_impacts, objet)

    return {
        'intensite_echo': intensite_echo,
        'positions_estimees': positions_estimees,
        'positions_reelles': positions_reelles,
        't_impacts': t_impacts,
        't0_pulses': t0_pulses,
        'distances_pulse': distances_pulse,
        'temps': temps,
    }


if __name__ == '__main__':
    params = {
        'taille': 1000,
        't_total': 3,
        'nx': 250,
        'ny': 250,
        'rho': 1000,
        'kappa': 2.2e9,
        'cfl': 1,
        'ratio_pml': 0.2,
        'puissance_pml': 0.6,
        'gamma_max': 30,
        'sigma_source': 2.1,
        'a0_source': 1,
        'tau_source': 0.005,
        'nb_frames': 100,
        'seed': 42,
    }

    params['capteurs'] = [
    {
        'nom': 'Capteur 1',
        'position_m': (258.0, 569.0),
        'emissions': [0.1],
    },
    {
        'nom': 'Capteur 2',
        'position_m': (430.0, 785.0),
        'emissions': [1.0],
    },
    {
        'nom': 'Capteur 3',
        'position_m': (480.0, 712.0),
        'emissions': [1.75],
    },
]
    objet = {
        'forme': 'cercle',
        'centre_init_m': (285.0, 387.0),
        'vitesse_m_s': (121.0, 88.0),
        'rayon_m': 10.0,
        'c_objet': 5200.0,
        'gamma_objet': 4000.0,
        'actif': True,
    }

    run_multilateration = True

    if run_multilateration:
        methode_multilateration(params, params['capteurs'], objet, afficher_echos=False)
    else:
        resultats = run_simul(
            **params,
            
            emission_defaut=True,
            t_depart_pulses=0.01,
            dt_pulse=0.01,
            objet=objet,
            afficher_pml=False,
            afficher_sources=True,
            afficher_animation=True,
            afficher_capteurs=True,
        )

        print('Simulation terminée')
        print('dt =', resultats['dt'])
        print('nt =', resultats['nt'])
        print('Nombre de pulses =', len(resultats['pulses']))
        print('Énergie résiduelle [%] =', resultats['pourcentage_residuel'])
