import pickle
import numpy as np
import matplotlib.pyplot as plt


def charger_meilleur_essai(nom_fichier):
    '''Charge le meilleur essai valide depuis le fichier pickle.'''
    with open(nom_fichier, 'rb') as f:
        data = pickle.load(f)

    historique = data.get('historique', [])
    valides = [
        h for h in historique
        if h.get('valide', True) and np.isfinite(h.get('score', np.inf))
    ]

    if len(valides) == 0:
        raise ValueError('Aucun essai valide trouvé.')

    meilleur = min(valides, key=lambda h: h['score'])
    return data, meilleur


def plot_scores_tests(ax, meilleur):
    '''Trace le score de chaque test aléatoire du meilleur essai.'''
    scores = meilleur.get('scores_individuels', [])

    if len(scores) == 0:
        ax.set_title('Aucun score individuel sauvegardé')
        return

    x = np.arange(1, len(scores) + 1)

    ax.plot(x, scores, 'o-')
    ax.axhline(np.mean(scores), linestyle='--', label=f'moyenne = {np.mean(scores):.3f}')
    ax.set_xlabel('Test aléatoire')
    ax.set_ylabel('Erreur moyenne [m]')
    ax.set_title('Scores des tests du meilleur essai')
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_hist_erreurs(ax, meilleur):
    '''Trace l histogramme de toutes les erreurs par pulse du meilleur essai.'''
    details = meilleur.get('details_tests', [])

    erreurs = []
    for d in details:
        erreurs.extend(d.get('erreurs', []))

    if len(erreurs) == 0:
        ax.set_title('Aucune erreur par pulse sauvegardée')
        return

    ax.hist(erreurs, bins=15)
    ax.set_xlabel('Erreur par pulse [m]')
    ax.set_ylabel('Nombre')
    ax.set_title('Distribution des erreurs par pulse')
    ax.grid(True, alpha=0.3)


def plot_box_erreurs(ax, meilleur):
    '''Trace un boxplot des erreurs par pulse pour chaque test.'''
    details = meilleur.get('details_tests', [])

    data_box = []
    labels = []

    for i, d in enumerate(details, start=1):
        err = d.get('erreurs', [])
        if len(err) > 0:
            data_box.append(err)
            labels.append(f'T{i}')

    if len(data_box) == 0:
        ax.set_title('Pas de données pour le boxplot')
        return

    ax.boxplot(data_box, labels=labels)
    ax.set_xlabel('Test')
    ax.set_ylabel('Erreur par pulse [m]')
    ax.set_title('Erreurs par pulse pour chaque test')
    ax.grid(True, alpha=0.3)


def plot_objets(ax, data, meilleur):
    '''Trace les positions initiales et directions des objets testés.'''
    params = data.get('params', {})
    taille = params.get('taille', None)
    ratio_pml = params.get('ratio_pml', None)

    details = meilleur.get('details_tests', [])
    scores = meilleur.get('scores_individuels', [])

    if len(details) == 0:
        ax.set_title('Aucun objet sauvegardé')
        return

    s_min = min(scores) if len(scores) > 0 else 0.0
    s_max = max(scores) if len(scores) > 0 else 1.0

    for i, d in enumerate(details):
        objet = d.get('objet', {})
        score = d.get('score', np.nan)

        if 'centre_init_m' not in objet or 'vitesse_m_s' not in objet:
            continue

        x0, y0 = objet['centre_init_m']
        vx, vy = objet['vitesse_m_s']

        if np.isfinite(score) and s_max > s_min:
            t = (score - s_min) / (s_max - s_min)
            couleur = plt.cm.viridis(1 - t)
        else:
            couleur = 'black'

        ax.scatter(x0, y0, color=couleur, s=50)
        ax.arrow(
            x0, y0,
            0.8 * vx, 0.8 * vy,
            head_width=10,
            head_length=15,
            length_includes_head=True,
            color=couleur,
            alpha=0.8,
        )
        ax.text(x0 + 5, y0 + 5, f'T{i+1}', fontsize=8)

    if taille is not None:
        ax.set_xlim(0, taille)
        ax.set_ylim(0, taille)

    if taille is not None and ratio_pml is not None:
        debut = ratio_pml * taille
        fin = (1 - ratio_pml) * taille

        ax.axvline(debut, color='gray', linestyle='--', alpha=0.6)
        ax.axvline(fin, color='gray', linestyle='--', alpha=0.6)
        ax.axhline(debut, color='gray', linestyle='--', alpha=0.6)
        ax.axhline(fin, color='gray', linestyle='--', alpha=0.6)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Objets testés pour le meilleur essai')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

def plot_top_capteurs(data, n_top=3):
    '''Affiche les positions des capteurs pour les n meilleurs essais.'''
    historique = data.get('historique', [])
    valides = [
        h for h in historique
        if h.get('valide', True) and np.isfinite(h.get('score', np.inf))
    ]

    if len(valides) == 0:
        print('Aucun essai valide trouvé.')
        return

    top = sorted(valides, key=lambda h: h['score'])[:n_top]

    params = data.get('params', {})
    taille = params.get('taille', None)
    ratio_pml = params.get('ratio_pml', None)

    fig, ax = plt.subplots(figsize=(8, 8))
    couleurs = ['red', 'blue', 'green', 'orange', 'purple']

    for k, essai in enumerate(top):
        capteurs = essai.get('capteurs', [])
        couleur = couleurs[k % len(couleurs)]

        for i, capteur in enumerate(capteurs, start=1):
            x_m, y_m = capteur['position_m']
            label = f'Essai {k+1} | score={essai["score"]:.3f}' if i == 1 else None

            ax.scatter(x_m, y_m, color=couleur, s=120, marker='^', label=label)
            ax.text(x_m + 5, y_m + 5, f'E{k+1}-C{i}', color=couleur, fontsize=9, weight='bold')

    if taille is not None:
        ax.set_xlim(0, taille)
        ax.set_ylim(0, taille)

    if taille is not None and ratio_pml is not None:
        debut = ratio_pml * taille
        fin = (1 - ratio_pml) * taille

        ax.axvline(debut, color='gray', linestyle='--', alpha=0.6)
        ax.axvline(fin, color='gray', linestyle='--', alpha=0.6)
        ax.axhline(debut, color='gray', linestyle='--', alpha=0.6)
        ax.axhline(fin, color='gray', linestyle='--', alpha=0.6)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'Positions des capteurs des {len(top)} meilleurs essais')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
def print_resume(meilleur):
    print('Meilleur score moyen =', meilleur.get('score'))
    print('Distance min entre capteurs =', meilleur.get('distance_min'))

    print('\nCapteurs du meilleur essai :')
    for capteur in meilleur.get('capteurs', []):
        print(capteur)

    scores = meilleur.get('scores_individuels', [])
    if len(scores) > 0:
        print('\nScores individuels :')
        print(np.round(scores, 4))


def main():
    nom_fichier = 'bo_positions_capteurs_test3.pkl'

    data, meilleur = charger_meilleur_essai(nom_fichier)
    print_resume(meilleur)
    plot_top_capteurs(data, n_top=3)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    plot_scores_tests(axes[0, 0], meilleur)
    plot_hist_erreurs(axes[0, 1], meilleur)
    plot_box_erreurs(axes[1, 0], meilleur)
    plot_objets(axes[1, 1], data, meilleur)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()