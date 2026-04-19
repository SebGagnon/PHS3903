import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def charger_resultats(path):
    '''Charge un fichier pickle de résultats BO.''' 
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def extraire_essais(data):
    '''Extrait gamma_max, puissance_pml et score depuis différents formats.''' 
    gamma_vals = []
    puissance_vals = []
    scores = []

    if isinstance(data, dict):
        if 'historique' in data and isinstance(data['historique'], list) and data['historique']:
            for entree in data['historique']:
                gamma = entree.get('gamma_max')
                puissance = entree.get('puissance_pml')
                score = entree.get('score')
                if gamma is None or puissance is None or score is None:
                    continue
                gamma_vals.append(float(gamma))
                puissance_vals.append(float(puissance))
                scores.append(float(score))
            if gamma_vals:
                return np.array(gamma_vals), np.array(puissance_vals), np.array(scores)

        if 'x_iters' in data and 'func_vals' in data:
            x_iters = data['x_iters']
            func_vals = data['func_vals']
            for x, y in zip(x_iters, func_vals):
                if len(x) < 2:
                    continue
                gamma_vals.append(float(x[0]))
                puissance_vals.append(float(x[1]))
                scores.append(float(y))
            if gamma_vals:
                return np.array(gamma_vals), np.array(puissance_vals), np.array(scores)

    if hasattr(data, 'x_iters') and hasattr(data, 'func_vals'):
        for x, y in zip(data.x_iters, data.func_vals):
            if len(x) < 2:
                continue
            gamma_vals.append(float(x[0]))
            puissance_vals.append(float(x[1]))
            scores.append(float(y))
        if gamma_vals:
            return np.array(gamma_vals), np.array(puissance_vals), np.array(scores)

    raise ValueError(
        'Format non reconnu. Il faut un dict avec historique/x_iters/func_vals ou un résultat skopt.'
    )


def trouver_meilleur(gamma_vals, puissance_vals, scores):
    '''Retourne le meilleur essai.''' 
    idx = int(np.argmin(scores))
    return {
        'idx': idx,
        'gamma_max': gamma_vals[idx],
        'puissance_pml': puissance_vals[idx],
        'score': scores[idx],
    }


def tracer_convergence(scores):
    '''Trace la convergence du score.''' 
    meilleur_cumul = np.minimum.accumulate(scores)

    plt.figure(figsize=(8, 4))
    plt.plot(scores, 'o-', label='score évalué')
    plt.plot(meilleur_cumul, '-', linewidth=2, label='meilleur cumulatif')
    plt.xlabel('Évaluation')
    plt.ylabel('Score')
    plt.title('Convergence')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()



def tracer_gamma_score(gamma_vals, scores, best):
    '''Trace score en fonction de gamma_max.''' 
    plt.figure(figsize=(8, 4))
    plt.scatter(gamma_vals, scores, c=np.arange(len(scores)), s=45)
    plt.scatter(best['gamma_max'], best['score'], marker='*', s=250, edgecolors='black', linewidths=0.8)
    plt.xscale('log')
    plt.xlabel('gamma_max')
    plt.ylabel('Score')
    plt.title('Score selon gamma_max')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()



def tracer_puissance_score(puissance_vals, scores, best):
    '''Trace score en fonction de puissance_pml.''' 
    plt.figure(figsize=(8, 4))
    plt.scatter(puissance_vals, scores, c=np.arange(len(scores)), s=45)
    plt.scatter(best['puissance_pml'], best['score'], marker='*', s=250, edgecolors='black', linewidths=0.8)
    plt.xlabel('puissance_pml')
    plt.ylabel('Score')
    plt.title('Score selon puissance_pml')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()



def tracer_carte_2d(gamma_vals, puissance_vals, scores, best):
    '''Réflection en fonction du gamma maximale et de la puissance''' 
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(gamma_vals, puissance_vals, c=scores, s=70)
    plt.scatter(best['gamma_max'], best['puissance_pml'], marker='*', s=260, edgecolors='black', linewidths=0.8)
    plt.xscale('log')
    plt.xlabel('gamma_max')
    plt.ylabel('puissance_pml')
    plt.title('Réflection en fonction du gamma maximale et de la puissance')
    plt.grid(True, alpha=0.3)
    plt.colorbar(sc, label='Score')
    plt.tight_layout()



def imprimer_resume(best, n_essais):
    '''Affiche un résumé texte.''' 
    print('\nRésumé')
    print(f'Nombre d essais    : {n_essais}')
    print(f'Meilleur gamma_max : {best["gamma_max"]:.6e}')
    print(f'Meilleure puissance: {best["puissance_pml"]:.6f}')
    print(f'Meilleur score     : {best["score"]:.6e}')
    print(f'Index meilleur     : {best["idx"]}')



def visualiser(path='bo_pml_result.pkl'):
    '''Charge et visualise les résultats d optimisation.''' 
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Fichier introuvable: {path}')

    data = charger_resultats(path)
    gamma_vals, puissance_vals, scores = extraire_essais(data)
    best = trouver_meilleur(gamma_vals, puissance_vals, scores)

    imprimer_resume(best, len(scores))
    tracer_convergence(scores)
    tracer_gamma_score(gamma_vals, scores, best)
    tracer_puissance_score(puissance_vals, scores, best)
    tracer_carte_2d(gamma_vals, puissance_vals, scores, best)
    plt.show()



def main():
   
    fichier = 'bo_pml_test.pkl'
    visualiser(fichier)


if __name__ == '__main__':
    main()
