import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# fichier final à ouvrir
# --------------------------------------------------
print("Dossier courant :", os.getcwd())
print("Fichiers dans ce dossier :")
print(os.listdir())
nom_fichier = "bo_pml_gp_trials40_seed200_final.pkl"

with open(nom_fichier, "rb") as f:
    data = pickle.load(f)

historique = data.get("historique_evaluations", [])
if len(historique) == 0:
    raise ValueError("Le fichier ne contient pas 'historique_evaluations'.")

# --------------------------------------------------
# extraction
# --------------------------------------------------
gamma_vals = np.array([d.get("gamma_max", np.nan) for d in historique], dtype=float)
puissances = np.array([d.get("puissance_pml", np.nan) for d in historique], dtype=float)
scores = np.array([d.get("score_moyen", np.nan) for d in historique], dtype=float)
trial_numbers = np.array([d.get("trial_number", i) for i, d in enumerate(historique)], dtype=float)
scores_par_seed = [d.get("scores_par_seed", []) for d in historique]

masque = np.isfinite(gamma_vals) & np.isfinite(puissances) & np.isfinite(scores)
gamma_vals = gamma_vals[masque]
puissances = puissances[masque]
scores = scores[masque]
trial_numbers = trial_numbers[masque]
scores_par_seed = [scores_par_seed[i] for i in range(len(masque)) if masque[i]]

if len(scores) == 0:
    raise ValueError("Aucune évaluation exploitable trouvée.")

# meilleure convergence
if "gbest_y_hist" in data:
    gbest_y_hist = np.array(data["gbest_y_hist"], dtype=float).reshape(-1)
else:
    gbest_y_hist = np.minimum.accumulate(scores)

best_gamma = float(data.get("best_gamma_max", gamma_vals[np.argmin(scores)]))
best_puissance = float(data.get("best_puissance", puissances[np.argmin(scores)]))
best_score = float(data.get("best_score", np.min(scores)))

print("===== Résumé =====")
print(f"Nombre d'évaluations : {len(scores)}")
print(f"Best gamma_max       : {best_gamma:.6f}")
print(f"Best puissance_pml   : {best_puissance:.6f}")
print(f"Best score           : {best_score:.6f}")

# --------------------------------------------------
# figure principale
# --------------------------------------------------
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.28)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

fig.suptitle(
    f"Résultats optimisation bayésienne\n"
    f"best score = {best_score:.6f} | "
    f"gamma = {best_gamma:.3f} | "
    f"puissance = {best_puissance:.3f}",
    fontsize=13
)

# 1) convergence
ax1.plot(gbest_y_hist, marker="o")
ax1.set_title("Convergence")
ax1.set_xlabel("Itération / essai")
ax1.set_ylabel("Meilleur score [%]")
ax1.grid(True)

# échelle couleur commune
vmin = np.min(scores)
vmax = np.max(scores)

# 2) gamma vs puissance
sc = ax2.scatter(
    gamma_vals,
    puissances,
    c=scores,
    s=65,
    vmin=vmin,
    vmax=vmax
)
ax2.scatter(best_gamma, best_puissance, marker="*", s=260, edgecolors="black")
ax2.set_title("gamma_max vs puissance_pml")
ax2.set_xlabel("gamma_max")
ax2.set_ylabel("puissance_pml")
ax2.set_xscale("log")
ax2.grid(True, alpha=0.3)

# 3) score vs gamma
ax3.scatter(gamma_vals, scores, c=scores, s=55, vmin=vmin, vmax=vmax)
ax3.axvline(best_gamma, linestyle="--")
ax3.set_title("Score vs gamma_max")
ax3.set_xlabel("gamma_max")
ax3.set_ylabel("score moyen [%]")
ax3.set_xscale("log")
ax3.grid(True, alpha=0.3)

# 4) score vs puissance
ax4.scatter(puissances, scores, c=scores, s=55, vmin=vmin, vmax=vmax)
ax4.axvline(best_puissance, linestyle="--")
ax4.set_title("Score vs puissance_pml")
ax4.set_xlabel("puissance_pml")
ax4.set_ylabel("score moyen [%]")
ax4.grid(True, alpha=0.3)

# colorbar séparée
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.70])
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.set_label("score moyen [%]")

plt.show()

# --------------------------------------------------
# top essais
# --------------------------------------------------
idx_tries = np.argsort(scores)
top_k = min(10, len(scores))

print("\n===== Top essais =====")
for rank, idx in enumerate(idx_tries[:top_k], start=1):
    print(
        f"{rank:02d} | trial={int(trial_numbers[idx])} | "
        f"score={scores[idx]:.6f} | "
        f"gamma={gamma_vals[idx]:.6f} | "
        f"puissance={puissances[idx]:.6f}"
    )

# --------------------------------------------------
# variabilité par seed pour les meilleurs essais
# --------------------------------------------------
top_seed_scores = [scores_par_seed[idx] for idx in idx_tries[:top_k]]
labels = [f"T{int(trial_numbers[idx])}" for idx in idx_tries[:top_k]]

plt.figure(figsize=(11, 5))
plt.boxplot(top_seed_scores, tick_labels=labels, showfliers=True)
plt.title("Variabilité entre seeds pour les meilleurs essais")
plt.xlabel("Trial")
plt.ylabel("Pourcentage résiduel [%]")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()