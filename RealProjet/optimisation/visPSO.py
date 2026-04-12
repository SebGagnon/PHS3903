import pickle
import numpy as np
import matplotlib.pyplot as plt

nom_fichier = "pso_pml_w0p8_c11p5_c21p5_pop15_max_iter15.pkl"

with open(nom_fichier, "rb") as f:
    data = pickle.load(f)

historique = data.get("historique_evaluations", [])

epaisseurs = np.array([d.get("epaisseur_pml_ratio", np.nan) for d in historique], dtype=float)
gamma_vals = np.array([d.get("gamma_max", np.nan) for d in historique], dtype=float)
puissances = np.array([d.get("puissance_pml", np.nan) for d in historique], dtype=float)
scores = np.array([d.get("score_moyen", np.nan) for d in historique], dtype=float)

masque = (
    np.isfinite(epaisseurs)
    & np.isfinite(gamma_vals)
    & np.isfinite(puissances)
    & np.isfinite(scores)
)

epaisseurs = epaisseurs[masque]
gamma_vals = gamma_vals[masque]
puissances = puissances[masque]
scores = scores[masque]

if "gbest_y_hist" in data:
    gbest_y_hist = np.array(data["gbest_y_hist"], dtype=float).reshape(-1)
else:
    gbest_y_hist = np.minimum.accumulate(scores)

if "best_epaisseur_ratio" in data:
    best_epaisseur = float(data["best_epaisseur_ratio"])
    best_gamma = float(data.get("best_gamma_max", np.nan))
    best_puissance = float(data.get("best_puissance", np.nan))
    best_score = float(data.get("best_score", np.nan))
else:
    idx_best = np.argmin(scores)
    best_epaisseur = epaisseurs[idx_best]
    best_gamma = gamma_vals[idx_best]
    best_puissance = puissances[idx_best]
    best_score = scores[idx_best]

# ---- figure avec espace réservé pour la colorbar ----
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle(
    f"Résultats PSO\n"
    f"best score = {best_score:.6f} | "
    f"ratio = {best_epaisseur:.5f} | "
    f"gamma = {best_gamma:.3f} | "
    f"puissance = {best_puissance:.3f}",
    fontsize=13
)

# on garde de la place à droite
plt.subplots_adjust(right=0.88, wspace=0.28, hspace=0.28)

# 1) convergence
ax = axes[0, 0]
ax.plot(gbest_y_hist, marker="o")
ax.set_title("Convergence")
ax.set_xlabel("Itération / évaluation")
ax.set_ylabel("Meilleur score [%]")
ax.grid(True)

# normalisation commune pour les 3 scatter
vmin = np.min(scores)
vmax = np.max(scores)

# 2) épaisseur vs gamma
ax = axes[0, 1]
sc1 = ax.scatter(epaisseurs, gamma_vals, c=scores, s=45, vmin=vmin, vmax=vmax)
ax.scatter(best_epaisseur, best_gamma, marker="*", s=260, edgecolors="black")
ax.set_title("épaisseur vs gamma")
ax.set_xlabel("epaisseur_pml_ratio")
ax.set_ylabel("gamma_max")

# 3) épaisseur vs puissance
ax = axes[1, 0]
sc2 = ax.scatter(epaisseurs, puissances, c=scores, s=45, vmin=vmin, vmax=vmax)
ax.scatter(best_epaisseur, best_puissance, marker="*", s=260, edgecolors="black")
ax.set_title("épaisseur vs puissance")
ax.set_xlabel("epaisseur_pml_ratio")
ax.set_ylabel("puissance_pml")

# 4) gamma vs puissance
ax = axes[1, 1]
sc3 = ax.scatter(gamma_vals, puissances, c=scores, s=45, vmin=vmin, vmax=vmax)
ax.scatter(best_gamma, best_puissance, marker="*", s=260, edgecolors="black")
ax.set_title("gamma vs puissance")
ax.set_xlabel("gamma_max")
ax.set_ylabel("puissance_pml")

# colorbar dans un axe séparé, à droite
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
cbar = fig.colorbar(sc3, cax=cbar_ax)
cbar.set_label("score moyen [%]")

plt.show()