import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

with open("optimisation_pml_residuel.pkl", "rb") as f:
    historique_bo = pickle.load(f)

positions = historique_bo["positions_essais_m"]
essais_bo = historique_bo["essais_bo"]
x_best = historique_bo["x_best"]
fun_best = historique_bo["fun_best"]

# -----------------------------
# 1) Positions aleatoires des essais spatiaux
# -----------------------------
x_pos = [p[0] for p in positions]
y_pos = [p[1] for p in positions]

plt.figure(figsize=(6, 6))
plt.scatter(x_pos, y_pos, s=60)
plt.axvline(500, linestyle="--", linewidth=1)
plt.axhline(500, linestyle="--", linewidth=1)

# carré utile 600x600 centré
plt.plot([200, 800, 800, 200, 200], [200, 200, 800, 800, 200], linewidth=1.5)

plt.xlim(0, 1000)
plt.ylim(0, 1000)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Positions aléatoires des essais")
plt.gca().set_aspect("equal")
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------
# 2) Points testés par l'optimisation bayésienne
# -----------------------------
gamma = [e["gamma_max"] for e in essais_bo]
puissance = [e["puissance_pml"] for e in essais_bo]
cout = [e["cout"] for e in essais_bo]

cout_array = np.array(cout)

fig,ax = plt.subplots(figsize=(7, 5))
vmin = np.percentile(cout_array, 5)
vmax = np.percentile(cout_array, 95)

sc = plt.scatter(
    gamma,
    puissance,
    c=cout_array,
    s=70,
    norm=LogNorm(vmin=max(vmin, 1e-12), vmax=vmax)
)

plt.scatter(
    x_best[0], x_best[1],
    marker="x", s=25, label="Meilleure combinaison"
)
ticks = [1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5]

plt.xlabel("gamma_max")
plt.ylabel("puissance_pml")
plt.colorbar(sc, label="Coût (échelle log)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 3) Evolution du coût au fil des essais
# -----------------------------
plt.figure(figsize=(8, 4))
plt.plot(cout, marker="o")
plt.axhline(fun_best, linestyle="--", linewidth=1, label="Meilleur coût")
plt.xlabel("Itération")
plt.ylabel("Coût")
plt.title("Évolution du coût")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------
# 4) Affichage texte
# -----------------------------
print("Meilleurs paramètres :", x_best)
print("Meilleur coût :", fun_best)