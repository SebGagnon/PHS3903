import pickle
import numpy as np
import matplotlib.pyplot as plt


def charger_pickle(nom_fichier):
    with open(nom_fichier, "rb") as f:
        return pickle.load(f)


def extraire_maillage(resultats_maillage):
    nx_vals = []
    erreur_moy = []

    for nx in sorted(resultats_maillage.keys()):
        bloc = resultats_maillage[nx]
        nx_vals.append(bloc["nx"])
        erreur_moy.append(bloc.get("erreur_moyenne", np.nan))

    return {
        "nx": np.array(nx_vals, dtype=float),
        "erreur_moyenne": np.array(erreur_moy, dtype=float),
    }


def extraire_dt(resultats_dt):
    dt_vals = []
    erreur_moy = []

    for dt in sorted(resultats_dt.keys()):
        bloc = resultats_dt[dt]
        dt_vals.append(bloc["dt"])
        erreur_moy.append(bloc.get("erreur_moyenne", np.nan))

    return {
        "dt": np.array(dt_vals, dtype=float),
        "erreur_moyenne": np.array(erreur_moy, dtype=float),
    }


def tracer_erreur_maillage(data):
    plt.figure(figsize=(8, 5))
    plt.plot(data["nx"], data["erreur_moyenne"], marker="o")
    plt.xlabel("nx = ny")
    plt.ylabel("Erreur moyenne [m]")
    plt.title("Erreur moyenne de multilateration en fonction du maillage")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()


def tracer_erreur_dt(data):
    plt.figure(figsize=(8, 5))
    plt.plot(data["dt"], data["erreur_moyenne"], marker="o")
    plt.xlabel("dt [s]")
    plt.ylabel("Erreur moyenne [m]")
    plt.title("Erreur moyenne de multilateration en fonction du pas de temps")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()


def afficher_resume(resultats_maillage, resultats_dt):
    print("\\n--- Resume maillage ---")
    for nx in sorted(resultats_maillage.keys()):
        bloc = resultats_maillage[nx]
        print(f"nx = {bloc['nx']:>4} | erreur moyenne = {bloc.get('erreur_moyenne', np.nan):.6f} m")

    print("\\n--- Resume dt ---")
    for dt in sorted(resultats_dt.keys()):
        bloc = resultats_dt[dt]
        print(f"dt = {bloc['dt']:.6e} s | erreur moyenne = {bloc.get('erreur_moyenne', np.nan):.6f} m")


if __name__ == "__main__":
    nom_pickle_maillage = "etude_maillage2.pkl"
    nom_pickle_dt = "etude_dt2.pkl"

    resultats_maillage = charger_pickle(nom_pickle_maillage)
    resultats_dt = charger_pickle(nom_pickle_dt)

    afficher_resume(resultats_maillage, resultats_dt)

    data_maillage = extraire_maillage(resultats_maillage)
    data_dt = extraire_dt(resultats_dt)

    tracer_erreur_maillage(data_maillage)
    tracer_erreur_dt(data_dt)

    plt.show()