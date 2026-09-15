import matplotlib.pyplot as plt
import numpy as np


def comp_rmse(u_truth, u_est_hist, t_eval):
    # 1. Error espacial instantáneo en cada paso de tiempo
    rmse_per_step = np.sqrt(np.mean((u_truth - u_est_hist) ** 2, axis=0))

    # 2. Error global promedio espacio-temporal
    global_rmse = np.sqrt(np.mean((u_truth - u_est_hist) ** 2))
    max_rmse = np.max(rmse_per_step)

    print(f"==================================================")
    print(f" EVALUACIÓN DE PRECISIÓN GLOBAL DEL AKF")
    print(f" -> RMSE Medio Global: {global_rmse:.6e} m")
    print(f" -> RMSE Máximo en Ráfaga: {max_rmse:.6e} m")
    print(f"==================================================")

    # 3. Gráfica de error transitorio
    plt.figure(figsize=(9, 4))
    plt.plot(
        t_eval,
        rmse_per_step,
        color="#1f77b4",
        linewidth=1.8,
        label="RMSE Espacial $RMSE(t)$",
    )
    plt.axhline(
        global_rmse,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label=f"RMSE Promedio Global ({global_rmse:.2e} m)",
    )

    plt.title("Evolución Temporal del Error de Reconstrucción (Todos los DOFs)")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Error RMSE (m)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("rmse.png")
    plt.show()


def plot_force_estimation(f_modal_truth, f_est_hist, t_eval):
    """Compara las fuerzas modales reales de la ráfaga de viento contra las

    fuerzas modales estimadas por el AKF en tiempo real.
    """
    num_modes_est = f_est_hist.shape[0]

    # Graficamos los primeros 3 modos principales (Flexión 1, Torsión 1, Flexión 2)
    modes_to_plot = min(3, num_modes_est)
    fig, axes = plt.subplots(modes_to_plot, 1, figsize=(9, 6), sharex=True)

    if modes_to_plot == 1:
        axes = [axes]

    for i in range(modes_to_plot):
        axes[i].plot(
            t_eval,
            f_modal_truth[i, :],
            "k--",
            linewidth=1.5,
            label=f"Fuerza Real (Modo {i+1})",
        )
        axes[i].plot(
            t_eval,
            f_est_hist[i, :],
            color="#2ca02c",
            linewidth=1.8,
            label=f"AKF Estimado (Modo {i+1})",
        )
        axes[i].set_ylabel(f"Fuerza Modal {i+1} (N)")
        axes[i].grid(True, linestyle="--", alpha=0.6)
        axes[i].legend(loc="upper right")

    axes[-1].set_xlabel("Tiempo (s)")
    fig.suptitle(
        "Reconstrucción Virtual Sensing de Cargas Aerodinámicas (Viento)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig("estimated_forces.png")
    plt.show()

def reconstruct_force_field(Phi, f_est_hist):
    """Proyecta las fuerzas modales estimadas (m, steps) a fuerzas físicas por

    nodo (nodes, steps).
    """
    # F_nodal (dofs, steps) = Phi * f_est
    f_nodal_dofs = Phi @ f_est_hist
    return f_nodal_dofs