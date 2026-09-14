import matplotlib.pyplot as plt
import numpy as np

def comp_rmse(u_truth, u_est_hist, t_eval):
    # 1. Error espacial instantaneous en cada paso de tiempo (sobre todos los 279 DOFs)
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
    plt.yscale("log")  # Escala logarítmica para apreciar transitorios
    plt.tight_layout()
    plt.savefig('rmse.png')
    plt.show()
