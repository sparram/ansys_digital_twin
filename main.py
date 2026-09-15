import numpy as np
from src.akf_filter import AugmentedKalmanFilter
from src.rom_loader import ModalROMLoader
from src.utils import comp_rmse
from src.visualizer import animate_digital_twin_3d


def main():
    rst_path = r"AGARD_445_6_files/dp0/SYS/MECH/file.rst"
    dt = 0.001
    std_noise = 1e-4

    # =====================================================================
    # 1. MODELO DE ALTA FIDELIDAD (REALIDAD / GROUND TRUTH): 20 MODOS
    # =====================================================================
    rom_truth = ModalROMLoader(rst_path, num_modes=20)
    A_cont_truth, B_cont_truth, _ = rom_truth.build_state_space()
    Phi_truth, num_nodes, active_dof, coords_orig = (
        rom_truth.extract_mode_shapes()
    )

    # =====================================================================
    # 2. MODELO DE BAJA FIDELIDAD (GEMELO DIGITAL / AKF): 4 MODOS
    # =====================================================================
    rom_filter = ModalROMLoader(rst_path, num_modes=4)
    A_cont_filter, B_cont_filter, _ = rom_filter.build_state_space()
    Phi_filter, _, _, _ = rom_filter.extract_mode_shapes()

    # Ubicación de sensores físicos
    sensor_nodes = [
        int(num_nodes * 0.1),
        int(num_nodes * 0.4),
        int(num_nodes * 0.8),
    ]
    sensor_dofs = [n * 3 + active_dof for n in sensor_nodes]

    # AKF inicializado ÚNICAMENTE con el modelo restringido de 4 modos
    akf = AugmentedKalmanFilter(
        A_cont_filter,
        B_cont_filter,
        Phi_filter,
        sensor_dofs,
        dt=dt,
        std_noise=std_noise,
    )

    # =====================================================================
    # 3. GENERACIÓN DE LA "REALIDAD" (20 MODOS EXCITADOS)
    # =====================================================================
    t_eval = np.arange(0, 1.0, dt)
    n_steps = len(t_eval)

    # Ráfaga + Turbulencia
    t_gust_start, t_gust_duration = 0.1, 0.3
    gust = np.zeros(n_steps)
    idx_gust = np.where(
        (t_eval >= t_gust_start) & (t_eval <= t_gust_start + t_gust_duration)
    )[0]
    gust[idx_gust] = 0.5 * (
        1.0 - np.cos(2 * np.pi * (t_eval[idx_gust] - t_gust_start) / t_gust_duration)
    )

    np.random.seed(42)
    f_wind_total = 80.0 * (gust + np.random.normal(0, 0.15, n_steps))

    # Aplicamos fuerza a los 20 modos de la realidad
    f_modal_truth = np.zeros((20, n_steps))
    f_modal_truth[0, :] = f_wind_total * 1.0  # Modo 1
    f_modal_truth[1, :] = f_wind_total * 0.4  # Modo 2
    f_modal_truth[2, :] = f_wind_total * 0.1  # Modo 3
    # Los modos 4 al 20 también reciben pequeña excitación residual de viento
    for m in range(3, 20):
        f_modal_truth[m, :] = f_wind_total * (0.05 / (m + 1))

    # Simulación temporal de la realidad (20 modos)
    nx_truth = A_cont_truth.shape[0]
    x_real = np.zeros((nx_truth, n_steps))
    u_truth = np.zeros((num_nodes * 3, n_steps))

    # Matriz de discretización ZOH para el sistema real de 20 modos
    from scipy.linalg import expm

    M_disc_truth = np.zeros((nx_truth + 20, nx_truth + 20))
    M_disc_truth[:nx_truth, :nx_truth] = A_cont_truth
    M_disc_truth[:nx_truth, nx_truth:] = B_cont_truth
    M_exp_truth = expm(M_disc_truth * dt)
    Ad_truth = M_exp_truth[:nx_truth, :nx_truth]
    Bd_truth = M_exp_truth[:nx_truth, nx_truth:]

    for k in range(n_steps - 1):
        x_real[:, k + 1] = Ad_truth @ x_real[:, k] + Bd_truth @ f_modal_truth[:, k]
        u_truth[:, k] = Phi_truth @ x_real[:20, k]
    u_truth[:, -1] = Phi_truth @ x_real[:20, -1]

    # =====================================================================
    # 4. MEDICIÓN FÍSICA Y ESTIMACIÓN DEL AKF (8 MODOS)
    # =====================================================================
    z_measured = u_truth[sensor_dofs, :] + np.random.normal(
        0, std_noise, size=(len(sensor_dofs), n_steps)
    )

    u_est_hist = np.zeros((num_nodes * 3, n_steps))
    for k in range(n_steps):
        q_est, _ = akf.step(z_measured[:, k])
        u_est_hist[:, k] = akf.reconstruct_fields(q_est)

    # =====================================================================
    # 5. RENDERIZADO 3D
    # =====================================================================

    # 5.1 Compute RMSE Error and Plots 
    comp_rmse(u_truth, u_est_hist, t_eval)

    animate_digital_twin_3d(
        coords_orig=coords_orig,
        u_truth=u_truth,
        u_est_hist=u_est_hist,
        sensor_nodes=sensor_nodes,
        active_dof=active_dof,
        vtk_grid=rom_truth.result_file.grid,  # <--- Malla VTK directa de ANSYS
        dt=dt,
        scale=15.0,
        save_gif=True,
    )


if __name__ == "__main__":
    main()