import numpy as np
from src.akf_filter import AugmentedKalmanFilter
from src.rom_loader import ModalROMLoader
from src.visualizer import animate_digital_twin_3d


def main():
    rst_path = r"AGARD_445_6_files/dp0/SYS/MECH/file.rst"

    # 1. Load Modal ROM
    rom_loader = ModalROMLoader(rst_path, num_modes=8)
    A_cont, B_cont, freqs = rom_loader.build_state_space()
    Phi, num_nodes, active_dof, coords_orig = rom_loader.extract_mode_shapes()

    # 2. Configure Physical Sensor Locations
    sensor_nodes = [
        int(num_nodes * 0.1),
        int(num_nodes * 0.4),
        int(num_nodes * 0.8),
    ]
    sensor_dofs = [n * 3 + active_dof for n in sensor_nodes]

    # 3. Initialize AKF Engine
    dt = 0.001
    std_noise = 1e-4
    akf = AugmentedKalmanFilter(
        A_cont, B_cont, Phi, sensor_dofs, dt=dt, std_noise=std_noise
    )

    # 4. Generate Ground Truth Dynamics
    # =====================================================================
    # GENERACIÓN DE FUERZA DE VIENTO REALISTA (RÁFAGA "1-cos" + TURBULENCIA)
    # =====================================================================
    t_eval = np.arange(0, 1.0, dt)  # Simulación de 1 segundo
    n_steps = len(t_eval)
    
    # 1. Ráfaga discreta estándar CS-25 ("1 - cos gust")
    t_gust_start = 0.1
    t_gust_duration = 0.3
    gust = np.zeros(n_steps)
    
    idx_gust = np.where(
        (t_eval >= t_gust_start) & (t_eval <= t_gust_start + t_gust_duration)
    )[0]
    t_rel = t_eval[idx_gust] - t_gust_start
    gust[idx_gust] = 0.5 * (1.0 - np.cos(2 * np.pi * t_rel / t_gust_duration))
    
    # 2. Turbulencia atmosférica (Ruido coloreado / viento aleatorio)
    np.random.seed(42)
    turbulence = np.random.normal(0, 0.15, n_steps)
    
    # Fuerza aerodinámica total en N (Empuje vertical ascendente)
    f_wind_total = 80.0 * (gust + turbulence)
    
    # 3. Proyección sobre las coordenadas modales (Excita Flexión y Torsión)
    f_modal_real = np.zeros((8, n_steps))
    f_modal_real[0, :] = f_wind_total * 1.0  # Modo 1: Flexión principal
    f_modal_real[1, :] = f_wind_total * 0.4  # Modo 2: Torsión del ala
    f_modal_real[2, :] = f_wind_total * 0.1  # Modo 3: Flexión secundaria

    nx = A_cont.shape[0]
    x_real = np.zeros((nx, n_steps))
    u_truth = np.zeros((num_nodes * 3, n_steps))

    for k in range(n_steps - 1):
        x_real[:, k + 1] = akf.Ad @ x_real[:, k] + akf.Bd @ f_modal_real[:, k]
        u_truth[:, k] = Phi @ x_real[:8, k]
    u_truth[:, -1] = Phi @ x_real[:8, -1]

    # Sensor measurements with additive noise
    z_measured = u_truth[sensor_dofs, :] + np.random.normal(
        0, std_noise, size=(len(sensor_dofs), n_steps)
    )

    # 5. Run AKF State Estimation
    u_est_hist = np.zeros((num_nodes * 3, n_steps))
    for k in range(n_steps):
        q_est, _ = akf.step(z_measured[:, k])
        u_est_hist[:, k] = akf.reconstruct_fields(q_est)

    # 6. Render Interactive 3D Digital Twin
    dof_virtual = int(num_nodes * 0.5) * 3 + active_dof
    animate_digital_twin_3d(
        coords_orig=coords_orig,
        u_truth=u_truth,
        u_est_hist=u_est_hist,
        sensor_nodes=sensor_nodes,
        active_dof=active_dof,  # <--- Pasa únicamente active_dof (vale 2 para Uz)
        dt=dt,
        scale=15.0,
        save_gif=True,
        gif_name="digital_twin_animation.gif",
    )


if __name__ == "__main__":
    main()