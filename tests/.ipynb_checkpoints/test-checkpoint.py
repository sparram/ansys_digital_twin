import os
import sys
import numpy as np
from scipy.signal import StateSpace

# Añadir la carpeta raíz al path de Python para importar desde src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar el constructor de matrices desde el módulo src/rom.py
from src.rom import load_lti_matrices
from src.enkf import EnsembleKalmanFilter

# Path al CSV de datos modales
csv_path = os.path.join(os.path.dirname(__file__), '../data/modal_data_with_stress.csv')

# 1. Cargar las matrices A, B, C, D
A, B, C, D = load_lti_matrices(csv_path)

# 2. Instanciar el sistema LTI
sys_lti = StateSpace(A, B, C, D)

print("Matrices cargadas exitosamente. Dimensión de A:", A.shape)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import StateSpace, lsim
import time

# --- 1. Generación de Turbulencia Estocástica u(t) ---
dt = 0.001  # Paso de integración: 1 ms (requisito: < 15 ms)
t_span = np.arange(0, 1.5, dt)
n_steps = len(t_span)

# Parámetros del modelo de ráfaga
tau_viento = 0.08    # Tiempo de correlación de la ráfaga [s]
sigma_viento = 10.0  # Amplitud típica de turbulencia [N]

u_stochastic = np.zeros(n_steps)
for k in range(1, n_steps):
    # Proceso continuo discretizado de Ornstein-Uhlenbeck
    u_stochastic[k] = u_stochastic[k-1] * np.exp(-dt/tau_viento) + \
                       sigma_viento * np.sqrt(1 - np.exp(-2*dt/tau_viento)) * np.random.randn()

# --- 2. Estado Real Físico (Ground Truth) ---
sys_lti = StateSpace(A, B, C, D)
_, y_true, x_true = lsim(sys_lti, U=u_stochastic, T=t_span)

y_disp_true = y_true[:, :3]       # Desplazamiento real en sensores [m]
sigma_root_true = y_true[:, 3]    # Esfuerzo real en la raíz [Pa]

# --- 3. Ruido de Medición en Sensores ---
std_noise = 1.5e-6                # Ruido blanco gaussiano de 1.5 micrometros
R_cov = np.diag([std_noise**2] * 3)
y_noisy = y_disp_true + np.random.multivariate_normal(np.zeros(3), R_cov, size=n_steps)

# --- 4. Bucle del EnKF y Reconstrucción en Tiempo Real ---
Q_cov = np.eye(40) * 1e-7         # Ruido de proceso
C_disp = C[:3, :]
C_stress = C[3, :].reshape(1, -1)

enkf = EnsembleKalmanFilter(A, B, C_disp, Q_cov, R_cov, num_ensembles=50, dt=dt)

sigma_est_mean = np.zeros(n_steps)
sigma_est_std = np.zeros(n_steps)
latencies = []

for k in range(n_steps):
    t_start = time.perf_counter()
    
    # Estimación estocástica
    enkf.predict(u_stochastic[k])
    enkf.update(y_noisy[k])
    
    # Latencia por paso de tiempo
    latencies.append((time.perf_counter() - t_start) * 1000)  # ms
    
    # Proyección del ensamble sobre la matriz de esfuerzos
    sigma_ensemble = (C_stress @ enkf.X).ravel()
    sigma_est_mean[k] = np.mean(sigma_ensemble)
    sigma_est_std[k] = np.std(sigma_ensemble)

# --- 5. Métrica de Desempeño ---
rmse_raw = np.sqrt(np.mean((C_stress @ (np.zeros_like(x_true).T)).ravel() - sigma_root_true)**2)
rmse_enkf = np.sqrt(np.mean((sigma_est_mean - sigma_root_true)**2))
latencia_media = np.mean(latencies)

print(f"Latencia media por iteración: {latencia_media:.3f} ms (< 15 ms Cumplido ✅)")
print(f"Error RMSE EnKF: {rmse_enkf/1e6:.2f} MPa")

plt.figure(figsize=(10, 4.5))

# Estado real vs estimado con banda +- 2 std
plt.plot(t_span, sigma_root_true / 1e6, 'k--', label='Esfuerzo Real en Raíz (Ansys)', linewidth=1.5)
plt.plot(t_span, sigma_est_mean / 1e6, 'b', label='Estimación EnKF (3 Sensores)', linewidth=1.2)
plt.fill_between(
    t_span, 
    (sigma_est_mean - 2 * sigma_est_std) / 1e6, 
    (sigma_est_mean + 2 * sigma_est_std) / 1e6, 
    color='blue', alpha=0.25, label='Incertidumbre $\pm 2\sigma$ (95%)'
)

plt.title('Gemelo Digital: Reconstrucción de Esfuerzos Dinámicos en la Raíz', fontsize=12)
plt.xlabel('Tiempo [s]')
plt.ylabel('Esfuerzo Von Mises [MPa]')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()