import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Cargar los datos exportados desde Ansys
# ---------------------------------------------------------
df = pd.read_csv(
    'AGARD_445_6_files/dp0/SYS/MECH/modal_data.csv',
    skiprows=1,
    header=None,
    names=['Mode', 'Freq_Hz', 'Force_UZ', 'Sens1_UZ', 'Sens2_UZ', 'Sens3_UZ'],
    skipinitialspace=True
)

modes = df['Mode'].values
freqs_hz = df['Freq_Hz'].values
omega = 2 * np.pi * freqs_hz  # Frecuencias en rad/s

# Modos en los puntos de interés (dirección Z)
phi_in = df['Force_UZ'].values        # Vector (m,)
phi_s1 = df['Sens1_UZ'].values
phi_s2 = df['Sens2_UZ'].values
phi_s3 = df['Sens3_UZ'].values

# Matriz de modos para los 3 sensores (3 x m)
phi_sens = np.vstack([phi_s1, phi_s2, phi_s3])
num_modes = len(modes)

# ---------------------------------------------------------
# 2. Definir Amortiguamiento Modal (\zeta)
# ---------------------------------------------------------
# Ratio de amortiguamiento (ejemplo: 2% para todos los modos)
zeta_val = 0.02
zeta = np.full(num_modes, zeta_val)

# ---------------------------------------------------------
# 3. Construir Matrices del Espacio de Estados (LTI ROM)
# ---------------------------------------------------------
O_m = np.zeros((num_modes, num_modes))
I_m = np.eye(num_modes)
Omega2 = np.diag(omega**2)
Gamma = np.diag(2 * zeta * omega)

# Matriz de Estado A (2m x 2m)
A = np.block([
    [O_m,     I_m],
    [-Omega2, -Gamma]
])

# Matriz de Entrada B (2m x 1)
B = np.block([
    [np.zeros((num_modes, 1))],
    [phi_in.reshape(-1, 1)]
])

# Matriz de Salida C (3 x 2m) para DESPLAZAMIENTO
C = np.block([phi_sens, np.zeros((3, num_modes))])
D = np.zeros((3, 1))

# Crear el sistema LTI en Scipy
sys_rom = signal.StateSpace(A, B, C, D)

# ---------------------------------------------------------
# 4. Simulación Temporal (Ejemplo: Fuerza Senoidal de Entrada)
# ---------------------------------------------------------
t = np.linspace(0, 5.0, 5000)  # 1 segundo, 1000 puntos
fuerza_frec = 15.0             # Fuerza senoidal a 15 Hz
u = np.sin(2 * np.pi * fuerza_frec * t)  # Entrada F(t) en N

# Resolver el sistema dinámico
t_out, y_out, _ = signal.lsim(sys_rom, U=u, T=t)

# ---------------------------------------------------------
# 5. Graficar la Respuesta de los Sensores
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(t_out, y_out[:, 0], label='Sensor 1 (Raíz/Medio)')
plt.plot(t_out, y_out[:, 1], label='Sensor 2 (Borde)')
plt.plot(t_out, y_out[:, 2], label='Sensor 3 (Punta)')
plt.title('Respuesta Temporal del ROM del Ala AGARD 445.6 en Python')
plt.xlabel('Tiempo [s]')
plt.ylabel('Desplazamiento Z [m]')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()