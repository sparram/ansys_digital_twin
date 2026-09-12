import numpy as np
import pandas as pd
from scipy.signal import StateSpace, lsim

# 1. Cargar datos modales con esfuerzos
df = pd.read_csv(
    'AGARD_445_6_files/dp0/SYS/MECH/modal_data_with_stress.csv',
    skiprows=1,
    header=None,
    names=['Mode', 'Freq_Hz', 'Force_UZ', 'Sens1_UZ', 'Sens2_UZ', 'Sens3_UZ', 'Root_SEQV'],
    skipinitialspace=True
)

num_modes = len(df)
freqs_hz = df['Freq_Hz'].values
omegas = 2 * np.pi * freqs_hz  # Frecuencias angulares [rad/s]
zeta = 0.02                    # Amortiguamiento modal (2%)

# 2. Matriz de Estado A (40x40)
Omega2 = np.diag(omegas**2)
Gamma = np.diag(2 * zeta * omegas)
A = np.block([
    [np.zeros((num_modes, num_modes)), np.eye(num_modes)],
    [-Omega2, -Gamma]
])

# 3. Matriz de Entrada B (40x1)
phi_force = df['Force_UZ'].values.reshape(-1, 1)
B = np.vstack([np.zeros((num_modes, 1)), phi_force])

# 4. Matriz de Salida C (4x40)
phi_sens = np.vstack([df['Sens1_UZ'].values, df['Sens2_UZ'].values, df['Sens3_UZ'].values])
psi_root = df['Root_SEQV'].values.reshape(1, -1)

C_disp = np.block([phi_sens, np.zeros((3, num_modes))])
C_stress = np.block([psi_root, np.zeros((1, num_modes))])
C = np.vstack([C_disp, C_stress])

# 5. Matriz de Transmisión Directa D (4x1)
D = np.zeros((4, 1))

# Sistema continuo LTI
sys_lti = StateSpace(A, B, C, D)