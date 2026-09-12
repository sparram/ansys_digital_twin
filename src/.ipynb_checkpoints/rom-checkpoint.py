import numpy as np
import pandas as pd


def load_lti_matrices(csv_path: str, zeta: float = 0.02):
    """Carga los datos modales y ensambla las matrices continuas LTI (A, B, C, D).

    Parámetros:
        csv_path (str): Ruta al archivo modal_data_with_stress.csv.
        zeta (float): Amortiguamiento modal (0.02 = 2%).

    Retorna:
        tuple: (A, B, C, D) matrices del sistema en formato numpy ndarray.
    """
    # 1. Leer CSV de datos modales
    df = pd.read_csv(
        csv_path,
        skiprows=1,
        header=None,
        names=[
            'Mode',
            'Freq_Hz',
            'Force_UZ',
            'Sens1_UZ',
            'Sens2_UZ',
            'Sens3_UZ',
            'Root_SEQV',
        ],
        skipinitialspace=True,
    )

    num_modes = len(df)
    freqs_hz = df['Freq_Hz'].values
    omegas = 2 * np.pi * freqs_hz  # Convertir Frecuencia Hz a Frecuencia Angular rad/s

    # 2. Ensamblar Matriz de Estado A (2m x 2m)
    Omega2 = np.diag(omegas**2)
    Gamma = np.diag(2 * zeta * omegas)
    A = np.block([
        [np.zeros((num_modes, num_modes)), np.eye(num_modes)],
        [-Omega2, -Gamma],
    ])

    # 3. Ensamblar Matriz de Entrada B (2m x 1)
    phi_force = df['Force_UZ'].values.reshape(-1, 1)
    B = np.vstack([np.zeros((num_modes, 1)), phi_force])

    # 4. Ensamblar Matriz de Salida C Expandida (4 x 2m)
    phi_sens = np.vstack(
        [df['Sens1_UZ'].values, df['Sens2_UZ'].values, df['Sens3_UZ'].values]
    )
    psi_root = df['Root_SEQV'].values.reshape(1, -1)

    C_disp = np.block([phi_sens, np.zeros((3, num_modes))])
    C_stress = np.block([psi_root, np.zeros((1, num_modes))])
    C = np.vstack([C_disp, C_stress])

    # 5. Matriz de Feedthrough D (4 x 1)
    D = np.zeros((4, 1))

    return A, B, C, D