import numpy as np
from scipy.linalg import expm

class EnsembleKalmanFilter:
    def __init__(self, A, B, C_disp, Q_cov, R_cov, num_ensembles=50, dt=0.001):
        self.dt = dt
        self.N = num_ensembles
        self.n_x = A.shape[0]        # 40 estados modales
        self.n_y = C_disp.shape[0]   # 3 sensores de desplazamiento
        
        # Discretización exacta del sistema continuo (ZOH)
        self.Ad = expm(A * dt)
        self.Bd = np.linalg.pinv(A) @ (self.Ad - np.eye(self.n_x)) @ B
        self.C_meas = C_disp
        
        self.Q = Q_cov  # Ruido de proceso (40x40)
        self.R = R_cov  # Ruido de medición de los sensores (3x3)
        
        # Generar ensamble inicial X_0 (40 x N) alrededor del origen
        self.X = np.random.multivariate_normal(
            mean=np.zeros(self.n_x), 
            cov=np.eye(self.n_x) * 1e-8, 
            size=self.N
        ).T

    def predict(self, u_k):
        """Paso de tiempo: propaga cada miembro del ensamble con la dinámica discreta"""
        process_noise = np.random.multivariate_normal(
            mean=np.zeros(self.n_x), cov=self.Q, size=self.N
        ).T
        self.X = self.Ad @ self.X + self.Bd * u_k + process_noise

    def update(self, y_k):
        """Paso de medición: corrige el ensamble comparando las lecturas reales ruidosas"""
        meas_noise = np.random.multivariate_normal(
            mean=np.zeros(self.n_y), cov=self.R, size=self.N
        ).T
        
        # Predicción de mediciones para cada miembro del ensamble
        Y_pred = self.C_meas @ self.X + meas_noise
        
        # Medias del ensamble
        x_mean = np.mean(self.X, axis=1, keepdims=True)
        y_mean = np.mean(Y_pred, axis=1, keepdims=True)
        
        # Matrices de desviación (anomalías)
        A_x = self.X - x_mean
        A_y = Y_pred - y_mean
        
        # Covarianzas muestrales
        P_xy = (A_x @ A_y.T) / (self.N - 1)
        P_yy = (A_y @ A_y.T) / (self.N - 1)
        
        # Ganancia de Kalman
        K = P_xy @ np.linalg.inv(P_yy)
        
        # Actualización del ensamble con el residuo de innovación
        y_k_vec = np.array(y_k).reshape(-1, 1)
        self.X = self.X + K @ (y_k_vec - Y_pred)

    @property
    def state_estimate(self):
        """Media del ensamble para el estado estimado x_hat"""
        return np.mean(self.X, axis=1)

    @property
    def state_std(self):
        """Banda de incertidumbre (desviación estándar por estado)"""
        return np.std(self.X, axis=1)