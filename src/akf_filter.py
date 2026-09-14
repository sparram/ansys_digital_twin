import numpy as np
from scipy.linalg import expm


class AugmentedKalmanFilter:

    def __init__(
        self, A_cont, B_cont, Phi, sensor_dofs, dt=0.001, std_noise=1e-4
    ):
        self.dt = dt
        self.Phi = Phi
        self.sensor_dofs = sensor_dofs
        self.num_modes = B_cont.shape[1]
        self.nx = A_cont.shape[0]
        self.np_sens = len(sensor_dofs)

        # Zero-Order Hold (ZOH) Discretization
        M_disc = np.zeros(
            (self.nx + self.num_modes, self.nx + self.num_modes)
        )
        M_disc[: self.nx, : self.nx] = A_cont
        M_disc[: self.nx, self.nx :] = B_cont
        M_exp = expm(M_disc * dt)

        self.Ad = M_exp[: self.nx, : self.nx]
        self.Bd = M_exp[: self.nx, self.nx :]

        # Augmented system setup
        Phi_s = Phi[sensor_dofs, :]
        C = np.hstack((Phi_s, np.zeros((self.np_sens, self.num_modes))))

        self.n_aug = self.nx + self.num_modes
        self.A_aug = np.zeros((self.n_aug, self.n_aug))
        self.A_aug[: self.nx, : self.nx] = self.Ad
        self.A_aug[: self.nx, self.nx :] = self.Bd
        self.A_aug[self.nx :, self.nx :] = np.eye(self.num_modes)

        self.C_aug = np.hstack((C, np.zeros((self.np_sens, self.num_modes))))

        # =====================================================================
        # AJUSTE Y REGULARIZACIÓN DE COVARIANZA Q (Evita arrugas espaciales)
        # =====================================================================
        self.Q = np.eye(self.n_aug) * 1e-5

        # Asignamos incertidumbre decreciente a las fuerzas modales:
        # Modos 1 y 2 (Flexión/Torsión suave) -> Alta libertad de estimación
        # Modos 4 al 8 (Modos altos)          -> Penalizados para evitar 'ondas'
        q_force_weights = np.array(
            [1e1, 1e1, 1e-1, 1e-3, 1e-4, 1e-5, 1e-5, 1e-5]
        )

        for i in range(self.num_modes):
            self.Q[self.nx + i, self.nx + i] = q_force_weights[i]
            
        self.R = np.eye(self.np_sens) * (std_noise**2)

        self.x_hat = np.zeros((self.n_aug, 1))
        self.P = np.eye(self.n_aug) * 1e-2

    def step(self, z_k):
        x_pred = self.A_aug @ self.x_hat
        P_pred = self.A_aug @ self.P @ self.A_aug.T + self.Q

        S = self.C_aug @ P_pred @ self.C_aug.T + self.R
        K = P_pred @ self.C_aug.T @ np.linalg.inv(S)

        innovation = z_k.reshape(-1, 1) - (self.C_aug @ x_pred)
        self.x_hat = x_pred + K @ innovation
        self.P = (np.eye(self.n_aug) - K @ self.C_aug) @ P_pred

        q_est = self.x_hat[: self.num_modes, 0]
        f_est = self.x_hat[self.nx :, 0]
        return q_est, f_est

    def reconstruct_fields(self, q_est):
        return self.Phi @ q_est