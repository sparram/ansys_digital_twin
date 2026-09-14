import numpy as np
from ansys.mapdl import reader as pymapdl_reader


class ModalROMLoader:

    def __init__(
        self, rst_path: str, num_modes: int = 8, damping_ratio: float = 0.02
    ):
        self.rst_path = rst_path
        self.num_modes = num_modes
        self.damping_ratio = damping_ratio
        self.result_file = pymapdl_reader.read_binary(rst_path)

    def build_state_space(self):
        freqs_hz = self.result_file.time_values[: self.num_modes]
        omega = 2 * np.pi * freqs_hz

        M_modal = np.eye(self.num_modes)
        K_modal = np.diag(omega**2)
        C_modal = np.diag(2 * self.damping_ratio * omega)

        I = np.eye(self.num_modes)
        Z = np.zeros((self.num_modes, self.num_modes))

        A_cont = np.vstack((np.hstack((Z, I)), np.hstack((-K_modal, -C_modal))))
        B_cont = np.vstack((Z, M_modal))

        return A_cont, B_cont, freqs_hz

    def extract_mode_shapes(self):
        nnum, _ = self.result_file.nodal_displacement(0)
        num_nodes = len(nnum)
        dof_per_node = 3
        total_dofs = num_nodes * dof_per_node

        Phi = np.zeros((total_dofs, self.num_modes))
        for i in range(self.num_modes):
            _, disp = self.result_file.nodal_displacement(i)
            disp_xyz = disp[:, :dof_per_node].copy()
            disp_xyz = np.nan_to_num(disp_xyz, nan=0.0, posinf=0.0, neginf=0.0)
            disp_xyz[np.abs(disp_xyz) > 1e20] = 0.0
            Phi[:, i] = disp_xyz.flatten()

        active_dof = int(
            np.argmax([
                np.max(np.abs(Phi[0::3, :])),
                np.max(np.abs(Phi[1::3, :])),
                np.max(np.abs(Phi[2::3, :])),
            ])
        )

        return Phi, num_nodes, active_dof, self.result_file.grid.points