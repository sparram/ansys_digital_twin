import numpy as np
import pyvista as pv


def render_digital_twin_3d(
    coords_orig,
    u_truth,
    u_est_hist,
    sensor_nodes,
    dof_virtual,
    active_dof,
    scale=10.0,
):
    idx_max = np.argmax(np.abs(u_truth[dof_virtual, :]))
    num_nodes = coords_orig.shape[0]

    u_real_t = u_truth[:, idx_max].reshape((num_nodes, 3))
    u_est_t = u_est_hist[:, idx_max].reshape((num_nodes, 3))

    pos_real = coords_orig + scale * u_real_t
    pos_est = coords_orig + scale * u_est_t

    mesh_real = pv.PolyData(pos_real)
    mesh_real["Uz (m)"] = u_real_t[:, active_dof]

    mesh_est = pv.PolyData(pos_est)
    mesh_est["Uz (m)"] = u_est_t[:, active_dof]

    surf_real = mesh_real.delaunay_2d()
    surf_est = mesh_est.delaunay_2d()

    plotter = pv.Plotter(shape=(1, 2), window_size=[1200, 500])

    plotter.subplot(0, 0)
    plotter.add_text("ANSYS Ground Truth (Real)", font_size=11)
    plotter.add_mesh(
        surf_real,
        scalars="Uz (m)",
        cmap="jet",
        show_edges=True,
        edge_color="black",
        line_width=0.5,
    )

    plotter.subplot(0, 1)
    plotter.add_text("Digital Twin (AKF Reconstruction)", font_size=11)
    plotter.add_mesh(
        surf_est,
        scalars="Uz (m)",
        cmap="jet",
        show_edges=True,
        edge_color="black",
        line_width=0.5,
    )

    sensors_pv = pv.PolyData(pos_est[sensor_nodes])
    plotter.add_mesh(
        sensors_pv, color="black", point_size=16, render_points_as_spheres=True
    )

    plotter.link_views()
    plotter.camera_position = "iso"
    plotter.show()