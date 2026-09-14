import numpy as np
import pyvista as pv


def animate_digital_twin_3d(
    coords_orig,
    u_truth,
    u_est_hist,
    sensor_nodes,
    active_dof,
    dt=0.001,
    scale=15.0,
    save_gif=True,
    gif_name="digital_twin_animation.gif",
):

    num_nodes = coords_orig.shape[0]
    n_steps = u_truth.shape[1]

    # Crear mallas iniciales en t = 0
    u_real_0 = u_truth[:, 0].reshape((num_nodes, 3))
    u_est_0 = u_est_hist[:, 0].reshape((num_nodes, 3))

    mesh_real = pv.PolyData(coords_orig + scale * u_real_0)
    mesh_real["Uz (m)"] = u_real_0[:, active_dof]

    mesh_est = pv.PolyData(coords_orig + scale * u_est_0)
    mesh_est["Uz (m)"] = u_est_0[:, active_dof]

    surf_real = mesh_real.delaunay_2d()
    surf_est = mesh_est.delaunay_2d()

    # Configurar plotter de PyVista
    plotter = pv.Plotter(shape=(1, 2), window_size=[1200, 500])

    plotter.subplot(0, 0)
    plotter.add_text("ANSYS Ground Truth (Real Wind Dynamic)", font_size=10)
    plotter.add_mesh(
        surf_real,
        scalars="Uz (m)",
        cmap="jet",
        show_edges=True,
        edge_color="black",
        line_width=0.5,
        clim=[-0.015, 0.015],
    )

    plotter.subplot(0, 1)
    plotter.add_text("Digital Twin (AKF Real-Time Sensing)", font_size=10)
    plotter.add_mesh(
        surf_est,
        scalars="Uz (m)",
        cmap="jet",
        show_edges=True,
        edge_color="black",
        line_width=0.5,
        clim=[-0.015, 0.015],
    )

    sensors_pv = pv.PolyData(coords_orig[sensor_nodes])
    plotter.add_mesh(
        sensors_pv, color="black", point_size=16, render_points_as_spheres=True
    )

    plotter.link_views()
    plotter.camera_position = "iso"

    if save_gif:
        plotter.open_gif(gif_name, fps=30)
        print(f"-> Generando animación GIF: {gif_name}...")

    # Submuestreo de cuadros para acelerar el renderizado (1 de cada 10 pasos)
    frame_step = 10
    for k in range(0, n_steps, frame_step):
        u_r = u_truth[:, k].reshape((num_nodes, 3))
        u_e = u_est_hist[:, k].reshape((num_nodes, 3))

        # Actualizar coordenadas deformadas
        surf_real.points = coords_orig + scale * u_r
        surf_real["Uz (m)"] = u_r[:, active_dof]

        surf_est.points = coords_orig + scale * u_e
        surf_est["Uz (m)"] = u_e[:, active_dof]

        # Actualizar posición de los sensores
        sensors_pv.points = coords_orig[sensor_nodes] + scale * u_e[sensor_nodes]

        if save_gif:
            plotter.write_frame()
        else:
            plotter.render()

    plotter.close()
    print("-> Animación 3D completada exitosamente.")