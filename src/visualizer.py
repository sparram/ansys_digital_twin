import numpy as np
import pyvista as pv


def animate_digital_twin_3d(
    coords_orig,
    u_truth,
    u_est_hist,
    sensor_nodes,
    active_dof,
    vtk_grid=None,
    dt=0.001,
    scale=15.0,
    save_gif=True,
    gif_name="digital_twin_animation.gif",
):
    """Renderiza la comparación 3D en tiempo real.

    Soporta nubes de puntos simples (2D) y mallas industriales 3D (Shell/Solid
    vía VTK).
    """
    num_nodes = coords_orig.shape[0]
    n_steps = u_truth.shape[1]

    # 1. Detección de topología: Malla 3D Industrial vs Nube 2D Básica
    if vtk_grid is not None:
        mesh_real = vtk_grid.copy(deep=True)
        mesh_est = vtk_grid.copy(deep=True)
        is_native_grid = True
    else:
        mesh_real = pv.PolyData(coords_orig)
        mesh_est = pv.PolyData(coords_orig)
        is_native_grid = False

    # Estado inicial t = 0
    u_r0 = u_truth[:, 0].reshape((num_nodes, 3))
    u_e0 = u_est_hist[:, 0].reshape((num_nodes, 3))

    mesh_real.points = coords_orig + scale * u_r0
    mesh_real["Uz (m)"] = u_r0[:, active_dof]

    mesh_est.points = coords_orig + scale * u_e0
    mesh_est["Uz (m)"] = u_e0[:, active_dof]

    if not is_native_grid:
        surf_real = mesh_real.delaunay_2d()
        surf_est = mesh_est.delaunay_2d()
    else:
        surf_real = mesh_real
        surf_est = mesh_est

    # 2. Barra de color vertical única a la derecha
    custom_scalar_bar = dict(
        title="Uz (m)",
        vertical=True,
        position_x=0.82,
        position_y=0.20,
        width=0.08,
        height=0.60,
        fmt="%.2e",
        title_font_size=10,
        label_font_size=8,
    )

    plotter = pv.Plotter(shape=(1, 2), window_size=[1200, 550])

    # Panel Izquierdo: Realidad
    plotter.subplot(0, 0)
    plotter.add_text("ANSYS Ground Truth (Real Dynamic)", font_size=10)
    plotter.add_mesh(
        surf_real,
        scalars="Uz (m)",
        cmap="jet",
        show_edges=True,
        edge_color="black",
        line_width=0.5,
        clim=[-0.015, 0.015],
        show_scalar_bar=False,
    )

    # Panel Derecho: Gemelo Digital
    plotter.subplot(0, 1)
    plotter.add_text("Digital Twin (AKF Real-Time)", font_size=10)
    plotter.add_mesh(
        surf_est,
        scalars="Uz (m)",
        cmap="jet",
        show_edges=True,
        edge_color="black",
        line_width=0.5,
        clim=[-0.015, 0.015],
        scalar_bar_args=custom_scalar_bar,
    )

    # Marcadores de Sensores
    sensors_pv = pv.PolyData(coords_orig[sensor_nodes])
    plotter.add_mesh(
        sensors_pv, color="black", point_size=16, render_points_as_spheres=True
    )

    plotter.link_views()
    plotter.camera_position = "iso"

    if save_gif:
        plotter.open_gif(gif_name, fps=30)
        print(f"-> Generando animación GIF: {gif_name}...")

    # Loop de actualización dinámica
    frame_step = 10
    for k in range(0, n_steps, frame_step):
        u_r = u_truth[:, k].reshape((num_nodes, 3))
        u_e = u_est_hist[:, k].reshape((num_nodes, 3))

        surf_real.points = coords_orig + scale * u_r
        surf_real["Uz (m)"] = u_r[:, active_dof]

        surf_est.points = coords_orig + scale * u_e
        surf_est["Uz (m)"] = u_e[:, active_dof]

        sensors_pv.points = coords_orig[sensor_nodes] + scale * u_e[sensor_nodes]

        if save_gif:
            plotter.write_frame()
        else:
            plotter.render()

    plotter.close()
    print("-> Animación completada exitosamente.")