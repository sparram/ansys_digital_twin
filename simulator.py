# simulator.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import solve_ivp

def construir_espacio_estados(V, M_m, K_m, C_m, GAF, num_modos=20, b_ref=0.25, k_frec=0.1):
    """Ensambla la matriz A corregida para una velocidad V del viento."""
    q_dyn = 0.5 * 1.225 * (V**2)
    K_total = K_m + q_dyn * np.real(GAF)
    C_total = C_m + q_dyn * (b_ref / (V * k_frec)) * np.imag(GAF)
    
    I, Z = np.eye(num_modos), np.zeros((num_modos, num_modos))
    M_inv = np.linalg.inv(M_m)
    return np.vstack((np.hstack((Z, I)), np.hstack((-M_inv @ K_total, -M_inv @ C_total))))

def correr_animacion_túnel(A, Phi, aerogrid, V, t_final=2.0, n_cuadros=100, num_modos=20):
    """Resuelve la EDO y guarda la animación del ala en un archivo GIF."""
    plt.ioff()
    t_eval = np.linspace(0, t_final, n_cuadros)
    x0, fuerza = np.zeros(2 * num_modos), np.zeros(2 * num_modos)
    fuerza[num_modos], fuerza[num_modos + 1] = 12.0, 4.0 # Empuje constante del viento
    
    sol = solve_ivp(lambda t, x: A @ x + fuerza, [0, t_final], x0, t_eval=t_eval)
    deflexion_Z = Phi @ sol.y[:num_modos, :] * 2.0 # Factor de escala visual = 2.0
    
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    def obtener_polys(t_idx):
        polys, cols = [], []
        for idx in range(aerogrid['n']):
            c = aerogrid['l'][idx]
            x1, y1, z1 = aerogrid['offset_P1'][idx, 0] - 0.25*c, aerogrid['offset_P1'][idx, 1], deflexion_Z[idx, t_idx]
            x2, y2, z2 = aerogrid['offset_P3'][idx, 0] - 0.25*c, aerogrid['offset_P3'][idx, 1], deflexion_Z[idx, t_idx]
            polys.append([[x1, y1, z1], [x2, y2, z2], [x2+c, y2, z2], [x1+c, y1, z1]])
            cols.append(abs(z1))
        return polys, cols

    polys_ini, cols_ini = obtener_polys(0)
    coleccion = Poly3DCollection(polys_ini, cmap='coolwarm', edgecolor='navy', linewidths=0.3)
    z_max = np.max(abs(deflexion_Z))
    coleccion.set_array(np.array(cols_ini))
    coleccion.set_clim(0, z_max + 0.01)
    ax.add_collection3d(coleccion)
    
    ax.set_xlim([0, 1.2]); ax.set_ylim([0, 0.85]); ax.set_zlim([-z_max*1.3, z_max*1.3])
    ax.set_box_aspect([1.2, 0.85, 0.4]); ax.view_init(elev=20, azim=-135)
    ax.set_title(f"Módulo Modularizado - Túnel de Viento V = {V} m/s")

    def actualizar(frame):
        global coleccion
        if len(ax.collections) > 0: ax.collections[0].remove()
        p, c = obtener_polys(frame)
        col = Poly3DCollection(p, cmap='coolwarm', edgecolor='navy', linewidths=0.3)
        col.set_array(np.array(c)); col.set_clim(0, z_max + 0.01)
        ax.add_collection3d(col)
        return col,

    ani = animation.FuncAnimation(fig, actualizar, frames=range(n_cuadros), blit=False)
    ani.save("gemelo_digital_modular.gif", writer='pillow', fps=20)
    plt.ion()
    print("¡Simulación terminada! Animación exportada con éxito como 'gemelo_digital_modular.gif'")
