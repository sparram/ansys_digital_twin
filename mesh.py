import numpy as np
from ansys.mapdl import reader as pymapdl_reader

# 1. Cargar el archivo de resultados (.rst) de tu análisis Modal
path_to_rst = r"AGARD_445_6_files/dp0/SYS/MECH/file.rst"
result = pymapdl_reader.read_binary(path_to_rst)

# 2. ¡Aquí está tu malla automáticamente!
# 'grid' contiene todos los nodos, elementos y la topología 3D del ala
malla_ala = result.grid
print(f"Nodos totales en la malla: {malla_ala.points.shape[0]}")

# 3. Graficar el Ala en su estado original (Estático)
# Esto abrirá una ventana interactiva 3D en Python para rotar y ver el ala
malla_ala.plot(show_edges=True, color="w")

# 4. Extraer los desplazamientos del MODO 1 y visualizar la deformación
# result.nodal_displacement(0) trae el primer modo calculado
nodal_shadow, disp_modo1 = result.nodal_displacement(0)

# Graficar el modo de vibración usando un mapa de calor de deformación
# 'displacement_factor' amplifica visualmente la deformación para ver el modo claramente
result.plot_nodal_displacement(
    rnum=0,  # 0 significa el primer modo (Set 1,1 en APDL)
    component="Z",  # Nos interesa el desplazamiento vertical (Z) para aeroelasticidad
    displacement_factor=0.5,  # Factor de escala visual
    show_edges=True,
)
