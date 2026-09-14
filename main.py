# main.py
import structural_rom
import aero_dlm
import simulator

# 1. Configuración de rutas y parámetros de control
path_rst_ansys = r"AGARD_445_6_files/dp0/SYS/MECH/file.rst"
VELOCIDAD_VIENTO = 30.0  # m/s

print("=== INICIANDO TUNEL DE VIENTO MODULAR ===")

# 2. Ejecutar Módulo Estructural (Ansys)
frecuencias, M_m, K_m, C_m = structural_rom.cargar_estructura_ansys(path_rst_ansys)

# 3. Ejecutar Módulo Aerodinámico (Geometría + DLM)
aerogrid = aero_dlm.generar_malla_agard()
GAF, Phi_aero = aero_dlm.calcular_matrices_gaf(aerogrid, Mach=0.496, k_frec=0.1)

# 4. Construir Espacio de Estados y Correr Simulación Dinámica
A_sistema = simulator.construir_espacio_estados(VELOCIDAD_VIENTO, M_m, K_m, C_m, GAF)
simulator.correr_animacion_túnel(A_sistema, Phi_aero, aerogrid, V=VELOCIDAD_VIENTO)

print("=== PROCESO COMPLETADO EN MILISEGUNDOS ===")
