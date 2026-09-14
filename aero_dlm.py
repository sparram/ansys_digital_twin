# aero_dlm.py
import numpy as np
from panelaero import DLM

def generar_malla_agard(n_cuerda=4, n_span=10):
    """Construye la red de paneles (aerogrid) estándar de la NASA para la AGARD 445.6."""
    cuerda_raiz, cuerda_punta, envergadura, sweep_deg = 0.5588, 0.3302, 0.762, 45.0
    p1_raiz_ataque = np.array([0.0, 0.0, 0.0])
    p2_raiz_salida = np.array([cuerda_raiz, 0.0, 0.0])
    x_punta_ataque = envergadura * np.tan(np.radians(sweep_deg))
    p3_punta_ataque = np.array([x_punta_ataque, envergadura, 0.0])
    p4_punta_salida = np.array([x_punta_ataque + cuerda_punta, envergadura, 0.0])
    
    total_paneles = n_cuerda * n_span
    aerogrid = {
        'n': total_paneles, 'ID': np.arange(total_paneles),
        'l': np.zeros(total_paneles), 'b': np.zeros(total_paneles), 'A': np.zeros(total_paneles),
        'offset_j': np.zeros((total_paneles, 3)), 'offset_P1': np.zeros((total_paneles, 3)),
        'offset_P3': np.zeros((total_paneles, 3)), 'offset_l': np.zeros((total_paneles, 3)),
        'N': np.zeros((total_paneles, 3)),
    }
    
    idx = 0
    dy = envergadura / n_span
    for j in range(n_span):
        eta_izq, eta_der = j / n_span, (j + 1) / n_span
        x_le_izq = (p1_raiz_ataque + eta_izq * (p3_punta_ataque - p1_raiz_ataque))[0]
        x_le_der = (p1_raiz_ataque + eta_der * (p3_punta_ataque - p1_raiz_ataque))[0]
        x_te_izq = (p2_raiz_salida + eta_izq * (p4_punta_salida - p2_raiz_salida))[0]
        x_te_der = (p2_raiz_salida + eta_der * (p4_punta_salida - p2_raiz_salida))[0]
        cuerda_izq, cuerda_der = x_te_izq - x_le_izq, x_te_der - x_le_der
        y_izq, y_der = eta_izq * envergadura, eta_der * envergadura
        y_centro = (y_izq + y_der) / 2.0
        
        for i in range(n_cuerda):
            xi_ant, xi_post = i / n_cuerda, (i + 1) / n_cuerda
            xi_centro_75 = xi_ant + 0.75 * (xi_post - xi_ant)
            xi_vortice_25 = xi_ant + 0.25 * (xi_post - xi_ant)
            x_centro = (x_le_izq + xi_centro_75 * cuerda_izq + x_le_der + xi_centro_75 * cuerda_der) / 2.0
            x_p1 = x_le_izq + xi_vortice_25 * cuerda_izq
            x_p3 = x_le_der + xi_vortice_25 * cuerda_der
            x_l = (x_p1 + x_p3) / 2.0
            cuerda_panel = (cuerda_izq / n_cuerda + cuerda_der / n_cuerda) / 2.0
            
            aerogrid['offset_j'][idx] = [x_centro, y_centro, 0.0]
            aerogrid['offset_P1'][idx], aerogrid['offset_P3'][idx] = [x_p1, y_izq, 0.0], [x_p3, y_der, 0.0]
            aerogrid['offset_l'][idx] = [x_l, y_centro, 0.0]
            aerogrid['l'][idx], aerogrid['b'][idx], aerogrid['A'][idx] = cuerda_panel, dy, cuerda_panel * dy
            aerogrid['N'][idx] = [0.0, 0.0, 1.0]
            idx += 1
    return aerogrid

def calcular_matrices_gaf(aerogrid, Mach, k_frec, envergadura=0.762, num_modos=20):
    """Ejecuta el DLM y acopla analíticamente los modos para obtener las GAF."""
    AIC = DLM.calc_Qjj(aerogrid, Mach, k_frec)
    Phi = np.zeros((aerogrid['n'], num_modos))
    for idx in range(aerogrid['n']):
        y_norm = aerogrid['offset_j'][idx, 1] / envergadura
        x_local = aerogrid['offset_j'][idx, 0]
        Phi[idx, 0] = y_norm**2
        Phi[idx, 1] = y_norm * (x_local - 0.2)
        for m in range(2, num_modos):
            Phi[idx, m] = 0.01 * np.sin(m * y_norm * np.pi)
    GAF = Phi.T @ AIC @ Phi
    return GAF, Phi
