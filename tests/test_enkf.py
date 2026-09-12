import os
import pytest
import time
import numpy as np
from src.rom import load_lti_matrices
from src.enkf import EnsembleKalmanFilter

@pytest.fixture
def enkf_instance():
    csv_path = os.path.join(os.path.dirname(__file__), '../AGARD_445_6_files/dp0/SYS/MECH/modal_data_with_stress.csv')
    A, B, C, D = load_lti_matrices(csv_path)
    C_disp = C[:3, :]
    Q = np.eye(40) * 1e-7
    R = np.eye(3) * (1.5e-6)**2
    return EnsembleKalmanFilter(A, B, C_disp, Q, R, num_ensembles=50, dt=0.001)

def test_enkf_dimensions(enkf_instance):
    """Verifica el tamaño del ensamble y del vector de estado estimado."""
    assert enkf_instance.X.shape == (40, 50)
    assert len(enkf_instance.state_estimate) == 40

def test_enkf_step_execution(enkf_instance):
    """Valida la ejecución del ciclo predicción-actualización sin valores NaN."""
    u_k = 10.0
    y_k = np.array([1e-4, 2e-4, 1.5e-4])
    
    enkf_instance.predict(u_k)
    enkf_instance.update(y_k)
    
    assert not np.isnan(enkf_instance.state_estimate).any()

def test_latency_constraint(enkf_instance):
    """Garantiza el requisito de tiempo real (< 15 ms por paso de integración)."""
    u_k = 5.0
    y_k = np.array([1e-4, 2e-4, 1.5e-4])
    
    t_start = time.perf_counter()
    enkf_instance.predict(u_k)
    enkf_instance.update(y_k)
    latency_ms = (time.perf_counter() - t_start) * 1000
    
    assert latency_ms < 15.0