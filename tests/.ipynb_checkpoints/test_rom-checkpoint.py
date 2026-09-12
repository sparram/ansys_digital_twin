import os
import pytest
import numpy as np
from src.rom import load_lti_matrices

@pytest.fixture
def csv_path():
    return os.path.join(os.path.dirname(__file__), '../AGARD_445_6_files/dp0/SYS/MECH/modal_data_with_stress.csv')

def test_matrix_dimensions(csv_path):
    """Verifica las dimensiones de las matrices del modelo reducido (20 modos -> 40 estados)."""
    A, B, C, D = load_lti_matrices(csv_path)
    assert A.shape == (40, 40)
    assert B.shape == (40, 1)
    assert C.shape == (4, 40)
    assert D.shape == (4, 1)

def test_system_stability(csv_path):
    """Verifica que el sistema sea estable (autovalores con parte real negativa o cero)."""
    A, _, _, _ = load_lti_matrices(csv_path)
    eigenvalues = np.linalg.eigvals(A)
    assert np.all(np.real(eigenvalues) <= 0)