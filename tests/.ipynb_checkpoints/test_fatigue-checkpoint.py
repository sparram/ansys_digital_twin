import numpy as np
from src.fatigue import FatigueEstimator

def test_fatigue_zero_damage():
    estimator = FatigueEstimator()
    flat_signal = np.ones(100) * 1e6  # Esfuerzo constante (sin fluctuación)
    damage = estimator.compute_cumulative_damage(flat_signal)
    assert damage == 0.0
    assert estimator.remaining_useful_life_percent(damage) == 100.0

def test_fatigue_damage_accumulation():
    estimator = FatigueEstimator(C_sn=1e6, m_sn=3.0)
    t = np.linspace(0, 2, 500)
    stress_signal = 100e6 * np.sin(2 * np.pi * 10 * t)  # Amplitud de 100 MPa
    
    damage = estimator.compute_cumulative_damage(stress_signal)
    assert damage > 0.0
    assert estimator.remaining_useful_life_percent(damage) < 100.0