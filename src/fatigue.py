import numpy as np

class FatigueEstimator:
    def __init__(self, C_sn: float = 2.0e12, m_sn: float = 3.0):
        """
        Modelo de fatiga basado en la curva S-N: N_i = C_sn / (S_amp^m_sn)
        
        Parámetros:
            C_sn: Coeficiente de resistencia a la fatiga del material.
            m_sn: Exponente de la curva S-N.
        """
        self.C_sn = C_sn
        self.m_sn = m_sn

    def extract_amplitudes(self, stress_signal: np.ndarray) -> np.ndarray:
        """Extrae las amplitudes de esfuerzo entre picos y valles consecutivos."""
        diffs = np.diff(stress_signal)
        peaks = (diffs[:-1] * diffs[1:] < 0)
        peak_indices = np.where(peaks)[0] + 1
        
        if len(peak_indices) < 2:
            return np.array([])
            
        peak_values = stress_signal[peak_indices]
        return np.abs(np.diff(peak_values)) / 2.0

    def compute_cumulative_damage(self, stress_history_pa: np.ndarray) -> float:
        """
        Calcula el daño acumulado D = sum(n_i / N_i) usando Palmgren-Miner.
        D >= 1.0 representa falla estructural.
        """
        stress_mpa = np.asanyarray(stress_history_pa) / 1e6
        amplitudes = self.extract_amplitudes(stress_mpa)
        
        if len(amplitudes) == 0:
            return 0.0
        
        # Inversa del número de ciclos permisibles N_i por cada ciclo detectado
        n_i = 1.0
        N_i = self.C_sn / (amplitudes ** self.m_sn)
        incremental_damage = np.sum(n_i / N_i)
        
        return float(incremental_damage)

    def remaining_useful_life_percent(self, cumulative_damage: float) -> float:
        """Retorna el porcentaje de Vida Útil Restante (RUL)."""
        return max(0.0, float((1.0 - cumulative_damage) * 100.0))