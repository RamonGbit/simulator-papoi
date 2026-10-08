"""
Módulo 2.1: Análisis del Consumo y Temperatura de Servidores (Modelos de una variable).
Utiliza cálculo diferencial simbólico con SymPy para determinar puntos estacionarios
(máximos, mínimos) y puntos de inflexión (umbral crítico de enfriamiento de emergencia).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import sympy as sp


@dataclass
class ThermalCriticalPoints:
    """Contenedor de resultados analíticos del análisis térmico de 1 variable."""
    expression_latex: str
    first_derivative_latex: str
    second_derivative_latex: str
    stationary_points: List[Dict[str, Any]]
    inflection_points: List[Dict[str, Any]]
    emergency_threshold_load: float
    emergency_threshold_temp: float
    is_superlinear_risk: bool


class ThermalAnalyzer1D:
    """
    Analizador Térmico Simbólico basado en Diferenciación (SymPy).
    
    Modela la temperatura T en función de la carga de trabajo L in [0, 1]:
        T(L) = T_0 + c_1 * L - c_2 * L^2 + c_3 * L^3
        
    Calcula:
        1. dT/dL = 0  -> Puntos estacionarios (extremos locales)
        2. d^2T/dL^2 = 0 -> Punto de inflexión (aceleración térmica crítica)
    """

    def __init__(self, t_ambient: float = 22.0, c1: float = 65.0, c2: float = 54.0, c3: float = 30.0):
        self.t_ambient = float(t_ambient)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.c3 = float(c3)

        self.L = sp.Symbol('L', real=True)
        # Función fenomenológica de temperatura
        self.T_expr = self.t_ambient + self.c1 * self.L - self.c2 * (self.L ** 2) + self.c3 * (self.L ** 3)
        self.dT_expr = sp.diff(self.T_expr, self.L)
        self.d2T_expr = sp.diff(self.dT_expr, self.L)

    def evaluate_temp(self, load: float) -> float:
        """Evalúa numéricamente la temperatura para una carga L in [0, 1]."""
        load_clamped = max(0.0, min(1.0, float(load)))
        val = float(self.T_expr.subs(self.L, load_clamped).evalf())
        return round(val, 2)

    def analyze(self) -> ThermalCriticalPoints:
        """
        Ejecuta el análisis diferencial completo utilizando SymPy.
        Determina puntos estacionarios, puntos de inflexión y umbral de emergencia.
        """
        # 1. Puntos Estacionarios: dT/dL = 0
        raw_stationary = sp.solve(self.dT_expr, self.L)
        stationary_points = []
        for sol in raw_stationary:
            try:
                sol_float = float(sp.re(sol.evalf()))
                # Clasificación por criterio de la segunda derivada
                d2_val = float(self.d2T_expr.subs(self.L, sol).evalf())
                temp_val = float(self.T_expr.subs(self.L, sol).evalf())
                tipo = "Punto de Silla"
                if d2_val > 1e-6:
                    tipo = "Mínimo Local"
                elif d2_val < -1e-6:
                    tipo = "Máximo Local"

                stationary_points.append({
                    "load": round(sol_float, 4),
                    "temp": round(temp_val, 2),
                    "second_derivative": round(d2_val, 4),
                    "type": tipo,
                    "within_operating_range": (0.0 <= sol_float <= 1.0)
                })
            except Exception:
                continue

        # 2. Puntos de Inflexión: d^2T/dL^2 = 0
        raw_inflection = sp.solve(self.d2T_expr, self.L)
        inflection_points = []
        emergency_load = 0.60
        emergency_temp = self.evaluate_temp(emergency_load)

        for sol in raw_inflection:
            try:
                sol_float = float(sp.re(sol.evalf()))
                temp_val = float(self.T_expr.subs(self.L, sol).evalf())
                inflection_points.append({
                    "load": round(sol_float, 4),
                    "temp": round(temp_val, 2),
                    "within_operating_range": (0.0 <= sol_float <= 1.0)
                })
                if 0.0 <= sol_float <= 1.0:
                    emergency_load = sol_float
                    emergency_temp = temp_val
            except Exception:
                continue

        # Riesgo super-lineal si la segunda derivada es positiva hacia carga plena L -> 1
        d2_at_full = float(self.d2T_expr.subs(self.L, 1.0).evalf())
        is_superlinear_risk = (d2_at_full > 0)

        return ThermalCriticalPoints(
            expression_latex=str(self.T_expr),
            first_derivative_latex=str(self.dT_expr),
            second_derivative_latex=str(self.d2T_expr),
            stationary_points=stationary_points,
            inflection_points=inflection_points,
            emergency_threshold_load=round(emergency_load, 4),
            emergency_threshold_temp=round(emergency_temp, 2),
            is_superlinear_risk=is_superlinear_risk
        )


if __name__ == "__main__":
    print("=== Análisis Térmico Diferencial (Modelos de 1 Variable - SymPy) ===")
    analyzer = ThermalAnalyzer1D(t_ambient=22.0, c1=65.0, c2=54.0, c3=30.0)
    resultado = analyzer.analyze()
    print(f"Función T(L): {resultado.expression_latex}")
    print(f"Primera Derivada T'(L): {resultado.first_derivative_latex}")
    print(f"Segunda Derivada T''(L): {resultado.second_derivative_latex}")
    print("\nPuntos Estacionarios (T'(L) = 0):")
    if not resultado.stationary_points:
        print("  - No existen raíces reales dentro del dominio (Crecimiento estrictamente monotónico).")
    for pt in resultado.stationary_points:
        print(f"  - Carga: {pt['load']*100:.1f}%, Temp: {pt['temp']}°C, Tipo: {pt['type']}")

    print("\nPuntos de Inflexión (T''(L) = 0):")
    for pt in resultado.inflection_points:
        print(f"  - Carga Crítica: {pt['load']*100:.1f}%, Temp: {pt['temp']}°C (En rango: {pt['within_operating_range']})")

    print(f"\n=> UMBRAL DE ENFRIAMIENTO DE EMERGENCIA: Carga >= {resultado.emergency_threshold_load*100:.1f}% "
          f"({resultado.emergency_threshold_temp}°C)")
