"""
Módulo 2.2: Ajuste Dinámico de Refrigeración (Método de Gradiente Descendente).
Optimiza continuamente la potencia de los ventiladores del centro de datos en tiempo real,
minimizando el consumo energético sin comprometer los límites térmicos operativos.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple


@dataclass
class CoolingOptimizationResult:
    """Resultado del proceso de optimización mediante gradiente descendente."""
    initial_power: float
    optimal_power: float
    initial_cost: float
    optimal_cost: float
    final_temp: float
    iterations: int
    converged: bool
    history_power: List[float] = field(default_factory=list)
    history_cost: List[float] = field(default_factory=list)
    history_gradient: List[float] = field(default_factory=list)


class GradientDescentCooling:
    """
    Optimizador de Refrigeración por Descenso de Gradiente.
    
    Función de Costo:
        J(P) = k_fan * P^2 + k_pen * max(0, T_base - eta * P - T_target)^2
    donde:
        P = Potencia del ventilador [min_power, max_power]
        T_base = Temperatura generada por la carga del servidor
        T_net(P) = T_base - eta * P
        J'(P) = 2 * k_fan * P - 2 * eta * k_pen * (T_base - eta * P - T_target)   (si T_net > T_target)
    """

    def __init__(
        self,
        k_fan: float = 0.04,
        k_penalty: float = 2.0,
        eta_cooling: float = 0.45,
        target_temp: float = 60.0,
        min_power: float = 5.0,
        max_power: float = 120.0,
        learning_rate: float = 0.12,
        tolerance: float = 1e-3,
        max_iterations: int = 150
    ):
        self.k_fan = float(k_fan)
        self.k_penalty = float(k_penalty)
        self.eta = float(eta_cooling)
        self.target_temp = float(target_temp)
        self.min_power = float(min_power)
        self.max_power = float(max_power)
        self.lr = float(learning_rate)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)

        # Estado en vivo para animación paso a paso
        self.current_power: float = self.min_power

    def compute_temperature(self, t_base: float, power: float) -> float:
        """Calcula la temperatura neta resultante para una potencia dada."""
        t_net = t_base - (self.eta * power)
        return max(20.0, t_net)

    def cost_function(self, power: float, t_base: float) -> float:
        """Calcula el costo total J(P)."""
        power_cost = self.k_fan * (power ** 2)
        t_net = self.compute_temperature(t_base, power)
        thermal_excess = max(0.0, t_net - self.target_temp)
        penalty_cost = self.k_penalty * (thermal_excess ** 2)
        return power_cost + penalty_cost

    def gradient(self, power: float, t_base: float) -> float:
        """Calcula la derivada dJ/dP."""
        grad_power = 2.0 * self.k_fan * power
        t_net = self.compute_temperature(t_base, power)
        thermal_excess = t_net - self.target_temp

        if thermal_excess > 0:
            grad_thermal = -2.0 * self.k_penalty * thermal_excess * self.eta
        else:
            # Si la temperatura está bajo el umbral seguro, sólo actúa la presión para ahorrar energía
            grad_thermal = 0.0

        return grad_power + grad_thermal

    def step(self, t_base: float) -> Tuple[float, float, float]:
        """
        Ejecuta un único paso de gradiente descendente (ideal para el bucle de Pygame a 60 FPS).
        Returns: (nueva_potencia, costo, gradiente)
        """
        grad = self.gradient(self.current_power, t_base)
        new_power = self.current_power - (self.lr * grad)
        # Acotamos a los límites físicos del ventilador
        self.current_power = max(self.min_power, min(self.max_power, new_power))
        cost = self.cost_function(self.current_power, t_base)
        return self.current_power, cost, grad

    def optimize_full(self, t_base: float, initial_power: Optional[float] = None) -> CoolingOptimizationResult:
        """
        Ejecuta el descenso de gradiente completo hasta convergencia o iteraciones máximas.
        """
        power = self.min_power if initial_power is None else float(initial_power)
        power = max(self.min_power, min(self.max_power, power))

        hist_p = [power]
        hist_c = [self.cost_function(power, t_base)]
        hist_g = []

        converged = False
        final_iter = 0

        for i in range(self.max_iterations):
            final_iter = i + 1
            grad = self.gradient(power, t_base)
            hist_g.append(grad)

            if abs(grad) < self.tolerance:
                converged = True
                break

            power = power - (self.lr * grad)
            power = max(self.min_power, min(self.max_power, power))
            cost = self.cost_function(power, t_base)

            hist_p.append(power)
            hist_c.append(cost)

        if not hist_g:
            hist_g.append(self.gradient(power, t_base))

        self.current_power = power

        return CoolingOptimizationResult(
            initial_power=round(hist_p[0], 2),
            optimal_power=round(power, 2),
            initial_cost=round(hist_c[0], 4),
            optimal_cost=round(hist_c[-1], 4),
            final_temp=round(self.compute_temperature(t_base, power), 2),
            iterations=final_iter,
            converged=converged,
            history_power=hist_p,
            history_cost=hist_c,
            history_gradient=hist_g
        )


if __name__ == "__main__":
    print("=== Prueba Independiente: Ajuste Dinámico por Método de Gradiente ===")
    cooler = GradientDescentCooling(
        k_fan=0.04,
        k_penalty=2.5,
        eta_cooling=0.40,
        target_temp=60.0,
        min_power=10.0,
        max_power=120.0,
        learning_rate=0.08
    )

    t_servidor_caliente = 88.0  # Servidor con sobrecarga térmica a 88°C
    res = cooler.optimize_full(t_base=t_servidor_caliente, initial_power=10.0)

    print(f"Temperatura Inicial (Sin refrigeración activa): {t_servidor_caliente}°C")
    print(f"Potencia Inicial del Ventilador: {res.initial_power} W | Costo: {res.initial_cost}")
    print(f"Potencia Óptima Convergida: {res.optimal_power} W | Costo Mínimo: {res.optimal_cost}")
    print(f"Temperatura Final Resultante: {res.final_temp}°C (Target: 60.0°C)")
    print(f"Iteraciones necesarias: {res.iterations} (Convergencia: {res.converged})")
    print(f"Primeros pasos de gradiente: {[round(g, 4) for g in res.history_gradient[:5]]}")
