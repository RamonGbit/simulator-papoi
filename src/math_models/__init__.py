"""
Módulo de modelos matemáticos puros para optimización de centros de datos.
Contiene implementaciones independientes de Programación Dinámica y Optimización No Lineal.
"""
from .dp_routing import DPRoutingOptimizer, RouteResult
from .dp_knapsack import DPKnapsackScheduler, TaskItem, KnapsackResult
from .thermal_analysis import ThermalAnalyzer1D, ThermalCriticalPoints
from .cooling_gradient import GradientDescentCooling, CoolingOptimizationResult
from .power_lagrange import LagrangePowerBudget, LagrangeOptimizationResult

__all__ = [
    "DPRoutingOptimizer",
    "RouteResult",
    "DPKnapsackScheduler",
    "TaskItem",
    "KnapsackResult",
    "ThermalAnalyzer1D",
    "ThermalCriticalPoints",
    "GradientDescentCooling",
    "CoolingOptimizationResult",
    "LagrangePowerBudget",
    "LagrangeOptimizationResult"
]
