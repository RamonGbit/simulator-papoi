"""
Módulo 2.3: Asignación de Presupuesto Eléctrico mediante Multiplicadores de Lagrange.
Modela el rendimiento global de la red del centro de datos como una función no lineal multivariable
con restricción de igualdad sobre el consumo energético total disponible.
Sigue estrictamente la metodología simbólica de SymPy y validación con SciPy minimize.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import sympy as sp
import numpy as np
from scipy.optimize import minimize


@dataclass
class ClusterEnergySpec:
    """Especificación de un clúster de servidores para asignación energética."""
    cluster_id: str
    name: str
    alpha: float  # Coeficiente lineal de ganancia de throughput
    beta: float   # Coeficiente cuadrático de rendimientos decrecientes


@dataclass
class LagrangeOptimizationResult:
    """Resultado de la optimización con multiplicadores de Lagrange."""
    total_budget: float
    cluster_allocations: Dict[str, float]
    lambda_multiplier: float
    total_performance: float
    is_budget_satisfied: bool
    lagrangian_latex: str
    partial_derivatives: Dict[str, str]
    symbolic_equations: List[str]


class LagrangePowerBudget:
    """
    Optimizador de Presupuesto Eléctrico Multivariable con Restricciones (Lagrange).
    
    Función Objetivo a Maximizar (Rendimiento Total):
        R(P_1, ..., P_k) = sum_{i=1}^k [ alpha_i * P_i - 0.5 * beta_i * (P_i)^2 ]
        
    Restricción de Presupuesto:
        g(P_1, ..., P_k) = sum_{i=1}^k P_i - P_total = 0
        
    Función Lagrangiana:
        L(P_1, ..., P_k, lambda) = R(P_1, ..., P_k) - lambda * g(P_1, ..., P_k)
        
    Condiciones Necesarias de Primer Orden:
        dL/dP_i = alpha_i - beta_i * P_i - lambda = 0   (para todo i = 1, ..., k)
        dL/d_lambda = -(sum_{i=1}^k P_i - P_total) = 0
    """

    def __init__(self, clusters: List[ClusterEnergySpec]):
        if len(clusters) < 2:
            raise ValueError("Se requieren al menos 2 clústeres para asignación multivariable.")
        self.clusters = clusters

    def optimize_symbolic(self, total_budget: float) -> LagrangeOptimizationResult:
        """
        Resuelve el sistema analítico usando SymPy, idéntico a las referencias del curso.
        """
        k = len(self.clusters)
        p_symbols = [sp.Symbol(f'P_{c.cluster_id}', real=True) for c in self.clusters]
        lam = sp.Symbol('lambda_mult', real=True)

        # 1. Definir Función Objetivo R(P)
        R_expr = sum(
            c.alpha * p_symbols[i] - 0.5 * c.beta * (p_symbols[i] ** 2)
            for i, c in enumerate(self.clusters)
        )

        # 2. Definir Restricción g(P) = sum(P_i) - P_total = 0
        g_expr = sum(p_symbols) - total_budget

        # 3. Construir Función Lagrangiana L = R - lambda * g
        L_expr = R_expr - lam * g_expr

        # 4. Derivadas Parciales de Primer Orden (dL/dP_i = 0 y dL/dlambda = 0)
        grad_eqs = [sp.diff(L_expr, p) for p in p_symbols]
        constraint_eq = -g_expr  # equivalente a dL/dlam
        system_equations = grad_eqs + [constraint_eq]

        # 5. Resolver el sistema algebraico
        unknowns = p_symbols + [lam]
        solutions = sp.solve(system_equations, unknowns, dict=True)

        if not solutions:
            # Fallback numérico con scipy en caso de singularidad
            return self._optimize_numeric_fallback(total_budget, L_expr, grad_eqs)

        sol = solutions[0]
        allocations: Dict[str, float] = {}
        for i, c in enumerate(self.clusters):
            val = float(sol[p_symbols[i]].evalf())
            allocations[c.cluster_id] = round(max(0.0, val), 2)

        lam_val = float(sol[lam].evalf())

        # Evaluar rendimiento total
        total_perf = float(R_expr.subs(sol).evalf())
        sum_allocated = sum(allocations.values())
        budget_satisfied = abs(sum_allocated - total_budget) < 0.5

        partial_dict = {
            f"dL/dP_{c.cluster_id}": str(sp.diff(L_expr, p_symbols[i]))
            for i, c in enumerate(self.clusters)
        }
        partial_dict["dL/dlambda"] = str(sp.diff(L_expr, lam))

        return LagrangeOptimizationResult(
            total_budget=round(total_budget, 2),
            cluster_allocations=allocations,
            lambda_multiplier=round(lam_val, 4),
            total_performance=round(total_perf, 2),
            is_budget_satisfied=budget_satisfied,
            lagrangian_latex=str(L_expr),
            partial_derivatives=partial_dict,
            symbolic_equations=[str(eq) + " = 0" for eq in system_equations]
        )

    def _optimize_numeric_fallback(
        self,
        total_budget: float,
        L_expr: sp.Expr,
        grad_eqs: List[sp.Expr]
    ) -> LagrangeOptimizationResult:
        """Respaldo mediante SciPy minimize."""
        k = len(self.clusters)

        def objective(p):
            return -sum(c.alpha * p[i] - 0.5 * c.beta * (p[i] ** 2) for i, c in enumerate(self.clusters))

        def constraint(p):
            return sum(p) - total_budget

        cons = {'type': 'eq', 'fun': constraint}
        p0 = [total_budget / k] * k
        bounds = [(0.0, total_budget) for _ in range(k)]
        res = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=cons)

        allocations = {c.cluster_id: round(float(res.x[i]), 2) for i, c in enumerate(self.clusters)}
        total_perf = -float(res.fun)

        return LagrangeOptimizationResult(
            total_budget=round(total_budget, 2),
            cluster_allocations=allocations,
            lambda_multiplier=0.0,
            total_performance=round(total_perf, 2),
            is_budget_satisfied=True,
            lagrangian_latex=str(L_expr),
            partial_derivatives={},
            symbolic_equations=[]
        )


if __name__ == "__main__":
    print("=== Prueba Independiente: Asignación con Multiplicadores de Lagrange (SymPy) ===")
    clusters_demo = [
        ClusterEnergySpec("web", "Clúster Web Frontend", alpha=8.0, beta=0.05),
        ClusterEnergySpec("db", "Clúster Base de Datos", alpha=12.0, beta=0.08),
        ClusterEnergySpec("ai", "Clúster Cómputo IA", alpha=15.0, beta=0.04),
    ]

    opt_lagrange = LagrangePowerBudget(clusters_demo)
    presupuesto = 300.0  # Presupuesto eléctrico total (kW)
    res_lagrange = opt_lagrange.optimize_symbolic(total_budget=presupuesto)

    print(f"Presupuesto Total Restricción: {res_lagrange.total_budget} kW")
    print(f"Lagrangiano L: {res_lagrange.lagrangian_latex}")
    print("\nDerivadas Parciales de Primer Orden (dL/dvar = 0):")
    for k_var, d_expr in res_lagrange.partial_derivatives.items():
        print(f"  {k_var}: {d_expr}")

    print(f"\nMultiplicador de Lagrange Optimo (lambda*): {res_lagrange.lambda_multiplier}")
    print("Distribución Óptima de Energía por Clúster:")
    for cid, pot in res_lagrange.cluster_allocations.items():
        print(f"  [+] Clúster {cid.upper()}: {pot} kW")

    print(f"Suma Total Asignada: {sum(res_lagrange.cluster_allocations.values()):.2f} kW")
    print(f"Rendimiento Global Máximo Obtenido: {res_lagrange.total_performance} unidades")
