"""
Pruebas Unitarias para el Módulo de Optimización No Lineal Generalizada (POO).
Valida los 4 métodos:
1. Univariable (Diferenciación, extremos e inflexión).
2. Multivariable sin restricciones (Gradiente y Matriz Hessiana con Sylvester).
3. Método del Gradiente (Descenso y convergencia).
4. Multiplicadores de Lagrange (Restricciones analíticas).
"""

import pytest
import numpy as np
from src.math_models.nonlinear_solver import (
    UnivariateOptimizer,
    MultivariateHessianOptimizer,
    GradientDescentOptimizer,
    ConstrainedLagrangeOptimizer,
    NonLinearLabSuite
)


def test_univariate_cubic_inflection_and_extrema():
    """Prueba una función cúbica conocida: f(x) = x^3 - 3*x^2 + 2."""
    optimizer = UnivariateOptimizer("x**3 - 3*x**2 + 2", var_name="x")
    res = optimizer.analyze()

    # f'(x) = 3*x^2 - 6*x = 0 -> x = 0 (Máx local), x = 2 (Mín local)
    assert len(res.stationary_points) == 2
    pts_by_clf = {p.classification: p.x for p in res.stationary_points}
    assert "Máximo Local" in pts_by_clf
    assert "Mínimo Local" in pts_by_clf
    assert pts_by_clf["Máximo Local"] == 0.0
    assert pts_by_clf["Mínimo Local"] == 2.0

    # f''(x) = 6*x - 6 = 0 -> x = 1.0 (Punto de inflexión)
    assert len(res.inflection_points) == 1
    assert res.inflection_points[0].x == 1.0
    assert res.inflection_points[0].is_inflection is True


def test_univariate_datacenter_thermal_model():
    """Prueba la función fenomenológica de temperatura de DataStream Networks."""
    preset = NonLinearLabSuite.get_datacenter_presets()["thermal_1d"]
    optimizer = UnivariateOptimizer(preset["expr"], var_name=preset["var"])
    res = optimizer.analyze()

    # Inflexión exacta en L = 0.60 (60% de carga)
    assert len(res.inflection_points) >= 1
    inflection_load = res.inflection_points[0].x
    assert abs(inflection_load - 0.60) < 1e-2


def test_multivariate_hessian_saddle_point():
    """Prueba clasificación de Punto de Silla con f(x, y) = x^2 - y^2."""
    optimizer = MultivariateHessianOptimizer("x**2 - y**2", ["x", "y"])
    res = optimizer.analyze()

    assert len(res.critical_points) == 1
    cp = res.critical_points[0]
    assert cp.point_dict == {"x": 0.0, "y": 0.0}
    assert cp.classification == "Punto de Silla (Saddle)"
    # det(H) = (2)*(-2) - 0 = -4 < 0
    assert cp.principal_minors[-1] == -4.0


def test_multivariate_hessian_datacenter_cpu_efficiency():
    """
    Prueba el modelo de eficiencia de disipación de CPU:
    f(f, v) = 4*f^2 + 6*v^2 - 2*f*v - 24*f - 30*v + 100
    """
    preset = NonLinearLabSuite.get_datacenter_presets()["cpu_efficiency_multivariate"]
    optimizer = MultivariateHessianOptimizer(preset["expr"], preset["vars"])
    res = optimizer.analyze()

    assert len(res.critical_points) == 1
    cp = res.critical_points[0]
    # Comprobar que es un Mínimo Local estricto (Hessiano definido positivo)
    assert cp.classification == "Mínimo Local"
    assert all(m > 0 for m in cp.principal_minors)


def test_gradient_descent_convergence():
    """Prueba convergencia de descenso de gradiente en una parábola: f(x, y) = (x-3)^2 + (y+2)^2."""
    optimizer = GradientDescentOptimizer(
        "(x - 3)**2 + (y + 2)**2",
        ["x", "y"],
        learning_rate=0.2,
        tolerance=1e-4,
        max_iterations=100
    )
    res = optimizer.optimize(initial_point={"x": 0.0, "y": 0.0}, mode="minimize")

    assert res.converged is True
    assert abs(res.optimal_point["x"] - 3.0) < 0.05
    assert abs(res.optimal_point["y"] - (-2.0)) < 0.05
    assert res.optimal_val < 0.01


def test_constrained_lagrange_budget():
    """
    Prueba el método de Lagrange con función cuadrática y restricción lineal:
    Max f(x, y) = 10*x - x^2 + 20*y - 2*y^2
    Sujeto a: x + y - 10 = 0
    """
    optimizer = ConstrainedLagrangeOptimizer(
        objective_expr="10*x - x**2 + 20*y - 2*y**2",
        constraint_exprs=["x + y - 10"],
        var_names=["x", "y"]
    )
    res = optimizer.solve(mode="maximize")

    assert len(res.optimal_solutions) >= 1
    best_pt = res.best_point
    assert best_pt is not None
    # Verificar que cumple la restricción x + y = 10
    total = best_pt["x"] + best_pt["y"]
    assert abs(total - 10.0) < 0.01
