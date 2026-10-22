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


def test_nonlinear_lab_tab5_custom_solver():
    """Valida el solucionador interactivo de problemas libres (Tab 5) y guardado a PDF."""
    import pygame
    from src.ui.theme import get_fonts
    from src.ui.renderers.nonlinear_lab_view import NonLinearLabRenderer, PRESETS_CUSTOM

    fonts = get_fonts()
    renderer = NonLinearLabRenderer((0, 0, 1200, 750), fonts)
    renderer.active_tab = 5

    # 1. Probar carga de preset 1D Maximización
    renderer._load_preset(PRESETS_CUSTOM[0])
    assert renderer.custom_solution is not None
    assert renderer.custom_solution["best_point"] is not None
    assert abs(renderer.custom_solution["best_point"].x - 3.0) < 1e-3
    assert abs(renderer.custom_solution["best_point"].f_x - 23.0) < 1e-3

    # 2. Guardar a PDF
    renderer._save_current_problem_to_pdf()
    assert len(renderer.custom_saved_problems) == 1
    assert renderer.custom_saved_problems[0]["goal"] == "Maximizar"

    # 3. Probar carga de preset 2D Minimización con Hessiano
    renderer._load_preset(PRESETS_CUSTOM[2])
    assert renderer.custom_solution is not None
    assert renderer.custom_solution["verdict"] == "Mínimo Local"
    assert renderer.custom_solution["best_point"].point_dict["x"] == 2.0
    assert renderer.custom_solution["best_point"].point_dict["y"] == 1.0

    # 4. Guardar segundo problema a PDF
    renderer._save_current_problem_to_pdf()
    assert len(renderer.custom_saved_problems) == 2
    assert renderer.custom_saved_problems[1]["goal"] == "Minimizar"


def test_custom_problem_math_sanitizer():
    """Valida la conversión de expresiones de usuario a sintaxis canónica de SymPy."""
    import pygame
    from src.ui.theme import get_fonts
    from src.ui.renderers.nonlinear_lab_view import NonLinearLabRenderer, to_pretty_math

    fonts = get_fonts()
    renderer = NonLinearLabRenderer((0, 0, 1200, 750), fonts)

    # 2x^2 + 3y -> 2*x**2 + 3*y
    sanitized = renderer._sanitize_math_input("2x^2 + 3y")
    assert sanitized == "2*x**2 + 3*y"

    # Notación matemática visual
    pretty = to_pretty_math("x**2 + 2*y**2")
    assert "²" in pretty


def test_custom_problem_constraint_parsing_and_kkt():
    """Valida el análisis de restricciones con igualdades y desigualdades (KKT/SLSQP)."""
    import pygame
    from src.ui.theme import get_fonts
    from src.ui.renderers.nonlinear_lab_view import NonLinearLabRenderer, PRESETS_CUSTOM

    fonts = get_fonts()
    renderer = NonLinearLabRenderer((0, 0, 1200, 750), fonts)
    renderer.active_tab = 5

    # 1. Parsing de restricciones múltiples y mixtas
    parsed = renderer._parse_constraints("x + 2*y <= 20; x >= 0, y >= 0")
    assert len(parsed) == 3
    assert parsed[0]["type"] == "<="
    assert parsed[1]["type"] == ">="
    assert parsed[2]["type"] == ">="

    # 2. Preset KKT 2D: Mín (x-2)²+(y-1)² s.a. x+y>=4
    renderer._load_preset(PRESETS_CUSTOM[4])
    assert renderer.custom_solution is not None
    assert renderer.custom_solution["type"] == "kkt"
    best_p = renderer.custom_solution["best_point"]
    assert abs(best_p.point_dict["x"] - 2.5) < 0.1
    assert abs(best_p.point_dict["y"] - 1.5) < 0.1
    assert abs(best_p.f_val - 0.5) < 0.1
    assert abs(best_p.x - 2.5) < 0.1
    assert abs(best_p["y"] - 1.5) < 0.1

    # Preset KKT 1D: Máx -2x²+12x+5 s.a. x<=2
    renderer._load_preset(PRESETS_CUSTOM[1])
    assert renderer.custom_solution is not None
    best_1d = renderer.custom_solution["best_point"]
    assert abs(best_1d.x - 2.0) < 0.1
    assert abs(best_1d.f_val - 21.0) < 0.1

    surf = pygame.Surface((1200, 750))
    renderer.render(surf)

    # 3. Vaciar restricción con el botón rápido
    clicked = renderer.handle_click(renderer.btn_clear_constraint.center)
    assert clicked is True
    assert renderer.input_values["constraint"] == ""
    # Al vaciar, se resuelve como libre
    assert renderer.custom_solution is not None

    # 4. Navegación con tecla TAB cicla por la restricción siempre
    renderer.focused_field = "vars"
    event_tab = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_TAB})
    renderer.handle_key_event(event_tab)
    assert renderer.focused_field == "constraint"


def test_custom_problem_tema8_multi_constraints():
    """Valida los ejercicios de Tema 8 (Programación No Lineal) con 3 y 4 restricciones simultáneas."""
    import pygame
    from src.ui.theme import get_fonts
    from src.ui.renderers.nonlinear_lab_view import NonLinearLabRenderer, PRESETS_CUSTOM

    fonts = get_fonts()
    renderer = NonLinearLabRenderer((0, 0, 1200, 750), fonts)
    renderer.active_tab = 5

    # 1. Tema 8 - Ejemplo 1: Fertilizante con 3 restricciones
    # Max Q = 4x1 + 2x2 - 0.5x1^2 - 0.25x2^2 s.a. 480x1 + 300x2 <= 24000; x1, x2 >= 0
    renderer._load_preset(PRESETS_CUSTOM[5])
    assert renderer.custom_solution is not None
    assert len(renderer.custom_solution["slack_analysis"]) == 3
    bp_fert = renderer.custom_solution["best_point"]
    assert abs(bp_fert.point_dict["x1"] - 4.0) < 0.1
    assert abs(bp_fert.point_dict["x2"] - 4.0) < 0.1
    assert abs(bp_fert.f_val - 12.0) < 0.1

    # 2. Tema 8 - Ejemplo 4: Petróleo con 4 restricciones no lineales y lineales
    # Max Z = x1*(30 - x1) + x2*(35 - x2) - x1^2 - 2*x2^2 s.a. x1^2+x2^2<=250; x1+x2<=20; x1, x2 >= 0
    renderer._load_preset(PRESETS_CUSTOM[6])
    assert renderer.custom_solution is not None
    assert len(renderer.custom_solution["slack_analysis"]) == 4
    bp_pet = renderer.custom_solution["best_point"]
    assert abs(bp_pet.point_dict["x1"] - 7.50) < 0.1
    assert abs(bp_pet.point_dict["x2"] - 5.83) < 0.1
    assert abs(bp_pet.f_val - 214.58) < 0.2

    # 3. Guardado en lista de PDF y verificación de estructura
    renderer._save_current_problem_to_pdf()
    assert len(renderer.custom_saved_problems) == 1
    assert "x1" in renderer.custom_saved_problems[0]["clean_expr"]
    assert "<=" in renderer.custom_saved_problems[0]["clean_constr"]


