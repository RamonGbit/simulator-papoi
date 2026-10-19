"""
Módulo de Optimización No Lineal Generalizada (POO).
DataStream Networks — Simulación Estocástica y Métodos Cuantitativos.
Universidad José Antonio Páez (UJAP).

Implementa los 4 métodos fundamentales de programación no lineal:
1. Modelos de una variable: Diferenciación simbólica, puntos estacionarios (máx/mín) y de inflexión.
2. Modelos de varias variables sin restricciones: Vector gradiente y clasificación con Matriz Hessiana.
3. Método del Gradiente: Optimización numérica iterativa por gradiente descendente / ascendente.
4. Método de Multiplicadores de Lagrange: Optimización no lineal sujeta a restricciones de igualdad.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Callable
import sympy as sp
import numpy as np


# =============================================================================
# DTOs / CONTENEDORES DE RESULTADOS
# =============================================================================

@dataclass
class UnivariateCriticalPoint:
    """Representa un punto crítico o estacionario de una función de 1 variable."""
    x: float
    f_x: float
    second_derivative_val: float
    classification: str  # "Mínimo Local", "Máximo Local", "Punto de Inflexión / Dudoso"
    is_inflection: bool = False


@dataclass
class UnivariateAnalysisResult:
    """Resultados del análisis diferencial de una función de 1 variable."""
    expression_str: str
    variable_name: str
    first_derivative_str: str
    second_derivative_str: str
    stationary_points: List[UnivariateCriticalPoint]
    inflection_points: List[UnivariateCriticalPoint]
    latex_formula: str


@dataclass
class MultivariateCriticalPoint:
    """Punto crítico multivariable clasificado mediante la Matriz Hessiana."""
    point_dict: Dict[str, float]
    f_val: float
    hessian_matrix_evaluated: List[List[float]]
    eigenvalues: List[float]
    principal_minors: List[float]
    classification: str  # "Mínimo Local", "Máximo Local", "Punto de Silla (Saddle)", "Indeterminado"


@dataclass
class MultivariateAnalysisResult:
    """Resultados del análisis de funciones de varias variables sin restricciones."""
    expression_str: str
    variables: List[str]
    gradient_str: Dict[str, str]
    hessian_matrix_symbolic: List[List[str]]
    critical_points: List[MultivariateCriticalPoint]


@dataclass
class GradientDescentIteration:
    """Registro de una iteración en el método del gradiente."""
    step: int
    coords: Dict[str, float]
    f_val: float
    grad_norm: float


@dataclass
class GradientDescentResult:
    """Resultados del método del gradiente numérico."""
    initial_point: Dict[str, float]
    optimal_point: Dict[str, float]
    optimal_val: float
    iterations: int
    converged: bool
    history: List[GradientDescentIteration]
    mode: str  # "minimize" | "maximize"


@dataclass
class ConstrainedLagrangeResult:
    """Resultados del método de Multiplicadores de Lagrange."""
    objective_str: str
    constraints_str: List[str]
    lagrangian_str: str
    optimal_solutions: List[Dict[str, float]]
    lambda_multipliers: List[Dict[str, float]]
    best_value: Optional[float]
    best_point: Optional[Dict[str, float]]


# =============================================================================
# 1. OPTIMIZADOR DE UNA VARIABLE (UNIVARIABLE)
# =============================================================================

class UnivariateOptimizer:
    """
    Analizador de modelos no lineales de 1 variable.
    Calcula analíticamente f'(x) y f''(x), determinando puntos estacionarios
    e inflexión mediante el criterio de la segunda derivada.
    """

    def __init__(self, expr_input: str | sp.Expr, var_name: str = "x"):
        self.var_name = var_name
        self.x = sp.Symbol(var_name, real=True)
        if isinstance(expr_input, str):
            # Parse seguro de expresión matemática
            self.expr = sp.sympify(expr_input, locals={var_name: self.x})
        else:
            self.expr = expr_input

        # Derivadas analíticas
        self.df = sp.diff(self.expr, self.x)
        self.d2f = sp.diff(self.df, self.x)

    def evaluate(self, val: float) -> float:
        """Evalúa numéricamente f(x)."""
        return float(self.expr.subs(self.x, float(val)).evalf())

    def analyze(self, domain_min: float = -1e6, domain_max: float = 1e6) -> UnivariateAnalysisResult:
        """
        Ejecuta el análisis completo:
        1. Puntos estacionarios: f'(x) = 0
        2. Clasificación con f''(x)
        3. Puntos de inflexión: f''(x) = 0
        """
        # 1. Resolver f'(x) = 0
        try:
            stationary_roots = sp.solve(self.df, self.x)
        except Exception:
            stationary_roots = []

        stationary_points: List[UnivariateCriticalPoint] = []
        for r in stationary_roots:
            try:
                r_num = complex(r.evalf())
                if abs(r_num.imag) < 1e-9:  # Raíz real
                    x_val = float(r_num.real)
                    if domain_min <= x_val <= domain_max:
                        f_val = float(self.expr.subs(self.x, x_val).evalf())
                        d2_val = float(self.d2f.subs(self.x, x_val).evalf())

                        if d2_val > 1e-7:
                            clf = "Mínimo Local"
                        elif d2_val < -1e-7:
                            clf = "Máximo Local"
                        else:
                            clf = "Punto de Inflexión / Dudoso"

                        stationary_points.append(UnivariateCriticalPoint(
                            x=round(x_val, 4),
                            f_x=round(f_val, 4),
                            second_derivative_val=round(d2_val, 4),
                            classification=clf,
                            is_inflection=(abs(d2_val) <= 1e-7)
                        ))
            except Exception:
                continue

        # 2. Resolver f''(x) = 0 (Puntos de inflexión)
        try:
            inflection_roots = sp.solve(self.d2f, self.x)
        except Exception:
            inflection_roots = []

        inflection_points: List[UnivariateCriticalPoint] = []
        for r in inflection_roots:
            try:
                r_num = complex(r.evalf())
                if abs(r_num.imag) < 1e-9:
                    x_val = float(r_num.real)
                    if domain_min <= x_val <= domain_max:
                        f_val = float(self.expr.subs(self.x, x_val).evalf())
                        inflection_points.append(UnivariateCriticalPoint(
                            x=round(x_val, 4),
                            f_x=round(f_val, 4),
                            second_derivative_val=0.0,
                            classification="Punto de Inflexión",
                            is_inflection=True
                        ))
            except Exception:
                continue

        return UnivariateAnalysisResult(
            expression_str=str(self.expr),
            variable_name=self.var_name,
            first_derivative_str=str(self.df),
            second_derivative_str=str(self.d2f),
            stationary_points=stationary_points,
            inflection_points=inflection_points,
            latex_formula=sp.latex(self.expr)
        )


# =============================================================================
# 2. OPTIMIZADOR MULTIVARIABLE SIN RESTRICCIONES (MATRIZ HESSIANA)
# =============================================================================

class MultivariateHessianOptimizer:
    """
    Optimizador de funciones de varias variables f(x1, x2, ..., xn) sin restricciones.
    Calcula el vector gradiente analítico y clasifica puntos críticos mediante
    la Matriz Hessiana y el Criterio de Sylvester (menores principales y eigenvalores).
    """

    def __init__(self, expr_input: str | sp.Expr, var_names: List[str]):
        if len(var_names) < 2:
            raise ValueError("El modelo multivariable requiere al menos 2 variables.")
        self.var_names = list(var_names)
        self.symbols = [sp.Symbol(name, real=True) for name in self.var_names]
        sym_dict = {name: s for name, s in zip(self.var_names, self.symbols)}

        if isinstance(expr_input, str):
            self.expr = sp.sympify(expr_input, locals=sym_dict)
        else:
            self.expr = expr_input

        # Vector gradiente analítico: [df/dx1, df/dx2, ...]
        self.gradient = [sp.diff(self.expr, s) for s in self.symbols]

        # Matriz Hessiana simbólica n x n: H_ij = d^2f / (dx_i dx_j)
        self.hessian_matrix = [
            [sp.diff(self.gradient[i], self.symbols[j]) for j in range(len(self.symbols))]
            for i in range(len(self.symbols))
        ]

    def analyze(self) -> MultivariateAnalysisResult:
        """
        Calcula los puntos críticos resolviendo el sistema gradiente = 0
        y los clasifica evaluando la Matriz Hessiana.
        """
        n = len(self.symbols)

        # 1. Resolver sistema gradiente = 0
        equations = [sp.Eq(g, 0) for g in self.gradient]
        try:
            crit_solutions = sp.solve(equations, self.symbols, dict=True)
        except Exception:
            crit_solutions = []

        critical_points: List[MultivariateCriticalPoint] = []

        for sol in crit_solutions:
            try:
                # Comprobar que todas las coordenadas sean reales
                is_real = True
                pt_dict: Dict[str, float] = {}
                sub_list = []
                for s, name in zip(self.symbols, self.var_names):
                    if s in sol:
                        val_c = complex(sol[s].evalf())
                        if abs(val_c.imag) > 1e-9:
                            is_real = False
                            break
                        pt_dict[name] = round(float(val_c.real), 4)
                        sub_list.append((s, sol[s]))
                    else:
                        is_real = False
                        break

                if not is_real:
                    continue

                # Evaluar f en el punto crítico
                f_val = float(self.expr.subs(sub_list).evalf())

                # Evaluar Matriz Hessiana numéricamente
                H_num = np.zeros((n, n), dtype=float)
                for i in range(n):
                    for j in range(n):
                        H_num[i, j] = float(self.hessian_matrix[i][j].subs(sub_list).evalf())

                # Menores principales (Determinantes principales de submatrices k x k)
                principal_minors: List[float] = []
                for k in range(1, n + 1):
                    sub_matrix = H_num[:k, :k]
                    minor_val = float(np.linalg.det(sub_matrix))
                    principal_minors.append(round(minor_val, 4))

                # Eigenvalores para clasificación robusta (tomando parte real)
                eigvals = [round(float(e.real), 4) for e in np.linalg.eigvals(H_num)]

                # Clasificación por Criterio de Sylvester / Eigenvalores
                pos_eigs = sum(1 for e in eigvals if e > 1e-6)
                neg_eigs = sum(1 for e in eigvals if e < -1e-6)

                if pos_eigs == n:
                    clf = "Mínimo Local"
                elif neg_eigs == n:
                    clf = "Máximo Local"
                elif pos_eigs > 0 and neg_eigs > 0:
                    clf = "Punto de Silla (Saddle)"
                else:
                    clf = "Indeterminado"

                critical_points.append(MultivariateCriticalPoint(
                    point_dict=pt_dict,
                    f_val=round(f_val, 4),
                    hessian_matrix_evaluated=[[round(float(val), 4) for val in row] for row in H_num],
                    eigenvalues=eigvals,
                    principal_minors=principal_minors,
                    classification=clf
                ))
            except Exception:
                continue

        grad_dict = {name: str(g) for name, g in zip(self.var_names, self.gradient)}
        hess_str = [[str(cell) for cell in row] for row in self.hessian_matrix]

        return MultivariateAnalysisResult(
            expression_str=str(self.expr),
            variables=self.var_names,
            gradient_str=grad_dict,
            hessian_matrix_symbolic=hess_str,
            critical_points=critical_points
        )


# =============================================================================
# 3. MÉTODO DEL GRADIENTE (OPTIMIZACIÓN NUMÉRICA ITERATIVA)
# =============================================================================

class GradientDescentOptimizer:
    """
    Optimizador por el Método del Gradiente para problemas multidimensionales.
    Permite minimizar o maximizar funciones no lineales suaves mediante:
        x^(k+1) = x^(k) - alpha * grad(f)   [Minimización]
        x^(k+1) = x^(k) + alpha * grad(f)   [Maximización]
    """

    def __init__(
        self,
        expr_input: str | sp.Expr,
        var_names: List[str],
        learning_rate: float = 0.05,
        tolerance: float = 1e-4,
        max_iterations: int = 150
    ):
        self.var_names = list(var_names)
        self.symbols = [sp.Symbol(name, real=True) for name in self.var_names]
        sym_dict = {name: s for name, s in zip(self.var_names, self.symbols)}

        if isinstance(expr_input, str):
            self.expr = sp.sympify(expr_input, locals=sym_dict)
        else:
            self.expr = expr_input

        self.lr = float(learning_rate)
        self.tol = float(tolerance)
        self.max_iter = int(max_iterations)

        # Gradientes simbólicos
        self.grad_exprs = [sp.diff(self.expr, s) for s in self.symbols]

    def optimize(
        self,
        initial_point: Dict[str, float],
        mode: str = "minimize"
    ) -> GradientDescentResult:
        """
        Ejecuta el ciclo iterativo de descenso o ascenso de gradiente.
        """
        current_pt = [float(initial_point.get(name, 0.0)) for name in self.var_names]
        history: List[GradientDescentIteration] = []
        sign = -1.0 if mode == "minimize" else 1.0

        converged = False
        for step in range(1, self.max_iter + 1):
            sub_list = [(s, current_pt[i]) for i, s in enumerate(self.symbols)]
            f_val = float(self.expr.subs(sub_list).evalf())

            # Calcular vector gradiente numérico
            grad_vals = [float(g.subs(sub_list).evalf()) for g in self.grad_exprs]
            grad_norm = float(np.linalg.norm(grad_vals))

            coords_dict = {name: round(current_pt[i], 4) for i, name in enumerate(self.var_names)}
            history.append(GradientDescentIteration(
                step=step,
                coords=coords_dict,
                f_val=round(f_val, 4),
                grad_norm=round(grad_norm, 6)
            ))

            # Criterio de parada
            if grad_norm < self.tol:
                converged = True
                break

            # Actualización de posición
            for i in range(len(current_pt)):
                current_pt[i] += sign * self.lr * grad_vals[i]

        opt_coords = {name: round(current_pt[i], 4) for i, name in enumerate(self.var_names)}
        sub_list_final = [(s, current_pt[i]) for i, s in enumerate(self.symbols)]
        opt_f = float(self.expr.subs(sub_list_final).evalf())

        return GradientDescentResult(
            initial_point=initial_point,
            optimal_point=opt_coords,
            optimal_val=round(opt_f, 4),
            iterations=len(history),
            converged=converged,
            history=history,
            mode=mode
        )


# =============================================================================
# 4. MÉTODO DE MULTIPLICADORES DE LAGRANGE (OPTIMIZACIÓN CON RESTRICCIONES)
# =============================================================================

class ConstrainedLagrangeOptimizer:
    """
    Optimizador con Multiplicadores de Lagrange para modelos no lineales con restricciones.
    Maximiza o minimiza f(x) sujeto a g_j(x) = 0.
    Construye analíticamente el Lagrangiano L(x, lambda) = f(x) - sum(lambda_j * g_j(x))
    y resuelve el sistema de ecuaciones no lineales.
    """

    def __init__(
        self,
        objective_expr: str | sp.Expr,
        constraint_exprs: List[str | sp.Expr],
        var_names: List[str]
    ):
        self.var_names = list(var_names)
        self.symbols = [sp.Symbol(name, real=True) for name in self.var_names]
        sym_dict = {name: s for name, s in zip(self.var_names, self.symbols)}

        if isinstance(objective_expr, str):
            self.f_expr = sp.sympify(objective_expr, locals=sym_dict)
        else:
            self.f_expr = objective_expr

        self.g_exprs = []
        for g in constraint_exprs:
            if isinstance(g, str):
                self.g_exprs.append(sp.sympify(g, locals=sym_dict))
            else:
                self.g_exprs.append(g)

        # Multiplicadores de Lagrange lambda_1, lambda_2, ...
        self.lam_symbols = [sp.Symbol(f"lambda_{i+1}", real=True) for i in range(len(self.g_exprs))]

        # Función Lagrangiana: L = f - sum(lambda_i * g_i)
        self.L_expr = self.f_expr - sum(lam * g for lam, g in zip(self.lam_symbols, self.g_exprs))

    def solve(self, mode: str = "maximize") -> ConstrainedLagrangeResult:
        """
        Resuelve el sistema de ecuaciones:
        1. dL/dx_i = 0  para cada variable x_i
        2. dL/dlambda_j = -g_j = 0  para cada multiplicador
        """
        all_variables = self.symbols + self.lam_symbols
        equations = [sp.Eq(sp.diff(self.L_expr, var), 0) for var in all_variables]

        try:
            solutions = sp.solve(equations, all_variables, dict=True)
        except Exception:
            solutions = []

        valid_points: List[Dict[str, float]] = []
        valid_lambdas: List[Dict[str, float]] = []

        best_val: Optional[float] = None
        best_point: Optional[Dict[str, float]] = None

        for sol in solutions:
            try:
                # Comprobar soluciones reales
                pt_dict: Dict[str, float] = {}
                lam_dict: Dict[str, float] = {}
                is_real = True

                for s, name in zip(self.symbols, self.var_names):
                    if s in sol:
                        val_c = complex(sol[s].evalf())
                        if abs(val_c.imag) > 1e-9:
                            is_real = False
                            break
                        pt_dict[name] = round(float(val_c.real), 4)

                for lam_s in self.lam_symbols:
                    if lam_s in sol:
                        lam_c = complex(sol[lam_s].evalf())
                        lam_dict[str(lam_s)] = round(float(lam_c.real), 4)

                if not is_real or len(pt_dict) < len(self.symbols):
                    continue

                sub_list = [(s, pt_dict[name]) for s, name in zip(self.symbols, self.var_names)]
                f_val = float(self.f_expr.subs(sub_list).evalf())

                valid_points.append(pt_dict)
                valid_lambdas.append(lam_dict)

                if best_val is None:
                    best_val = f_val
                    best_point = pt_dict
                elif mode == "maximize" and f_val > best_val:
                    best_val = f_val
                    best_point = pt_dict
                elif mode == "minimize" and f_val < best_val:
                    best_val = f_val
                    best_point = pt_dict
            except Exception:
                continue

        return ConstrainedLagrangeResult(
            objective_str=str(self.f_expr),
            constraints_str=[str(g) for g in self.g_exprs],
            lagrangian_str=str(self.L_expr),
            optimal_solutions=valid_points,
            lambda_multipliers=valid_lambdas,
            best_value=round(best_val, 4) if best_val is not None else None,
            best_point=best_point
        )


# =============================================================================
# SUITE INTEGRADA / FACHADA CON PRESETS DE DATASTREAM NETWORKS
# =============================================================================

class NonLinearLabSuite:
    """
    Fachada que reúne todos los métodos de programación no lineal.
    Proporciona presets contextualizados para DataStream Networks
    y métodos universales para evaluación libre.
    """

    @staticmethod
    def get_datacenter_presets() -> Dict[str, Dict[str, Any]]:
        """Devuelve los casos de optimización no lineal típicos del Centro de Datos."""
        return {
            "thermal_1d": {
                "name": "Análisis Térmico de Servidor (1 Variable)",
                "type": "univariate",
                "expr": "30*L**3 - 54*L**2 + 65*L + 22",
                "var": "L",
                "description": "Modela la temperatura (°C) respecto a la carga computacional L en [0, 1]. Inflexión en 60%."
            },
            "cpu_efficiency_multivariate": {
                "name": "Eficiencia Energética Frecuencia-Voltaje (Multivariable sin restricciones)",
                "type": "multivariate",
                "expr": "4*f**2 + 6*v**2 - 2*f*v - 24*f - 30*v + 100",
                "vars": ["f", "v"],
                "description": "Función de disipación conjunta. f: Frecuencia (GHz), v: Voltaje (V). Determina mínimo con Matriz Hessiana."
            },
            "cooling_gradient": {
                "name": "Descenso de Gradiente para Ventilador",
                "type": "gradient",
                "expr": "0.04*P**2 + 2.0*(85.0 - 0.45*P - 60.0)**2",
                "vars": ["P"],
                "initial": {"P": 10.0},
                "lr": 0.08,
                "description": "Ajuste dinámico de potencia P para estabilizar la temperatura del servidor en 60°C."
            },
            "cluster_lagrange": {
                "name": "Distribución Eléctrica entre Clústeres (Lagrange con restricción)",
                "type": "lagrange",
                "objective": "(8*P1 - 0.025*P1**2) + (12*P2 - 0.04*P2**2) + (15*P3 - 0.02*P3**2)",
                "constraints": ["P1 + P2 + P3 - 300"],
                "vars": ["P1", "P2", "P3"],
                "description": "Maximiza el rendimiento conjunto sujeto a un presupuesto energético total de 300 kW."
            }
        }
