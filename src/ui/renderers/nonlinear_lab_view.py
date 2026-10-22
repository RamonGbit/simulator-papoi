"""
Renderizador del Laboratorio de Programación No Lineal para Pygame.
Permite explorar interactivamente los 4 modelos no lineales con navegación PASO A PASO,
animaciones dinámicas de tangentes, curvas de nivel, descenso de gradiente y sensibilidad de Lagrange:
1. Univariable: Ecuación térmica, recta tangente animada T'(L), curvatura d^2T/dL^2 y zonas seguras/críticas.
2. Multivariable: Vector Gradiente, Matriz Hessiana y Criterio de Sylvester con veredicto analítico.
3. Método del Gradiente: Descenso interactivo paso a paso con bolita rodante sobre la parábola y vector tangente.
4. Multiplicadores de Lagrange: Deducción paso a paso y simulador interactivo de presupuesto con SymPy en vivo.
"""

from __future__ import annotations
import math
import time
import re
from typing import Dict, Any, List, Optional, Tuple
import pygame
import numpy as np
import scipy.optimize as opt
import sympy as sp

from src.ui.theme import (
    PANEL_BG, CARD_BG, BORDER_COLOR, BORDER_ACTIVE,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    ROUTER_ACTIVE, SERVER_NORMAL, SERVER_WARNING, SERVER_EMERGENCY,
    CHART_LINE_GRADIENT, CHART_LINE_COST, CHART_GRID, CHART_AXIS
)
from src.math_models.nonlinear_solver import (
    UnivariateOptimizer,
    MultivariateHessianOptimizer,
    GradientDescentOptimizer,
    ConstrainedLagrangeOptimizer,
    NonLinearLabSuite
)


def to_pretty_math(expr_str: str) -> str:
    """Convierte representaciones algebraicas de SymPy a notación matemática legible con superíndices."""
    s = str(expr_str)
    s = s.replace("**3", "³").replace("**2", "²").replace("^3", "³").replace("^2", "²")
    s = re.sub(r'(\d+(?:\.\d+)?)\*([A-Za-z])', r'\1\2', s)
    s = re.sub(r'([A-Za-z])\*([A-Za-z])', r'\1\2', s)
    s = s.replace("*", " · ")
    return s


class KKTPointResult:
    """Representación unificada de un punto óptimo numérico (SLSQP / KKT)."""
    def __init__(self, point_dict: dict[str, float], f_val: float):
        self.point_dict = point_dict
        self.f_val = f_val
        self.f_x = f_val
        self.lambdas: dict[str, float] = {}
        for k, v in point_dict.items():
            setattr(self, k, v)
        if "x" in point_dict:
            self.x = point_dict["x"]
        if "y" in point_dict:
            self.y = point_dict["y"]

    def __getitem__(self, item):
        return self.point_dict[item]

    def __contains__(self, item):
        return item in self.point_dict

    def get(self, item, default=None):
        return self.point_dict.get(item, default)

    def items(self):
        return self.point_dict.items()

    def keys(self):
        return self.point_dict.keys()

    def values(self):
        return self.point_dict.values()


PRESETS_CUSTOM: List[Dict[str, Any]] = [
    {
        "title": "1D: Máx -2x²+12x+5 (Libre)",
        "type": "univariate",
        "goal": "maximize",
        "expr": "-2*x**2 + 12*x + 5",
        "vars": "x",
        "constraint": "",
        "init_val": "0.0"
    },
    {
        "title": "[KKT 1D] Máx: -2x²+12x+5, x<=2",
        "type": "univariate",
        "goal": "maximize",
        "expr": "-2*x**2 + 12*x + 5",
        "vars": "x",
        "constraint": "x <= 2",
        "init_val": "0.0"
    },
    {
        "title": "2D: Mín (x-2)²+(y-1)²+4 (Libre)",
        "type": "multivariate",
        "goal": "minimize",
        "expr": "(x - 2)**2 + (y - 1)**2 + 4",
        "vars": "x, y",
        "constraint": "",
        "init_val": "0.0"
    },
    {
        "title": "[Lagr 2D] Máx: xy, x+2y=20",
        "type": "lagrange",
        "goal": "maximize",
        "expr": "x*y",
        "vars": "x, y",
        "constraint": "x + 2*y = 20",
        "init_val": "0.0"
    },
    {
        "title": "[KKT 2D] Mín: (x-2)²+(y-1)², x+y>=4",
        "type": "multivariate",
        "goal": "minimize",
        "expr": "(x - 2)**2 + (y - 1)**2",
        "vars": "x, y",
        "constraint": "x + y >= 4",
        "init_val": "0.0"
    },
    {
        "title": "[Tema 8] Fertilizante (3 rest.)",
        "type": "multivariate",
        "goal": "maximize",
        "expr": "4*x1 + 2*x2 - 0.5*x1**2 - 0.25*x2**2",
        "vars": "x1, x2",
        "constraint": "480*x1 + 300*x2 <= 24000; x1, x2 >= 0",
        "init_val": "0.0"
    },
    {
        "title": "[Tema 8] Petróleo (4 rest.)",
        "type": "multivariate",
        "goal": "maximize",
        "expr": "x1*(30 - x1) + x2*(35 - x2) - x1**2 - 2*x2**2",
        "vars": "x1, x2",
        "constraint": "x1**2 + x2**2 <= 250; x1 + x2 <= 20; x1, x2 >= 0",
        "init_val": "0.0"
    },
    {
        "title": "[PDF] Círculo: Máx xy, x²+y²=1",
        "type": "lagrange",
        "goal": "maximize",
        "expr": "x*y",
        "vars": "x, y",
        "constraint": "x**2 + y**2 = 1",
        "init_val": "0.0"
    }
]


class NonLinearLabRenderer:
    """
    Renderizador interactivo del modal de Laboratorio No Lineal con navegación paso a paso y animaciones.
    Incluye solucionador libre para ingresar problemas nuevos de maximización y minimización.
    """

    def __init__(self, rect: tuple[int, int, int, int], fonts: dict[str, pygame.font.Font]):
        self.rect = pygame.Rect(rect)
        self.fonts = fonts
        self.active_tab: int = 1  # 1: 1D, 2: Hessiano, 3: Gradiente, 4: Lagrange, 5: Custom

        # Estados de paso activo por cada método (Paso a paso)
        self.step_1d: int = 1         # Pasos 1 a 4
        self.step_hessian: int = 1    # Pasos 1 a 5
        self.step_grad: int = 0       # Pasos 0 a N iteraciones
        self.step_lagr: int = 1       # Pasos 1 a 5

        # Reproducción automática (Auto-play)
        self.is_auto_playing: bool = False
        self.last_auto_tick: float = time.time()

        # Presupuesto interactivo para el simulador de Lagrange
        self.lab_budget: float = 300.0

        # Cargar presets matemáticos pre-calculados
        self.presets = NonLinearLabSuite.get_datacenter_presets()
        self._init_models()

        # Botones de pestañas superiores (5 pestañas)
        self.tab_rects: List[pygame.Rect] = []
        tab_w = (self.rect.width - 40) // 5
        for i in range(5):
            rx = self.rect.x + 20 + i * tab_w
            ry = self.rect.y + 60
            self.tab_rects.append(pygame.Rect(rx, ry, tab_w - 6, 34))

        # Barra de control paso a paso
        ctrl_y = self.rect.y + 102
        self.btn_prev_rect = pygame.Rect(self.rect.x + 20, ctrl_y, 115, 30)
        self.btn_next_rect = pygame.Rect(self.rect.x + 145, ctrl_y, 115, 30)
        self.btn_auto_rect = pygame.Rect(self.rect.x + 270, ctrl_y, 130, 30)
        self.btn_reset_rect = pygame.Rect(self.rect.x + 410, ctrl_y, 90, 30)

        # Botones de Cerrar [X] y Volver al Datacenter
        self.should_close: bool = False
        self.btn_close_rect = pygame.Rect(self.rect.right - 46, self.rect.y + 14, 30, 30)
        self.btn_back_rect = pygame.Rect(self.rect.right - 195, self.rect.y + 14, 140, 30)

        # Botones de presupuesto para el simulador de Lagrange (Tab 4)
        self.btn_lagr_minus = pygame.Rect(0, 0, 0, 0)
        self.btn_lagr_plus = pygame.Rect(0, 0, 0, 0)

        # Estado de Tab 5: Solucionador de Problemas Personalizados (Máx / Mín)
        self.custom_model_type: str = "lagrange"    # "lagrange", "univariate", "multivariate", "gradient"
        self.custom_goal: str = "maximize"          # "maximize" | "minimize"
        self.focused_field: Optional[str] = "expr"   # "expr", "vars", "constraint", "init_val"
        self.input_values: Dict[str, str] = {
            "expr": "x*y",
            "vars": "x, y",
            "constraint": "x + 2*y = 20",
            "init_val": "0.0"
        }
        self.custom_solution: Optional[Dict[str, Any]] = None
        self.custom_error_msg: Optional[str] = None
        self.custom_saved_problems: List[Dict[str, Any]] = []
        self.custom_toast_msg: str = ""
        self.custom_toast_timer: float = 0.0

        # Botones y campos interactivos para Tab 5
        self.type_button_rects: Dict[str, pygame.Rect] = {}
        self.goal_button_rects: Dict[str, pygame.Rect] = {}
        self.input_rects: Dict[str, pygame.Rect] = {}
        self.preset_button_rects: List[Tuple[pygame.Rect, Dict[str, Any]]] = []
        self.btn_solve_custom = pygame.Rect(0, 0, 0, 0)
        self.btn_save_custom_pdf = pygame.Rect(0, 0, 0, 0)
        self.btn_clear_custom = pygame.Rect(0, 0, 0, 0)
        self.btn_clear_constraint = pygame.Rect(0, 0, 0, 0)

        # Resolver el problema inicial por defecto
        self.solve_custom_problem()

    def _init_models(self) -> None:
        """Inicializa los análisis de los 4 modelos de demostración."""
        # 1. Univariable
        t_preset = self.presets["thermal_1d"]
        self.uni_solver = UnivariateOptimizer(t_preset["expr"], var_name=t_preset["var"])
        self.uni_res = self.uni_solver.analyze(domain_min=0.0, domain_max=1.0)

        # 2. Multivariable Hessiano
        m_preset = self.presets["cpu_efficiency_multivariate"]
        self.multi_solver = MultivariateHessianOptimizer(m_preset["expr"], m_preset["vars"])
        self.multi_res = self.multi_solver.analyze()

        # 3. Gradiente
        g_preset = self.presets["cooling_gradient"]
        self.grad_solver = GradientDescentOptimizer(
            g_preset["expr"],
            g_preset["vars"],
            learning_rate=g_preset["lr"],
            tolerance=1e-3,
            max_iterations=80
        )
        self.grad_res = self.grad_solver.optimize(g_preset["initial"], mode="minimize")

        # 4. Lagrange
        self._update_lab_lagrange(self.lab_budget)

    def _update_lab_lagrange(self, budget: float) -> None:
        """Resuelve interactivamente Lagrange para cualquier presupuesto especificado."""
        self.lab_budget = max(100.0, min(600.0, budget))
        l_preset = self.presets["cluster_lagrange"]
        constraints = [f"P1 + P2 + P3 - {self.lab_budget:.1f}"]
        self.lagr_solver = ConstrainedLagrangeOptimizer(
            l_preset["objective"],
            constraints,
            l_preset["vars"]
        )
        self.lagr_res = self.lagr_solver.solve(mode="maximize")

    # -------------------------------------------------------------------------
    # CONTROL DE EVENTOS
    # -------------------------------------------------------------------------
    def handle_click(self, pos: tuple[int, int]) -> bool:
        """Gestiona clics en pestañas, controles de paso y botones de simulación."""
        # Botones de Cerrar [X] y Volver al Datacenter
        if self.btn_close_rect.collidepoint(pos) or self.btn_back_rect.collidepoint(pos):
            self.should_close = True
            return True

        # Clics en Pestañas
        for i, r in enumerate(self.tab_rects):
            if r.collidepoint(pos):
                self.active_tab = i + 1
                return True

        # Controles de navegación paso a paso
        if self.btn_prev_rect.collidepoint(pos):
            self.prev_step()
            return True

        if self.btn_next_rect.collidepoint(pos):
            self.next_step()
            return True

        if self.btn_auto_rect.collidepoint(pos):
            self.is_auto_playing = not self.is_auto_playing
            return True

        if self.btn_reset_rect.collidepoint(pos):
            self.reset_step()
            return True

        # Controles específicos de Tab 4 (Lagrange Budget)
        if self.active_tab == 4:
            if self.btn_lagr_minus.collidepoint(pos):
                self._update_lab_lagrange(self.lab_budget - 25.0)
                return True
            if self.btn_lagr_plus.collidepoint(pos):
                self._update_lab_lagrange(self.lab_budget + 25.0)
                return True

        # Controles específicos de Tab 5: Solucionador de Problemas Personalizados
        if self.active_tab == 5:
            # Botón Resolver
            if self.btn_solve_custom.collidepoint(pos):
                self.solve_custom_problem()
                return True

            # Botón Guardar en PDF
            if self.btn_save_custom_pdf.collidepoint(pos):
                self._save_current_problem_to_pdf()
                return True

            # Botón Limpiar Todo
            if self.btn_clear_custom.collidepoint(pos):
                self.input_values = {
                    "expr": "",
                    "vars": "x" if self.custom_model_type == "univariate" else "x, y",
                    "constraint": "",
                    "init_val": "0.0"
                }
                self.focused_field = "expr"
                self.custom_solution = None
                self.custom_error_msg = None
                return True

            # Botón Vaciar Restricciones
            if self.btn_clear_constraint.collidepoint(pos):
                self.input_values["constraint"] = ""
                self.focused_field = "constraint"
                self.solve_custom_problem()
                return True

            # Selección de Tipo de Modelo
            for m_type, rect in self.type_button_rects.items():
                if rect.collidepoint(pos):
                    if self.custom_model_type != m_type:
                        self.custom_model_type = m_type
                        if m_type == "univariate":
                            self.input_values["expr"] = "-2*x**2 + 12*x + 5"
                            self.input_values["vars"] = "x"
                        elif m_type == "multivariate":
                            self.input_values["expr"] = "x**2 + 2*y**2 - 4*x - 4*y + 10"
                            self.input_values["vars"] = "x, y"
                        elif m_type == "lagrange":
                            self.input_values["expr"] = "x*y"
                            self.input_values["vars"] = "x, y"
                            if not self.input_values.get("constraint"):
                                self.input_values["constraint"] = "x + 2*y = 20"
                        elif m_type == "gradient":
                            self.input_values["expr"] = "x**2 + y**2"
                            self.input_values["vars"] = "x, y"
                            self.input_values["init_val"] = "2.0, 2.0"
                        self.solve_custom_problem()
                    return True

            # Selección de Objetivo (Maximizar / Minimizar)
            for goal, rect in self.goal_button_rects.items():
                if rect.collidepoint(pos):
                    if self.custom_goal != goal:
                        self.custom_goal = goal
                        self.solve_custom_problem()
                    return True

            # Focus en campos de texto
            for field, rect in self.input_rects.items():
                if rect.collidepoint(pos):
                    self.focused_field = field
                    return True

            # Presets académicos rápidos
            for rect, preset in self.preset_button_rects:
                if rect.collidepoint(pos):
                    self._load_preset(preset)
                    return True

        return False

    def handle_key_event(self, event: pygame.event.Event) -> bool:
        """Gestiona eventos KEYDOWN completos para permitir escribir en los campos de texto."""
        if self.active_tab == 5 and self.focused_field:
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.solve_custom_problem()
                return True
            elif event.key == pygame.K_BACKSPACE:
                val = self.input_values.get(self.focused_field, "")
                if len(val) > 0:
                    self.input_values[self.focused_field] = val[:-1]
                return True
            elif event.key == pygame.K_TAB:
                # Rotar cíclicamente entre campos disponibles: siempre incluye 'constraint'
                if self.custom_model_type == "gradient":
                    cycle = ["expr", "vars", "constraint", "init_val"]
                else:
                    cycle = ["expr", "vars", "constraint"]

                if self.focused_field in cycle:
                    idx = (cycle.index(self.focused_field) + 1) % len(cycle)
                    self.focused_field = cycle[idx]
                else:
                    self.focused_field = cycle[0]
                return True
            elif event.key == pygame.K_ESCAPE:
                self.focused_field = None
                return True
            else:
                if event.unicode and len(event.unicode) == 1 and event.unicode >= ' ':
                    self.input_values[self.focused_field] = self.input_values.get(self.focused_field, "") + event.unicode
                    return True

        return self.handle_key(event.key)

    def handle_key(self, key: int) -> bool:
        """Gestiona atajos de teclado numéricos y de navegación."""
        if key in (pygame.K_1, pygame.K_KP1):
            self.active_tab = 1
            return True
        elif key in (pygame.K_2, pygame.K_KP2):
            self.active_tab = 2
            return True
        elif key in (pygame.K_3, pygame.K_KP3):
            self.active_tab = 3
            return True
        elif key in (pygame.K_4, pygame.K_KP4):
            self.active_tab = 4
            return True
        elif key in (pygame.K_5, pygame.K_KP5):
            self.active_tab = 5
            return True
        elif key in (pygame.K_LEFT, pygame.K_a):
            self.prev_step()
            return True
        elif key in (pygame.K_RIGHT, pygame.K_d):
            self.next_step()
            return True
        elif key == pygame.K_SPACE:
            self.is_auto_playing = not self.is_auto_playing
            return True
        return False

    def next_step(self) -> None:
        """Avanza al siguiente paso de la demostración actual."""
        if self.active_tab == 1:
            self.step_1d = min(4, self.step_1d + 1)
        elif self.active_tab == 2:
            self.step_hessian = min(5, self.step_hessian + 1)
        elif self.active_tab == 3:
            max_g = len(self.grad_res.history) - 1
            self.step_grad = min(max_g, self.step_grad + 1)
        elif self.active_tab == 4:
            self.step_lagr = min(5, self.step_lagr + 1)

    def prev_step(self) -> None:
        """Retrocede al paso anterior."""
        if self.active_tab == 1:
            self.step_1d = max(1, self.step_1d - 1)
        elif self.active_tab == 2:
            self.step_hessian = max(1, self.step_hessian - 1)
        elif self.active_tab == 3:
            self.step_grad = max(0, self.step_grad - 1)
        elif self.active_tab == 4:
            self.step_lagr = max(1, self.step_lagr - 1)

    def reset_step(self) -> None:
        """Reinicia los pasos del método activo."""
        if self.active_tab == 1:
            self.step_1d = 1
        elif self.active_tab == 2:
            self.step_hessian = 1
        elif self.active_tab == 3:
            self.step_grad = 0
        elif self.active_tab == 4:
            self.step_lagr = 1

    # -------------------------------------------------------------------------
    # RENDER PRINCIPAL
    # -------------------------------------------------------------------------
    def render(self, surface: pygame.Surface) -> None:
        """Dibuja el panel completo del Laboratorio No Lineal con animaciones."""
        now = time.time()

        # Actualizar auto-play periódicamente
        if self.is_auto_playing and now - self.last_auto_tick > 1.6:
            self.last_auto_tick = now
            if self.active_tab == 3:
                max_g = len(self.grad_res.history) - 1
                if self.step_grad >= max_g:
                    self.step_grad = 0
                else:
                    self.step_grad += 1
            else:
                curr, max_s = self._get_current_step_info()
                if curr >= max_s:
                    self.reset_step()
                else:
                    self.next_step()

        # Fondo oscuro del modal
        overlay = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 185))
        surface.blit(overlay, (0, 0))

        # Tarjeta principal
        pygame.draw.rect(surface, PANEL_BG, self.rect, border_radius=12)
        pygame.draw.rect(surface, BORDER_ACTIVE, self.rect, width=2, border_radius=12)

        # Encabezado
        title_surf = self.fonts["title"].render(
            "LABORATORIO DE PROGRAMACIÓN NO LINEAL - MÉTODOS CUANTITATIVOS (UJAP)",
            True,
            ROUTER_ACTIVE
        )
        surface.blit(title_surf, (self.rect.x + 24, self.rect.y + 16))

        sub_surf = self.fonts["small"].render(
            "Navegación paso a paso: Usa [<< Anterior] / [Siguiente >>] o teclas [A]/[D] | Presiona [N] o [ESC] para volver al Datacenter",
            True,
            TEXT_SECONDARY
        )
        surface.blit(sub_surf, (self.rect.x + 24, self.rect.y + 40))

        # --- Botón Volver al Datacenter ---
        pygame.draw.rect(surface, (25, 32, 44), self.btn_back_rect, border_radius=6)
        pygame.draw.rect(surface, (88, 166, 255), self.btn_back_rect, width=1, border_radius=6)
        t_back = self.fonts["regular"].render("<< Volver [ESC]", True, (140, 205, 255))
        surface.blit(t_back, t_back.get_rect(center=self.btn_back_rect.center))

        # --- Botón [X] Cerrar ---
        pygame.draw.rect(surface, (60, 22, 26), self.btn_close_rect, border_radius=6)
        pygame.draw.rect(surface, (248, 81, 73), self.btn_close_rect, width=1, border_radius=6)
        cx, cy = self.btn_close_rect.center
        pygame.draw.line(surface, (255, 220, 220), (cx - 6, cy - 6), (cx + 6, cy + 6), 2)
        pygame.draw.line(surface, (255, 220, 220), (cx + 6, cy - 6), (cx - 6, cy + 6), 2)

        # Renderizar pestañas superiores y barra de control
        self._render_tabs(surface)
        self._render_step_controls(surface)

        # Área de contenido
        content_rect = pygame.Rect(
            self.rect.x + 20,
            self.rect.y + 140,
            self.rect.width - 40,
            self.rect.height - 158
        )

        if self.active_tab == 1:
            self._render_tab_1d(surface, content_rect, now)
        elif self.active_tab == 2:
            self._render_tab_hessian(surface, content_rect)
        elif self.active_tab == 3:
            self._render_tab_gradient(surface, content_rect, now)
        elif self.active_tab == 4:
            self._render_tab_lagrange(surface, content_rect)
        elif self.active_tab == 5:
            self._render_tab_custom(surface, content_rect, now)

    def _get_current_step_info(self) -> tuple[int, int]:
        """Retorna (paso_actual, total_pasos) según la pestaña activa."""
        if self.active_tab == 1:
            return self.step_1d, 4
        elif self.active_tab == 2:
            return self.step_hessian, 5
        elif self.active_tab == 3:
            return self.step_grad, len(self.grad_res.history) - 1
        elif self.active_tab == 4:
            return self.step_lagr, 5
        else:
            return 1, 1

    def _render_tabs(self, surface: pygame.Surface) -> None:
        """Renderiza las 5 pestañas superiores."""
        tab_names = [
            "[1] 1D: Térmica",
            "[2] 2D: Hessiano",
            "[3] Gradiente",
            "[4] Lagrange",
            "[5] + Nuevo Problema"
        ]

        for i, rect in enumerate(self.tab_rects):
            is_active = (self.active_tab == i + 1)
            bg = BORDER_ACTIVE if is_active else CARD_BG
            text_col = (13, 17, 23) if is_active else TEXT_PRIMARY

            pygame.draw.rect(surface, bg, rect, border_radius=6)
            if not is_active:
                pygame.draw.rect(surface, BORDER_COLOR, rect, width=1, border_radius=6)

            txt = self.fonts["regular"].render(tab_names[i], True, text_col)
            surface.blit(txt, txt.get_rect(center=rect.center))

    def _render_step_controls(self, surface: pygame.Surface) -> None:
        """Renderiza la barra de control interactivo dibujando iconos vectoriales limpios."""
        ctrl_y = self.btn_prev_rect.y

        # Barra específica para Tab 5: Solucionador de Problemas Personalizados
        if self.active_tab == 5:
            self.btn_solve_custom = pygame.Rect(self.rect.x + 20, ctrl_y, 160, 30)
            pygame.draw.rect(surface, (16, 185, 129), self.btn_solve_custom, border_radius=6)
            t_solve = self.fonts["header"].render("▶ Resolver [ENTER]", True, (10, 25, 20))
            surface.blit(t_solve, t_solve.get_rect(center=self.btn_solve_custom.center))

            self.btn_save_custom_pdf = pygame.Rect(self.rect.x + 190, ctrl_y, 195, 30)
            saved_count = len(self.custom_saved_problems)
            btn_pdf_text = f"💾 Añadir a PDF ({saved_count})" if saved_count > 0 else "💾 Añadir a PDF"
            pygame.draw.rect(surface, (30, 41, 59), self.btn_save_custom_pdf, border_radius=6)
            pygame.draw.rect(surface, (56, 189, 248), self.btn_save_custom_pdf, width=1, border_radius=6)
            t_pdf = self.fonts["regular"].render(btn_pdf_text, True, (56, 189, 248))
            surface.blit(t_pdf, t_pdf.get_rect(center=self.btn_save_custom_pdf.center))

            self.btn_clear_custom = pygame.Rect(self.rect.x + 395, ctrl_y, 105, 30)
            pygame.draw.rect(surface, (25, 30, 40), self.btn_clear_custom, border_radius=6)
            pygame.draw.rect(surface, BORDER_COLOR, self.btn_clear_custom, width=1, border_radius=6)
            t_clr = self.fonts["regular"].render("Limpiar", True, TEXT_SECONDARY)
            surface.blit(t_clr, t_clr.get_rect(center=self.btn_clear_custom.center))

            if self.custom_toast_msg and time.time() < self.custom_toast_timer:
                t_toast = self.fonts["header"].render(self.custom_toast_msg, True, (255, 215, 0))
                surface.blit(t_toast, (self.rect.x + 515, ctrl_y + 6))
            else:
                hint_txt = self.fonts["small"].render("Define la función objetivo f y presiona ENTER para resolver analíticamente", True, TEXT_MUTED)
                surface.blit(hint_txt, (self.rect.x + 515, ctrl_y + 8))
            return

        curr, total = self._get_current_step_info()
        ctrl_y = self.btn_prev_rect.y

        # --- Botón Anterior ---
        can_prev = curr > 1 or (self.active_tab == 3 and curr > 0)
        col_prev = TEXT_PRIMARY if can_prev else TEXT_MUTED
        pygame.draw.rect(surface, CARD_BG, self.btn_prev_rect, border_radius=6)
        pygame.draw.rect(surface, BORDER_COLOR, self.btn_prev_rect, width=1, border_radius=6)

        # Dibujar flecha vectorial izquierda
        ay = self.btn_prev_rect.centery
        ax = self.btn_prev_rect.x + 14
        pygame.draw.polygon(surface, col_prev, [(ax, ay), (ax + 7, ay - 5), (ax + 7, ay + 5)])
        t_prev = self.fonts["regular"].render("Anterior [A]", True, col_prev)
        surface.blit(t_prev, (self.btn_prev_rect.x + 27, self.btn_prev_rect.centery - t_prev.get_height() // 2))

        # --- Botón Siguiente ---
        can_next = curr < total
        col_next = TEXT_PRIMARY if can_next else TEXT_MUTED
        pygame.draw.rect(surface, CARD_BG, self.btn_next_rect, border_radius=6)
        pygame.draw.rect(surface, BORDER_COLOR, self.btn_next_rect, width=1, border_radius=6)

        t_next = self.fonts["regular"].render("Siguiente [D]", True, col_next)
        surface.blit(t_next, (self.btn_next_rect.x + 14, self.btn_next_rect.centery - t_next.get_height() // 2))
        # Dibujar flecha vectorial derecha
        anx = self.btn_next_rect.right - 14
        pygame.draw.polygon(surface, col_next, [(anx, ay), (anx - 7, ay - 5), (anx - 7, ay + 5)])

        # --- Botón Auto-play ---
        auto_bg = (30, 75, 45) if self.is_auto_playing else CARD_BG
        col_auto = (200, 255, 200) if self.is_auto_playing else TEXT_PRIMARY
        pygame.draw.rect(surface, auto_bg, self.btn_auto_rect, border_radius=6)
        pygame.draw.rect(surface, SERVER_NORMAL if self.is_auto_playing else BORDER_COLOR, self.btn_auto_rect, width=1, border_radius=6)

        if self.is_auto_playing:
            # Dibujar icono de pausa (dos barras verticales)
            pbx = self.btn_auto_rect.x + 14
            pby = self.btn_auto_rect.centery - 6
            pygame.draw.rect(surface, col_auto, (pbx, pby, 3, 12))
            pygame.draw.rect(surface, col_auto, (pbx + 6, pby, 3, 12))
            t_auto = self.fonts["regular"].render("Pausar [ESP]", True, col_auto)
            surface.blit(t_auto, (self.btn_auto_rect.x + 30, self.btn_auto_rect.centery - t_auto.get_height() // 2))
        else:
            # Dibujar icono de play (triángulo)
            plx = self.btn_auto_rect.x + 14
            ply = self.btn_auto_rect.centery
            pygame.draw.polygon(surface, col_auto, [(plx, ply - 6), (plx, ply + 6), (plx + 9, ply)])
            t_auto = self.fonts["regular"].render("Animar Auto", True, col_auto)
            surface.blit(t_auto, (self.btn_auto_rect.x + 28, self.btn_auto_rect.centery - t_auto.get_height() // 2))

        # --- Botón Reset ---
        pygame.draw.rect(surface, CARD_BG, self.btn_reset_rect, border_radius=6)
        pygame.draw.rect(surface, BORDER_COLOR, self.btn_reset_rect, width=1, border_radius=6)
        t_res = self.fonts["regular"].render("Reiniciar", True, TEXT_SECONDARY)
        surface.blit(t_res, t_res.get_rect(center=self.btn_reset_rect.center))

        # --- Título descriptivo del paso actual con notación matemática formal ---
        step_titles_1d = [
            "Paso 1/4: Función de Temperatura T(L) y Comportamiento Base",
            "Paso 2/4: Primera Derivada T'(L) y Tasa de Variación Positiva (Creciente)",
            "Paso 3/4: Segunda Derivada d²T/dL² y Determinación del Punto de Inflexión (60%)",
            "Paso 4/4: Delimitación de Regímenes y Protocolo de Enfriamiento de Emergencia"
        ]
        step_titles_hess = [
            "Paso 1/5: Función Objetivo de Disipación Energética E(f, v)",
            "Paso 2/5: Vector Gradiente ∇E = [0, 0]ᵀ (Condición de 1.ᵉʳ Orden)",
            "Paso 3/5: Resolución Analítica y Localización del Punto Crítico (f*, v*)",
            "Paso 4/5: Construcción de la Matriz Hessiana Simbólica H",
            "Paso 5/5: Criterio de Sylvester (Menores Principales) y Veredicto de Mínimo"
        ]
        step_titles_lagr = [
            "Paso 1/5: Función de Rendimiento Global R(P) con Rendimientos Decrecientes",
            "Paso 2/5: Restricción de Presupuesto Eléctrico Fijo g(P) = 0",
            "Paso 3/5: Ensamblaje de la Función Lagrangiana L(P, λ)",
            "Paso 4/5: Condiciones Necesarias de Primer Orden ∇L = 0",
            "Paso 5/5: Simulador de Sensibilidad: Variación de Presupuesto y Multiplicador λ*"
        ]

        if self.active_tab == 1:
            lbl_desc = step_titles_1d[curr - 1]
        elif self.active_tab == 2:
            lbl_desc = step_titles_hess[curr - 1]
        elif self.active_tab == 3:
            lbl_desc = f"Paso {curr}/{total}: Iteración k = {curr} del Descenso de Gradiente | P = {self.grad_res.history[curr].coords['P']:.2f} W"
        else:
            lbl_desc = step_titles_lagr[curr - 1]

        lbl_surf = self.fonts["header"].render(lbl_desc, True, (255, 215, 0))
        surface.blit(lbl_surf, (self.rect.x + 515, ctrl_y + 6))

    # =========================================================================
    # TAB 1: MODELO 1D (ANÁLISIS TÉRMICO E INFLEXIÓN)
    # =========================================================================
    def _render_tab_1d(self, surface: pygame.Surface, rect: pygame.Rect, now: float) -> None:
        left_w = int(rect.width * 0.46)
        right_w = rect.width - left_w - 20

        # Panel Izquierdo: Fórmulas y Pasos
        left_rect = pygame.Rect(rect.x, rect.y, left_w, rect.height)
        pygame.draw.rect(surface, CARD_BG, left_rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, left_rect, width=1, border_radius=8)

        y = left_rect.y + 14

        # Paso 1: Función
        p1_col = (255, 215, 0) if self.step_1d == 1 else TEXT_PRIMARY
        surface.blit(self.fonts["header"].render("1. Función de Temperatura T(L):", True, p1_col), (left_rect.x + 16, y))
        y += 22
        surface.blit(self.fonts["mono_bold"].render("  T(L) = 30L³ - 54L² + 65L + 22", True, ROUTER_ACTIVE), (left_rect.x + 16, y))
        y += 20
        surface.blit(self.fonts["small"].render("  Donde L ∈ [0, 1] es el factor de carga computacional de CPU/RAM.", True, TEXT_MUTED), (left_rect.x + 16, y))
        y += 28

        # Paso 2: Derivada 1
        p2_col = (255, 215, 0) if self.step_1d == 2 else TEXT_PRIMARY
        surface.blit(self.fonts["header"].render("2. Primera Derivada T'(L) (Tasa de Variación):", True, p2_col), (left_rect.x + 16, y))
        y += 22
        surface.blit(self.fonts["mono_bold"].render("  T'(L) = 90L² - 108L + 65", True, SERVER_NORMAL), (left_rect.x + 16, y))
        y += 20
        surface.blit(self.fonts["small"].render("  T'(L) > 0, ∀ L ∈ [0, 1] → Calentamiento monótono positivo.", True, TEXT_MUTED), (left_rect.x + 16, y))
        y += 28

        # Paso 3: Derivada 2 e Inflexión
        p3_col = (255, 215, 0) if self.step_1d == 3 else TEXT_PRIMARY
        surface.blit(self.fonts["header"].render("3. Segunda Derivada d²T/dL² y Punto de Inflexión:", True, p3_col), (left_rect.x + 16, y))
        y += 22
        surface.blit(self.fonts["mono_bold"].render("  d²T/dL² = 180L - 108 = 0", True, CHART_LINE_COST), (left_rect.x + 16, y))
        y += 20
        surface.blit(self.fonts["mono_bold"].render("  → 180L = 108  →  L_inflexión = 0.60 (60% de Carga)", True, SERVER_EMERGENCY), (left_rect.x + 16, y))
        y += 20
        surface.blit(self.fonts["small"].render("  Temperatura en la inflexión: T(0.60) = 48.04 °C", True, TEXT_SECONDARY), (left_rect.x + 16, y))
        y += 28

        # Paso 4: Dictamen y Protocolo
        p4_col = (255, 215, 0) if self.step_1d == 4 else TEXT_PRIMARY
        surface.blit(self.fonts["header"].render("4. Régimen Superlineal y Alerta Operativa:", True, p4_col), (left_rect.x + 16, y))
        y += 22
        surface.blit(self.fonts["regular"].render("  • Si L < 60%: d²T/dL² < 0 (Convexa hacia abajo, régimen seguro).", True, SERVER_NORMAL), (left_rect.x + 16, y))
        y += 18
        surface.blit(self.fonts["regular"].render("  • Si L ≥ 60%: d²T/dL² > 0 (Aceleración superlineal descontrolada).", True, SERVER_EMERGENCY), (left_rect.x + 16, y))
        y += 22
        box_prot = pygame.Rect(left_rect.x + 16, y, left_w - 32, 34)
        pygame.draw.rect(surface, (50, 15, 18), box_prot, border_radius=6)
        pygame.draw.rect(surface, SERVER_EMERGENCY, box_prot, width=1, border_radius=6)
        t_em = self.fonts["mono_bold"].render("ALERTA: Protocolo de Enfriamiento de Emergencia activo si L ≥ 60%", True, (255, 180, 180))
        surface.blit(t_em, t_em.get_rect(center=box_prot.center))

        # Panel Derecho: Gráfica Dinámica
        right_rect = pygame.Rect(rect.x + left_w + 20, rect.y, right_w, rect.height)
        pygame.draw.rect(surface, CARD_BG, right_rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, right_rect, width=1, border_radius=8)

        g_title = self.fonts["header"].render("Visualización Dinámica de la Curva Térmica T(L)", True, TEXT_PRIMARY)
        surface.blit(g_title, (right_rect.x + 16, right_rect.y + 14))

        plot_rect = pygame.Rect(right_rect.x + 50, right_rect.y + 46, right_rect.width - 70, right_rect.height - 76)
        pygame.draw.rect(surface, (15, 18, 24), plot_rect)
        pygame.draw.rect(surface, CHART_GRID, plot_rect, width=1)

        # Regiones sombreadas en Paso 4
        if self.step_1d == 4:
            split_x = plot_rect.x + int(0.60 * plot_rect.width)
            # Zona Segura (Verde)
            safe_surf = pygame.Surface((split_x - plot_rect.x, plot_rect.height), pygame.SRCALPHA)
            safe_surf.fill((35, 134, 54, 45))
            surface.blit(safe_surf, (plot_rect.x, plot_rect.y))
            # Zona Crítica (Roja)
            crit_surf = pygame.Surface((plot_rect.right - split_x, plot_rect.height), pygame.SRCALPHA)
            crit_surf.fill((218, 54, 51, 60))
            surface.blit(crit_surf, (split_x, plot_rect.y))

            surface.blit(self.fonts["small"].render("ZONA SEGURA (0% - 60%)", True, SERVER_NORMAL), (plot_rect.x + 10, plot_rect.y + 10))
            surface.blit(self.fonts["small"].render("ZONA CRÍTICA SUPERLINEAL (≥ 60%)", True, SERVER_EMERGENCY), (split_x + 10, plot_rect.y + 10))

        # Muestreo de la curva
        steps = 60
        points = []
        for i in range(steps + 1):
            L_val = i / steps
            T_val = self.uni_solver.evaluate(L_val)
            px = plot_rect.x + int(L_val * plot_rect.width)
            py = plot_rect.bottom - int(((T_val - 20.0) / (70.0 - 20.0)) * plot_rect.height)
            points.append((px, py))

        if len(points) > 1:
            pygame.draw.lines(surface, ROUTER_ACTIVE, False, points, 3)

        # Paso 2: Recta Tangente Animada deslizándose
        if self.step_1d == 2:
            tang_L = 0.5 + 0.4 * math.sin(now * 1.8)
            tang_T = self.uni_solver.evaluate(tang_L)
            tang_px = plot_rect.x + int(tang_L * plot_rect.width)
            tang_py = plot_rect.bottom - int(((tang_T - 20.0) / (70.0 - 20.0)) * plot_rect.height)

            # Pendiente T'(L)
            slope = 90.0 * (tang_L ** 2) - 108.0 * tang_L + 65.0
            dx = 35
            dy = int(slope * 0.4)
            pygame.draw.line(surface, SERVER_NORMAL, (tang_px - dx, tang_py + dy), (tang_px + dx, tang_py - dy), 3)
            pygame.draw.circle(surface, (255, 255, 255), (tang_px, tang_py), 5)
            pygame.draw.circle(surface, SERVER_NORMAL, (tang_px, tang_py), 3)

            lbl_tang = self.fonts["small"].render(f"Tangente: T'({tang_L:.2f}) = +{slope:.1f} > 0", True, SERVER_NORMAL)
            surface.blit(lbl_tang, (tang_px - 40, tang_py - 24))

        # Paso 3 o 4: Punto de Inflexión marcado
        if self.step_1d >= 3:
            inf_px = plot_rect.x + int(0.60 * plot_rect.width)
            inf_py = plot_rect.bottom - int(((48.04 - 20.0) / (70.0 - 20.0)) * plot_rect.height)

            # Líneas guía
            pygame.draw.line(surface, (140, 150, 160), (inf_px, plot_rect.bottom), (inf_px, inf_py), 1)
            pygame.draw.line(surface, (140, 150, 160), (plot_rect.x, inf_py), (inf_px, inf_py), 1)

            # Punto parpadeante
            pulse = int(abs(math.sin(now * 6.0)) * 4) + 5
            pygame.draw.circle(surface, SERVER_EMERGENCY, (inf_px, inf_py), pulse)
            pygame.draw.circle(surface, (255, 255, 255), (inf_px, inf_py), 3)

            surface.blit(self.fonts["mono_bold"].render("PUNTO DE INFLEXIÓN (60%, 48.04 °C)", True, SERVER_EMERGENCY), (inf_px - 85, inf_py - 24))

        # Etiquetas de Ejes
        surface.blit(self.fonts["small"].render("70 °C", True, TEXT_MUTED), (right_rect.x + 12, plot_rect.y))
        surface.blit(self.fonts["small"].render("48.04 °C", True, SERVER_EMERGENCY), (right_rect.x + 4, plot_rect.bottom - int(((48.04 - 20.0) / (70.0 - 20.0)) * plot_rect.height) - 6))
        surface.blit(self.fonts["small"].render("20 °C", True, TEXT_MUTED), (right_rect.x + 12, plot_rect.bottom - 10))

        surface.blit(self.fonts["small"].render("0% Carga", True, TEXT_MUTED), (plot_rect.x, plot_rect.bottom + 4))
        surface.blit(self.fonts["small"].render("60% Inflexión", True, SERVER_EMERGENCY), (plot_rect.x + int(0.60 * plot_rect.width) - 25, plot_rect.bottom + 4))
        surface.blit(self.fonts["small"].render("100% Carga", True, TEXT_MUTED), (plot_rect.right - 55, plot_rect.bottom + 4))

    # =========================================================================
    # TAB 2: MULTIVARIABLE CON MATRIZ HESSIANA Y SYLVESTER
    # =========================================================================
    def _render_tab_hessian(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        left_w = int(rect.width * 0.48)
        right_w = rect.width - left_w - 20

        # Panel Izquierdo: Pasos 1 a 3
        left_rect = pygame.Rect(rect.x, rect.y, left_w, rect.height)
        pygame.draw.rect(surface, CARD_BG, left_rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, left_rect, width=1, border_radius=8)

        y = left_rect.y + 14

        # Paso 1: Función
        p1_col = (255, 215, 0) if self.step_hessian == 1 else TEXT_PRIMARY
        surface.blit(self.fonts["header"].render("1. Función de Disipación de CPU E(f, v):", True, p1_col), (left_rect.x + 16, y))
        y += 22
        surface.blit(self.fonts["mono_bold"].render("  E(f, v) = 4f² + 6v² - 2fv - 24f - 30v + 100", True, ROUTER_ACTIVE), (left_rect.x + 16, y))
        y += 20
        surface.blit(self.fonts["small"].render("  f = Frecuencia de reloj (GHz), v = Voltaje de núcleo (V)", True, TEXT_MUTED), (left_rect.x + 16, y))
        y += 28

        # Paso 2: Gradiente
        p2_col = (255, 215, 0) if self.step_hessian == 2 else TEXT_PRIMARY
        surface.blit(self.fonts["header"].render("2. Vector Gradiente ∇E = [0, 0]ᵀ (1.ᵉʳ Orden):", True, p2_col), (left_rect.x + 16, y))
        y += 22
        surface.blit(self.fonts["mono_bold"].render("  • ∂E/∂f = 8f - 2v - 24 = 0", True, SERVER_NORMAL), (left_rect.x + 16, y))
        y += 18
        surface.blit(self.fonts["mono_bold"].render("  • ∂E/∂v = 12v - 2f - 30 = 0", True, SERVER_NORMAL), (left_rect.x + 16, y))
        y += 28

        # Paso 3: Punto Crítico
        p3_col = (255, 215, 0) if self.step_hessian == 3 else TEXT_PRIMARY
        surface.blit(self.fonts["header"].render("3. Resolución Analítica y Punto Estacionario:", True, p3_col), (left_rect.x + 16, y))
        y += 22
        surface.blit(self.fonts["mono_bold"].render("  → f* = 3.80 GHz  |  v* = 3.13 V", True, (255, 215, 0)), (left_rect.x + 16, y))
        y += 18
        surface.blit(self.fonts["regular"].render("  Disipación en el óptimo: E(3.80, 3.13) = 7.39 unidades", True, TEXT_SECONDARY), (left_rect.x + 16, y))

        # Panel Derecho: Matriz Hessiana y Sylvester (Pasos 4 y 5)
        right_rect = pygame.Rect(rect.x + left_w + 20, rect.y, right_w, rect.height)
        pygame.draw.rect(surface, CARD_BG, right_rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, right_rect, width=1, border_radius=8)

        ry = right_rect.y + 14

        # Paso 4: Matriz Hessiana
        p4_col = (255, 215, 0) if self.step_hessian == 4 else TEXT_PRIMARY
        surface.blit(self.fonts["header"].render("4. Matriz Hessiana H (2.º Orden):", True, p4_col), (right_rect.x + 16, ry))
        ry += 24

        box_h = pygame.Rect(right_rect.x + 24, ry, right_w - 48, 64)
        pygame.draw.rect(surface, (18, 22, 28), box_h, border_radius=6)
        pygame.draw.rect(surface, (255, 166, 87) if self.step_hessian == 4 else BORDER_COLOR, box_h, width=1, border_radius=6)

        surface.blit(self.fonts["mono_bold"].render("       [  ∂²E/∂f²     ∂²E/∂f∂v  ]       [  8   -2  ]", True, CHART_LINE_COST), (box_h.x + 20, box_h.y + 12))
        surface.blit(self.fonts["mono_bold"].render("  H =  [                        ]  =    [          ]", True, CHART_LINE_COST), (box_h.x + 20, box_h.y + 26))
        surface.blit(self.fonts["mono_bold"].render("       [  ∂²E/∂v∂f     ∂²E/∂v²  ]       [ -2   12  ]", True, CHART_LINE_COST), (box_h.x + 20, box_h.y + 40))
        ry += 80

        # Paso 5: Criterio de Sylvester
        p5_col = (255, 215, 0) if self.step_hessian == 5 else TEXT_PRIMARY
        surface.blit(self.fonts["header"].render("5. Criterio de Sylvester (Menores Principales):", True, p5_col), (right_rect.x + 16, ry))
        ry += 24

        surface.blit(self.fonts["mono_bold"].render("  • Menor Principal 1:  Δ₁ = |H₁₁| = 8 > 0", True, SERVER_NORMAL), (right_rect.x + 20, ry))
        ry += 20
        surface.blit(self.fonts["mono_bold"].render("  • Menor Principal 2:  Δ₂ = det(H) = (8)(12) - (-2)² = 92 > 0", True, SERVER_NORMAL), (right_rect.x + 20, ry))
        ry += 26

        # Veredicto de Sylvester animado
        verdict_box = pygame.Rect(right_rect.x + 16, ry, right_w - 32, 54)
        is_highlight = (self.step_hessian == 5)
        bg_v = (20, 50, 32) if is_highlight else (18, 22, 28)
        border_v = SERVER_NORMAL if is_highlight else BORDER_COLOR
        pygame.draw.rect(surface, bg_v, verdict_box, border_radius=8)
        pygame.draw.rect(surface, border_v, verdict_box, width=2 if is_highlight else 1, border_radius=8)

        v_t1 = self.fonts["header"].render("VEREDICTO: MATRIZ DEFINIDA POSITIVA (Δ₁ > 0 y Δ₂ > 0)", True, SERVER_NORMAL)
        v_t2 = self.fonts["mono_bold"].render("→ EL PUNTO (3.80 GHz, 3.13 V) ES UN MÍNIMO LOCAL ESTRICTO", True, (255, 235, 120))
        surface.blit(v_t1, (verdict_box.x + 16, verdict_box.y + 8))
        surface.blit(v_t2, (verdict_box.x + 16, verdict_box.y + 28))

    # =========================================================================
    # TAB 3: MÉTODO DEL GRADIENTE DESCENDENTE (INTERACTIVO PASO A PASO)
    # =========================================================================
    def _render_tab_gradient(self, surface: pygame.Surface, rect: pygame.Rect, now: float) -> None:
        left_w = int(rect.width * 0.44)
        right_w = rect.width - left_w - 20

        hist = self.grad_res.history
        k = min(self.step_grad, len(hist) - 1)
        curr_it = hist[k]

        # Panel Izquierdo: Estado de la iteración actual
        left_rect = pygame.Rect(rect.x, rect.y, left_w, rect.height)
        pygame.draw.rect(surface, CARD_BG, left_rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, left_rect, width=1, border_radius=8)

        y = left_rect.y + 14
        surface.blit(self.fonts["header"].render(f"Iteración Actual: Paso k = {k}", True, (255, 215, 0)), (left_rect.x + 16, y))
        y += 24

        # Estado en vivo
        stats = [
            ("Potencia del Ventilador P:", f"{curr_it.coords['P']:.2f} W", ROUTER_ACTIVE),
            ("Valor de Costo J(P):", f"{curr_it.f_val:.4f}", CHART_LINE_COST),
            ("Norma del Gradiente ||∇J||:", f"{curr_it.grad_norm:.6f}", SERVER_NORMAL),
            ("Tasa de Aprendizaje α:", "0.08", TEXT_SECONDARY),
        ]
        for lbl, val, col in stats:
            surface.blit(self.fonts["var_label"].render(lbl, True, TEXT_SECONDARY), (left_rect.x + 16, y))
            y += 18
            surface.blit(self.fonts["mono_bold"].render(f"  {val}", True, col), (left_rect.x + 16, y))
            y += 22

        # Ecuación de actualización sustituida numéricamente
        y += 6
        box_eq = pygame.Rect(left_rect.x + 16, y, left_w - 32, 52)
        pygame.draw.rect(surface, (18, 22, 28), box_eq, border_radius=6)
        pygame.draw.rect(surface, BORDER_COLOR, box_eq, width=1, border_radius=6)

        surface.blit(self.fonts["small"].render("Ecuación de Actualización Iterativa (P^(k+1) = P^k - α · ∇J):", True, TEXT_MUTED), (box_eq.x + 10, box_eq.y + 6))
        eq_txt = f"P^({k+1}) = {curr_it.coords['P']:.2f} - (0.08 · ∇J)"
        surface.blit(self.fonts["mono_bold"].render(eq_txt, True, (255, 215, 0)), (box_eq.x + 10, box_eq.y + 24))
        y += 64

        # Tabla de los últimos pasos
        surface.blit(self.fonts["header"].render("Historial Reciente de Convergencia:", True, TEXT_PRIMARY), (left_rect.x + 16, y))
        y += 22

        header_txt = f"{'Paso':<6} {'P (W)':<10} {'Costo J(P)':<14} {'||∇J||':<10}"
        surface.blit(self.fonts["mono_bold"].render(header_txt, True, ROUTER_ACTIVE), (left_rect.x + 16, y))
        y += 16

        start_show = max(0, k - 4)
        for it in hist[start_show:k + 1]:
            is_active = (it.step == k)
            row_col = (255, 215, 0) if is_active else TEXT_MUTED
            row_txt = f"{it.step:<6} {it.coords['P']:<10.2f} {it.f_val:<14.2f} {it.grad_norm:<10.4f}"
            surface.blit(self.fonts["mono"].render(row_txt, True, row_col), (left_rect.x + 16, y))
            y += 16

        # Panel Derecho: Curva de la Parábola de Costo y Bolita Rodante
        right_rect = pygame.Rect(rect.x + left_w + 20, rect.y, right_w, rect.height)
        pygame.draw.rect(surface, CARD_BG, right_rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, right_rect, width=1, border_radius=8)

        g_title = self.fonts["header"].render("Trayectoria Dinámica sobre la Parábola de Costo J(P)", True, TEXT_PRIMARY)
        surface.blit(g_title, (right_rect.x + 16, right_rect.y + 14))

        plot_rect = pygame.Rect(right_rect.x + 40, right_rect.y + 46, right_rect.width - 60, right_rect.height - 76)
        pygame.draw.rect(surface, (15, 18, 24), plot_rect)
        pygame.draw.rect(surface, CHART_GRID, plot_rect, width=1)

        # Muestrear curva de costo J(P) para P in [5, 120]
        pts_curve = []
        p_min, p_max = 5.0, 110.0
        # Simular costo J(P) para el servidor Web con T_base = 85°C
        for i in range(50):
            p_val = p_min + (i / 49.0) * (p_max - p_min)
            t_net = 85.0 - (0.45 * p_val)
            excess = max(0.0, t_net - 60.0)
            c_val = 0.04 * (p_val ** 2) + 2.0 * (excess ** 2)

            px = plot_rect.x + int(((p_val - p_min) / (p_max - p_min)) * plot_rect.width)
            py = plot_rect.bottom - int((min(1200.0, c_val) / 1200.0) * plot_rect.height)
            pts_curve.append((px, py))

        if len(pts_curve) > 1:
            pygame.draw.lines(surface, (60, 75, 95), False, pts_curve, 2)

        # Estela de los pasos dados hasta k
        visited_pts = []
        for step_i in range(k + 1):
            p_it = hist[step_i].coords['P']
            c_it = hist[step_i].f_val
            vx = plot_rect.x + int(((p_it - p_min) / (p_max - p_min)) * plot_rect.width)
            vy = plot_rect.bottom - int((min(1200.0, c_it) / 1200.0) * plot_rect.height)
            visited_pts.append((vx, vy))

        if len(visited_pts) > 1:
            pygame.draw.lines(surface, CHART_LINE_GRADIENT, False, visited_pts, 3)

        # Bolita Rodante en la posición actual
        curr_p = curr_it.coords['P']
        curr_c = curr_it.f_val
        bx = plot_rect.x + int(((curr_p - p_min) / (p_max - p_min)) * plot_rect.width)
        by = plot_rect.bottom - int((min(1200.0, curr_c) / 1200.0) * plot_rect.height)

        pulse = int(abs(math.sin(now * 8.0)) * 4) + 6
        pygame.draw.circle(surface, (255, 215, 0), (bx, by), pulse)
        pygame.draw.circle(surface, (255, 255, 255), (bx, by), 3)

        # Vector tangente en dirección opuesta al gradiente
        if curr_it.grad_norm > 0.05:
            arrow_dx = -int(math.copysign(28, curr_it.grad_norm))
            pygame.draw.line(surface, (255, 80, 80), (bx, by), (bx + arrow_dx, by + 12), 3)

        lbl_ball = self.fonts["small"].render(f"P({k}) = {curr_p:.1f} W", True, (255, 215, 0))
        surface.blit(lbl_ball, (bx - 25, by - 22))

    # =========================================================================
    # TAB 4: MULTIPLICADORES DE LAGRANGE Y SIMULADOR DE PRESUPUESTO
    # =========================================================================
    def _render_tab_lagrange(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        left_w = int(rect.width * 0.46)
        right_w = rect.width - left_w - 20

        # Panel Izquierdo: Formulación Simbólica
        left_rect = pygame.Rect(rect.x, rect.y, left_w, rect.height)
        pygame.draw.rect(surface, CARD_BG, left_rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, left_rect, width=1, border_radius=8)

        y = left_rect.y + 14

        # Paso 1: Objetivo
        p1_col = (255, 215, 0) if self.step_lagr == 1 else TEXT_PRIMARY
        surface.blit(self.fonts["header"].render("1. Función Objetivo a Maximizar R(P):", True, p1_col), (left_rect.x + 16, y))
        y += 22
        surface.blit(self.fonts["mono"].render("  R(P) = (8P₁ - 0.025P₁²) + (12P₂ - 0.04P₂²) + (15P₃ - 0.02P₃²)", True, ROUTER_ACTIVE), (left_rect.x + 16, y))
        y += 20
        surface.blit(self.fonts["small"].render("  Rendimientos marginales decrecientes (términos cuadráticos -½ · β · P²).", True, TEXT_MUTED), (left_rect.x + 16, y))
        y += 26

        # Paso 2: Restricción
        p2_col = (255, 215, 0) if self.step_lagr == 2 else TEXT_PRIMARY
        surface.blit(self.fonts["header"].render("2. Restricción de Presupuesto Global g(P) = 0:", True, p2_col), (left_rect.x + 16, y))
        y += 22
        surface.blit(self.fonts["mono"].render(f"  g(P) = P₁ + P₂ + P₃ - {self.lab_budget:.0f} = 0   (Presupuesto B = {self.lab_budget:.0f} kW)", True, SERVER_WARNING), (left_rect.x + 16, y))
        y += 26

        # Paso 3: Lagrangiano
        p3_col = (255, 215, 0) if self.step_lagr == 3 else TEXT_PRIMARY
        surface.blit(self.fonts["header"].render("3. Función Lagrangiana Ensamblada L(P, λ):", True, p3_col), (left_rect.x + 16, y))
        y += 22
        surface.blit(self.fonts["mono"].render(f"  L(P, λ) = R(P) - λ · (P₁ + P₂ + P₃ - {self.lab_budget:.0f})", True, CHART_LINE_COST), (left_rect.x + 16, y))
        y += 26

        # Paso 4: Condiciones
        p4_col = (255, 215, 0) if self.step_lagr == 4 else TEXT_PRIMARY
        surface.blit(self.fonts["header"].render("4. Condiciones Necesarias de Primer Orden ∇L = 0:", True, p4_col), (left_rect.x + 16, y))
        y += 20
        surface.blit(self.fonts["mono"].render("  • ∂L/∂P₁ = 8 - 0.05P₁ - λ = 0   →   P₁ = (8 - λ) / 0.05", True, SERVER_NORMAL), (left_rect.x + 16, y))
        y += 18
        surface.blit(self.fonts["mono"].render("  • ∂L/∂P₂ = 12 - 0.08P₂ - λ = 0  →   P₂ = (12 - λ) / 0.08", True, SERVER_NORMAL), (left_rect.x + 16, y))
        y += 18
        surface.blit(self.fonts["mono"].render("  • ∂L/∂P₃ = 15 - 0.04P₃ - λ = 0  →   P₃ = (15 - λ) / 0.04", True, SERVER_NORMAL), (left_rect.x + 16, y))
        y += 18
        surface.blit(self.fonts["mono"].render(f"  • ∂L/∂λ  = -(P₁ + P₂ + P₃ - {self.lab_budget:.0f}) = 0", True, (255, 215, 0)), (left_rect.x + 16, y))

        # Panel Derecho: Simulador Interactivo de Presupuesto (Paso 5)
        right_rect = pygame.Rect(rect.x + left_w + 20, rect.y, right_w, rect.height)
        pygame.draw.rect(surface, CARD_BG, right_rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, right_rect, width=1, border_radius=8)

        ry = right_rect.y + 14
        surface.blit(self.fonts["header"].render("5. Simulador de Sensibilidad de Presupuesto (En Vivo)", True, (255, 215, 0) if self.step_lagr == 5 else TEXT_PRIMARY), (right_rect.x + 16, ry))
        ry += 28

        # Controles de Presupuesto
        self.btn_lagr_minus = pygame.Rect(right_rect.x + 16, ry, 110, 32)
        self.btn_lagr_plus = pygame.Rect(right_rect.x + 135, ry, 110, 32)

        pygame.draw.rect(surface, (25, 30, 40), self.btn_lagr_minus, border_radius=6)
        pygame.draw.rect(surface, BORDER_COLOR, self.btn_lagr_minus, width=1, border_radius=6)
        t_minus = self.fonts["regular"].render("[-] 25 kW", True, TEXT_PRIMARY)
        surface.blit(t_minus, t_minus.get_rect(center=self.btn_lagr_minus.center))

        pygame.draw.rect(surface, (25, 30, 40), self.btn_lagr_plus, border_radius=6)
        pygame.draw.rect(surface, BORDER_COLOR, self.btn_lagr_plus, width=1, border_radius=6)
        t_plus = self.fonts["regular"].render("[+] 25 kW", True, TEXT_PRIMARY)
        surface.blit(t_plus, t_plus.get_rect(center=self.btn_lagr_plus.center))

        pres_lbl = self.fonts["header"].render(f"Presupuesto: {self.lab_budget:.0f} kW", True, (255, 215, 0))
        surface.blit(pres_lbl, (right_rect.x + 265, ry + 6))
        ry += 42

        # Multiplicador sombra lambda*
        lambda_val = 6.6957
        if self.lagr_res.lambda_multipliers:
            first_m = self.lagr_res.lambda_multipliers[0]
            if isinstance(first_m, dict):
                lambda_val = float(next(iter(first_m.values())))
            else:
                lambda_val = float(first_m)

        box_lam = pygame.Rect(right_rect.x + 16, ry, right_w - 32, 40)
        pygame.draw.rect(surface, (18, 22, 28), box_lam, border_radius=6)
        pygame.draw.rect(surface, (163, 113, 247), box_lam, width=1, border_radius=6)

        txt_l1 = f"Multiplicador Óptimo λ* = {lambda_val:.4f}"
        txt_l2 = f"(Rendimiento marginal ∂R*/∂B: +{lambda_val:.2f} unidades por cada kW extra)"
        surface.blit(self.fonts["mono_bold"].render(txt_l1, True, (163, 113, 247)), (box_lam.x + 12, box_lam.y + 6))
        surface.blit(self.fonts["small"].render(txt_l2, True, TEXT_MUTED), (box_lam.x + 12, box_lam.y + 22))
        ry += 48

        # Barras de Asignación por Clúster
        if self.lagr_res.best_point:
            clusters_info = [
                ("Clúster WEB (P₁)", self.lagr_res.best_point.get("P1", 26.0), ROUTER_ACTIVE),
                ("Clúster DB (P₂)", self.lagr_res.best_point.get("P2", 66.0), SERVER_NORMAL),
                ("Clúster IA (P₃)", self.lagr_res.best_point.get("P3", 208.0), (163, 113, 247))
            ]

            for name, val, col in clusters_info:
                ratio = val / max(1.0, self.lab_budget)
                surface.blit(self.fonts["var_label"].render(f"{name}: {val:.2f} kW ({ratio * 100:.1f}%)", True, TEXT_PRIMARY), (right_rect.x + 16, ry))
                ry += 20

                bar_bg = pygame.Rect(right_rect.x + 16, ry, right_w - 32, 20)
                pygame.draw.rect(surface, (15, 18, 24), bar_bg, border_radius=4)

                fill_w = int(ratio * (right_w - 32))
                bar_fill = pygame.Rect(right_rect.x + 16, ry, max(4, fill_w), 20)
                pygame.draw.rect(surface, col, bar_fill, border_radius=4)
                ry += 30

            # Rendimiento Total R*
            val_r = self.lagr_res.best_value or 0.0
            surface.blit(self.fonts["mono_bold"].render(f"Rendimiento Global Máximo R* = {val_r:.2f} unidades", True, (255, 235, 120)), (right_rect.x + 16, ry + 6))

    # =========================================================================
    # TAB 5: SOLUCIONADOR LIBRE DE PROBLEMAS NUEVOS (MÁX / MÍN)
    # =========================================================================
    def _sanitize_math_input(self, s: str) -> str:
        """Limpia y estandariza expresiones matemáticas para que SymPy las interprete."""
        s = s.strip()
        s = s.replace("^", "**")
        s = re.sub(r'(\d+(?:\.\d+)?)\s*([a-zA-Z])', r'\1*\2', s)
        s = re.sub(r'\)\s*([a-zA-Z])', r')*\1', s)
        s = re.sub(r'([a-zA-Z])\s*\(', r'\1*(', s)
        return s

    def _load_preset(self, preset: Dict[str, Any]) -> None:
        """Carga un preset académico preconfigurado."""
        self.custom_model_type = preset["type"]
        self.custom_goal = preset["goal"]
        self.input_values["expr"] = preset["expr"]
        self.input_values["vars"] = preset["vars"]
        self.input_values["constraint"] = preset.get("constraint", "")
        self.input_values["init_val"] = preset.get("init_val", "0.0")
        self.solve_custom_problem()
        self.custom_toast_msg = f"Cargado: {preset['title']}"
        self.custom_toast_timer = time.time() + 2.5

    def _parse_constraints(self, raw_str: str) -> List[Dict[str, str]]:
        """
        Analiza restricciones en cualquier formato:
        - Igualdades: x + 2y = 20
        - Desigualdades: x + 2y <= 20, x + y >= 10, x <= 2, x >= 0
        - Restricciones múltiples de no negatividad: x1, x2 >= 0  o  x, y, w >= 0
        - Múltiples restricciones separadas por comas, punto y coma, 'and' o saltos de línea.
        """
        if not raw_str or not raw_str.strip():
            return []

        # 1. Normalizar palabras 'and' (sin tocar 'y' suelta ya que puede ser la variable de decisión)
        s = re.sub(r'\b(?:and)\b', ';', raw_str.strip(), flags=re.IGNORECASE)

        # 2. Dividir inicialmente por ';' y saltos de línea
        chunks = [c.strip() for c in re.split(r'[;\n]+', s) if c.strip()]

        raw_list: List[str] = []
        op_pattern = re.compile(r'<=|>=|<|>|=')

        for chunk in chunks:
            # Limpiar prefijos habituales como 's.a.' o 'sujeto a:'
            chunk = re.sub(r'^(?:s\.?a\.?|sujeto\s+a)\s*:?', '', chunk, flags=re.IGNORECASE).strip()
            if not chunk:
                continue

            ops_found = op_pattern.findall(chunk)

            if len(ops_found) > 1 and ',' in chunk:
                # Varios operadores separados por comas: ej. 'x + y <= 1, x >= 0, y >= 0'
                parts = [p.strip() for p in chunk.split(',') if p.strip()]
                cur = ''
                for p in parts:
                    if op_pattern.search(p):
                        if cur:
                            raw_list.append(cur)
                        cur = p
                    else:
                        cur = f'{cur}, {p}' if cur else p
                if cur:
                    raw_list.append(cur)
            elif len(ops_found) == 1 and ',' in chunk:
                # Un solo operador con comas en lhs: ej. 'x, y >= 0' o 'x1, x2, x3 >= 0'
                m = re.match(r'^(.*?)(<=|>=|<|>|=)(.*)$', chunk)
                if m:
                    lhs, op_m, rhs = m.group(1).strip(), m.group(2), m.group(3).strip()
                    lhs_vars = [v.strip() for v in lhs.split(',') if v.strip()]
                    if len(lhs_vars) > 1 and all(re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', v) for v in lhs_vars):
                        for v in lhs_vars:
                            raw_list.append(f'{v} {op_m} {rhs}')
                    else:
                        raw_list.append(chunk)
                else:
                    raw_list.append(chunk)
            else:
                raw_list.append(chunk)

        parsed = []
        for item in raw_list:
            item = item.strip()
            if not item:
                continue
            if '<=' in item:
                parts = item.split('<=')
                op = '<='
            elif '>=' in item:
                parts = item.split('>=')
                op = '>='
            elif '<' in item:
                parts = item.split('<')
                op = '<='
            elif '>' in item:
                parts = item.split('>')
                op = '>='
            elif '=' in item:
                parts = item.split('=')
                op = '='
            else:
                parts = [item, '0']
                op = '='

            lhs = self._sanitize_math_input(parts[0])
            rhs = self._sanitize_math_input(parts[1]) if len(parts) > 1 and parts[1].strip() else '0'
            diff_expr = f"({lhs}) - ({rhs})"
            parsed.append({
                'raw': item,
                'op': op,
                'type': op,
                'lhs': lhs,
                'rhs': rhs,
                'diff_expr': diff_expr
            })
        return parsed

    def _save_current_problem_to_pdf(self) -> None:
        """Guarda el problema resuelto actual en la lista para el reporte PDF."""
        if not self.custom_solution:
            self.solve_custom_problem()
        if not self.custom_solution:
            return

        type_names = {
            "univariate": "1D Univariable",
            "multivariate": "2D Hessiano",
            "lagrange": "Lagrange",
            "kkt": "KKT Desigualdad",
            "gradient": "Gradiente"
        }
        goal_names = {
            "maximize": "Maximizar",
            "minimize": "Minimizar"
        }

        m_type = type_names.get(self.custom_solution.get("type", self.custom_model_type), self.custom_model_type)
        goal = goal_names.get(self.custom_goal, self.custom_goal)
        clean_expr = to_pretty_math(self.input_values.get("expr", ""))
        clean_constr = to_pretty_math(self.input_values.get("constraint", ""))

        prob_entry = {
            "type": m_type,
            "goal": goal,
            "clean_expr": clean_expr,
            "clean_constr": clean_constr,
            "best_point_str": self.custom_solution.get("best_point_str", "-"),
            "best_value_str": self.custom_solution.get("best_value_str", "-"),
            "verdict": self.custom_solution.get("verdict", "Óptimo Encontrado")
        }

        if not any(p["clean_expr"] == clean_expr and p["clean_constr"] == clean_constr and p["goal"] == goal for p in self.custom_saved_problems):
            self.custom_saved_problems.append(prob_entry)
            self.custom_toast_msg = f"✓ Problema #{len(self.custom_saved_problems)} agregado al PDF!"
        else:
            self.custom_toast_msg = "ℹ Ya está en la lista del PDF"

        self.custom_toast_timer = time.time() + 3.0

    def solve_custom_problem(self) -> None:
        """Resuelve el problema actual analíticamente y numéricamente según el tipo, objetivo y restricciones."""
        self.custom_error_msg = None
        self.custom_solution = None

        raw_expr = self.input_values.get("expr", "").strip()
        if not raw_expr:
            self.custom_error_msg = "Por favor ingresa una función objetivo f."
            return

        clean_expr = self._sanitize_math_input(raw_expr)
        raw_vars = self.input_values.get("vars", "").strip()
        vars_list = [v.strip() for v in raw_vars.split(",") if v.strip()]
        if not vars_list:
            vars_list = ["x"]

        symbols = [sp.Symbol(v, real=True) for v in vars_list]
        sym_dict = {v: s for v, s in zip(vars_list, symbols)}

        raw_constr = self.input_values.get("constraint", "").strip()
        parsed_constrs = self._parse_constraints(raw_constr)

        try:
            # -----------------------------------------------------------------
            # CASO A: CON RESTRICCIONES (Lagrange / KKT / SLSQP)
            # -----------------------------------------------------------------
            if parsed_constrs:
                all_eq = all(c['op'] == '=' for c in parsed_constrs)

                # Si son solo igualdades en >= 2 variables, intentar Lagrange analítico con SymPy
                lagr_analytic_res = None
                if all_eq and len(vars_list) >= 2:
                    try:
                        solver = ConstrainedLagrangeOptimizer(clean_expr, [c['diff_expr'] for c in parsed_constrs], vars_list)
                        lagr_analytic_res = solver.solve(mode=self.custom_goal)
                    except Exception:
                        lagr_analytic_res = None

                if lagr_analytic_res and lagr_analytic_res.best_point:
                    res = lagr_analytic_res
                    pt_coords = ", ".join(f"{k}*={v:.2f}" for k, v in res.best_point.items())
                    best_pt_str = f"({pt_coords})"
                    best_val_str = f"{res.best_value:.4f}" if res.best_value is not None else "-"
                    verdict = f"{'Máximo' if self.custom_goal == 'maximize' else 'Mínimo'} Condicionado (Lagrange)"

                    lambda_val = None
                    if res.lambda_multipliers:
                        first_m = res.lambda_multipliers[0]
                        if isinstance(first_m, dict):
                            lambda_val = float(next(iter(first_m.values())))
                        else:
                            lambda_val = float(first_m)

                    clean_c_display = "; ".join(f"{to_pretty_math(c['lhs'])} = {to_pretty_math(c['rhs'])}" for c in parsed_constrs)

                    self.custom_solution = {
                        "type": "lagrange",
                        "solver_result": res,
                        "vars": vars_list,
                        "best_point": res.best_point,
                        "best_point_str": best_pt_str,
                        "best_value_str": best_val_str,
                        "verdict": verdict,
                        "lambda_val": lambda_val,
                        "clean_constr": clean_c_display,
                        "expr_clean": to_pretty_math(clean_expr),
                        "has_constraints": True
                    }
                    return

                # Si hay desigualdades (<=, >=) o Lagrange simbólico no convergió:
                # Optimizador Numérico Robusto SLSQP (KKT)
                f_sym = sp.sympify(clean_expr, locals=sym_dict)
                f_lambd = sp.lambdify(symbols, f_sym, "numpy")
                sign = -1.0 if self.custom_goal == "maximize" else 1.0

                def obj_fun(x_arr):
                    try:
                        return sign * float(f_lambd(*x_arr))
                    except Exception:
                        return 1e12

                scipy_constrs = []
                for c in parsed_constrs:
                    diff_sym = sp.sympify(c['diff_expr'], locals=sym_dict)
                    diff_lambd = sp.lambdify(symbols, diff_sym, "numpy")
                    if c['op'] == '=':
                        scipy_constrs.append({'type': 'eq', 'fun': lambda x, fn=diff_lambd: float(fn(*x))})
                    elif c['op'] == '<=':
                        scipy_constrs.append({'type': 'ineq', 'fun': lambda x, fn=diff_lambd: -float(fn(*x))})
                    elif c['op'] == '>=':
                        scipy_constrs.append({'type': 'ineq', 'fun': lambda x, fn=diff_lambd: float(fn(*x))})

                best_nlp_res = None
                best_nlp_val = float('-inf') if self.custom_goal == "maximize" else float('inf')

                start_guesses = [
                    [1.0 for _ in vars_list],
                    [0.0 for _ in vars_list],
                    [5.0 for _ in vars_list],
                    [20.0 for _ in vars_list],
                    [50.0 for _ in vars_list],
                    [-1.0 for _ in vars_list]
                ]

                for x0 in start_guesses:
                    try:
                        res_try = opt.minimize(obj_fun, x0, method='SLSQP', constraints=scipy_constrs)
                        if res_try.success:
                            actual_val = float(f_lambd(*res_try.x))
                            if self.custom_goal == "maximize" and actual_val > best_nlp_val:
                                best_nlp_val = actual_val
                                best_nlp_res = res_try
                            elif self.custom_goal == "minimize" and actual_val < best_nlp_val:
                                best_nlp_val = actual_val
                                best_nlp_res = res_try
                    except Exception:
                        pass

                if best_nlp_res is None:
                    best_nlp_res = opt.minimize(obj_fun, [1.0 for _ in vars_list], method='SLSQP', constraints=scipy_constrs)
                    best_nlp_val = float(f_lambd(*best_nlp_res.x))

                best_pt_dict = {v: round(float(best_nlp_res.x[i]), 4) for i, v in enumerate(vars_list)}
                pt_coords = ", ".join(f"{k}*={v:.2f}" for k, v in best_pt_dict.items())
                best_pt_str = f"({pt_coords})" if len(vars_list) > 1 else f"{vars_list[0]}* = {best_pt_dict[vars_list[0]]:.4f}"
                best_val_str = f"{best_nlp_val:.4f}"

                active_list = []
                slack_analysis = []
                for c in parsed_constrs:
                    diff_sym = sp.sympify(c['diff_expr'], locals=sym_dict)
                    diff_val = float(diff_sym.subs([(s, best_pt_dict[v]) for v, s in zip(vars_list, symbols)]).evalf())
                    is_active = abs(diff_val) < 0.05
                    label_str = f"{to_pretty_math(c['lhs'])} {c['op']} {to_pretty_math(c['rhs'])}"
                    if is_active:
                        active_list.append(label_str)
                        slack_analysis.append({"label": label_str, "status": "ACTIVA", "slack": 0.0})
                    else:
                        slack_analysis.append({"label": label_str, "status": "INACTIVA", "slack": abs(diff_val)})

                if active_list:
                    verdict = f"{'Máximo' if self.custom_goal == 'maximize' else 'Mínimo'} Condicionado (KKT: Activa)"
                else:
                    verdict = f"{'Máximo' if self.custom_goal == 'maximize' else 'Mínimo'} en Región Factible"

                clean_c_display = "; ".join(f"{to_pretty_math(c['lhs'])} {c['op']} {to_pretty_math(c['rhs'])}" for c in parsed_constrs)

                self.custom_solution = {
                    "type": "lagrange" if all_eq else "kkt",
                    "solver_result": best_nlp_res,
                    "vars": vars_list,
                    "best_point": KKTPointResult(best_pt_dict, best_nlp_val),
                    "best_point_str": best_pt_str,
                    "best_value_str": best_val_str,
                    "verdict": verdict,
                    "clean_constr": clean_c_display,
                    "active_constrs": active_list,
                    "slack_analysis": slack_analysis,
                    "expr_clean": to_pretty_math(clean_expr),
                    "has_constraints": True
                }
                return

            # -----------------------------------------------------------------
            # CASO B: SIN RESTRICCIONES (Libre: Univariable, Multivariable, Gradiente)
            # -----------------------------------------------------------------
            if self.custom_model_type == "univariate" or len(vars_list) == 1:
                var_name = vars_list[0]
                solver = UnivariateOptimizer(clean_expr, var_name=var_name)
                res = solver.analyze()

                best_pt = None
                target_clf = "Máximo Local" if self.custom_goal == "maximize" else "Mínimo Local"
                matching = [p for p in res.stationary_points if p.classification == target_clf]
                if matching:
                    best_pt = max(matching, key=lambda p: p.f_x) if self.custom_goal == "maximize" else min(matching, key=lambda p: p.f_x)
                elif res.stationary_points:
                    best_pt = max(res.stationary_points, key=lambda p: p.f_x) if self.custom_goal == "maximize" else min(res.stationary_points, key=lambda p: p.f_x)

                if best_pt:
                    best_pt_str = f"{var_name}* = {best_pt.x:.4f}"
                    best_val_str = f"{best_pt.f_x:.4f}"
                    verdict = best_pt.classification
                else:
                    best_pt_str = "Sin puntos estacionarios"
                    best_val_str = "-"
                    verdict = "Monótona (Sin extremos)"

                self.custom_solution = {
                    "type": "univariate",
                    "solver_result": res,
                    "var_name": var_name,
                    "best_point": best_pt,
                    "best_point_str": best_pt_str,
                    "best_value_str": best_val_str,
                    "verdict": verdict,
                    "df_str": to_pretty_math(res.first_derivative_str),
                    "d2f_str": to_pretty_math(res.second_derivative_str),
                    "expr_clean": to_pretty_math(clean_expr),
                    "stationary_points": res.stationary_points,
                    "has_constraints": False
                }

            elif self.custom_model_type == "gradient":
                raw_init = self.input_values.get("init_val", "0.0, 0.0").strip()
                init_parts = [p.strip() for p in raw_init.split(",") if p.strip()]
                init_dict = {v: float(init_parts[i]) if i < len(init_parts) else 0.0 for i, v in enumerate(vars_list)}

                solver = GradientDescentOptimizer(clean_expr, vars_list, learning_rate=0.08, tolerance=1e-3, max_iterations=60)
                res = solver.optimize(init_dict, mode=self.custom_goal)

                pt_coords = ", ".join(f"{k}*={v:.4f}" for k, v in res.optimal_point.items())
                best_pt_str = f"({pt_coords})"
                best_val_str = f"{res.optimal_value:.4f}"
                verdict = f"Convergencia en {res.iterations} iteraciones" if res.converged else "Límite alcanzado"

                self.custom_solution = {
                    "type": "gradient",
                    "solver_result": res,
                    "vars": vars_list,
                    "best_point": res.optimal_point,
                    "best_point_str": best_pt_str,
                    "best_value_str": best_val_str,
                    "verdict": verdict,
                    "iterations": res.iterations,
                    "converged": res.converged,
                    "expr_clean": to_pretty_math(clean_expr),
                    "history": res.history,
                    "has_constraints": False
                }

            else:  # multivariate / default unconstrained
                if len(vars_list) < 2:
                    vars_list = [vars_list[0], "y"]
                    self.input_values["vars"] = ", ".join(vars_list)

                solver = MultivariateHessianOptimizer(clean_expr, vars_list)
                res = solver.analyze()

                target_clf = "Máximo Local" if self.custom_goal == "maximize" else "Mínimo Local"
                best_cp = None
                matching = [p for p in res.critical_points if p.classification == target_clf]
                if matching:
                    best_cp = max(matching, key=lambda p: p.f_val) if self.custom_goal == "maximize" else min(matching, key=lambda p: p.f_val)
                elif res.critical_points:
                    best_cp = max(res.critical_points, key=lambda p: p.f_val) if self.custom_goal == "maximize" else min(res.critical_points, key=lambda p: p.f_val)

                if best_cp:
                    pt_coords = ", ".join(f"{k}*={v:.2f}" for k, v in best_cp.point_dict.items())
                    best_pt_str = f"({pt_coords})"
                    best_val_str = f"{best_cp.f_val:.4f}"
                    verdict = best_cp.classification
                else:
                    best_pt_str = "Sin puntos críticos reales"
                    best_val_str = "-"
                    verdict = "Gradiente no nulo en R²"

                self.custom_solution = {
                    "type": "multivariate",
                    "solver_result": res,
                    "vars": vars_list,
                    "best_point": best_cp,
                    "best_point_str": best_pt_str,
                    "best_value_str": best_val_str,
                    "verdict": verdict,
                    "grad_dict": {k: to_pretty_math(v) for k, v in res.gradient_str.items()},
                    "hess_str": [[to_pretty_math(c) for c in row] for row in res.hessian_matrix_symbolic],
                    "expr_clean": to_pretty_math(clean_expr),
                    "critical_points": res.critical_points,
                    "has_constraints": False
                }

        except Exception as e:
            self.custom_error_msg = f"Error de sintaxis / cálculo: {str(e)}"
            self.custom_solution = None

    def _render_tab_custom(self, surface: pygame.Surface, rect: pygame.Rect, now: float) -> None:
        """Renderiza la interfaz completa de Tab 5: Solucionador de Problemas Personalizados."""
        left_w = 460
        left_rect = pygame.Rect(rect.x, rect.y, left_w, rect.height)
        right_w = rect.width - left_w - 16
        right_rect = pygame.Rect(left_rect.right + 16, rect.y, right_w, rect.height)

        # ---------------------------------------------------------------------
        # PANEL IZQUIERDO: FORMULARIO Y PRESETS
        # ---------------------------------------------------------------------
        pygame.draw.rect(surface, CARD_BG, left_rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, left_rect, width=1, border_radius=8)

        # Encabezado del Formulario
        surface.blit(self.fonts["header"].render("CONFIGURACIÓN DEL PROBLEMA", True, ROUTER_ACTIVE), (left_rect.x + 16, left_rect.y + 14))
        surface.blit(self.fonts["small"].render("Elige el modelo, objetivo (Máx/Mín) y edita la función y restricciones:", True, TEXT_MUTED), (left_rect.x + 16, left_rect.y + 34))

        ly = left_rect.y + 56

        # 1. Selector de Tipo de Modelo (Pills)
        surface.blit(self.fonts["var_label"].render("Modelo Matemático:", True, TEXT_PRIMARY), (left_rect.x + 16, ly))
        ly += 22

        models = [
            ("univariate", "1D Univar"),
            ("multivariate", "2D Hessiano"),
            ("lagrange", "Lagrange"),
            ("gradient", "Gradiente")
        ]
        btn_w = (left_w - 32 - 3 * 8) // 4
        self.type_button_rects = {}
        for idx, (m_key, m_label) in enumerate(models):
            bx = left_rect.x + 16 + idx * (btn_w + 8)
            b_rect = pygame.Rect(bx, ly, btn_w, 28)
            self.type_button_rects[m_key] = b_rect

            is_sel = (self.custom_model_type == m_key)
            bg = (28, 55, 88) if is_sel else (18, 22, 28)
            b_border = (56, 189, 248) if is_sel else BORDER_COLOR
            t_col = (255, 255, 255) if is_sel else TEXT_SECONDARY

            pygame.draw.rect(surface, bg, b_rect, border_radius=6)
            pygame.draw.rect(surface, b_border, b_rect, width=2 if is_sel else 1, border_radius=6)
            txt_surf = self.fonts["small"].render(m_label, True, t_col)
            surface.blit(txt_surf, txt_surf.get_rect(center=b_rect.center))
        ly += 36

        # 2. Selector de Objetivo (Maximizar / Minimizar)
        surface.blit(self.fonts["var_label"].render("Objetivo de Optimización:", True, TEXT_PRIMARY), (left_rect.x + 16, ly))
        ly += 22

        goals = [
            ("maximize", "▲ Maximizar (Máx)", (22, 101, 52), SERVER_NORMAL),
            ("minimize", "▼ Minimizar (Mín)", (127, 29, 29), SERVER_EMERGENCY)
        ]
        goal_w = (left_w - 32 - 10) // 2
        self.goal_button_rects = {}
        for idx, (g_key, g_label, g_bg, g_col) in enumerate(goals):
            gx = left_rect.x + 16 + idx * (goal_w + 10)
            g_rect = pygame.Rect(gx, ly, goal_w, 30)
            self.goal_button_rects[g_key] = g_rect

            is_sel = (self.custom_goal == g_key)
            bg = g_bg if is_sel else (18, 22, 28)
            b_border = g_col if is_sel else BORDER_COLOR
            t_col = (255, 255, 255) if is_sel else TEXT_SECONDARY

            pygame.draw.rect(surface, bg, g_rect, border_radius=6)
            pygame.draw.rect(surface, b_border, g_rect, width=2 if is_sel else 1, border_radius=6)
            txt_g = self.fonts["regular"].render(g_label, True, t_col)
            surface.blit(txt_g, txt_g.get_rect(center=g_rect.center))
        ly += 38

        # 3. Campos de Texto Interactivos
        self.input_rects = {}

        # Campo: Función f
        surface.blit(self.fonts["var_label"].render("Función Objetivo f(x) o f(x, y):", True, TEXT_PRIMARY), (left_rect.x + 16, ly))
        ly += 20
        r_expr = pygame.Rect(left_rect.x + 16, ly, left_w - 32, 30)
        self.input_rects["expr"] = r_expr
        is_focused_expr = (self.focused_field == "expr")
        pygame.draw.rect(surface, (15, 25, 38) if is_focused_expr else (15, 18, 24), r_expr, border_radius=6)
        pygame.draw.rect(surface, (56, 189, 248) if is_focused_expr else BORDER_COLOR, r_expr, width=2 if is_focused_expr else 1, border_radius=6)

        txt_expr = self.input_values.get("expr", "")
        if is_focused_expr and int(now * 2) % 2 == 0:
            txt_expr += "|"
        surface.blit(self.fonts["mono_bold"].render(txt_expr, True, (255, 255, 255)), (r_expr.x + 10, r_expr.y + 6))
        ly += 36

        # Campo: Variables
        surface.blit(self.fonts["var_label"].render("Variables (separadas por coma):", True, TEXT_PRIMARY), (left_rect.x + 16, ly))
        ly += 20
        r_vars = pygame.Rect(left_rect.x + 16, ly, left_w - 32, 28)
        self.input_rects["vars"] = r_vars
        is_focused_vars = (self.focused_field == "vars")
        pygame.draw.rect(surface, (15, 25, 38) if is_focused_vars else (15, 18, 24), r_vars, border_radius=6)
        pygame.draw.rect(surface, (56, 189, 248) if is_focused_vars else BORDER_COLOR, r_vars, width=2 if is_focused_vars else 1, border_radius=6)

        txt_vars = self.input_values.get("vars", "")
        if is_focused_vars and int(now * 2) % 2 == 0:
            txt_vars += "|"
        surface.blit(self.fonts["mono"].render(txt_vars, True, (255, 255, 255)), (r_vars.x + 10, r_vars.y + 5))
        ly += 34

        # Campo: Restricciones (Sujeto a / s.a.) - SIEMPRE VISIBLE PARA CUALQUIER PROBLEMA
        txt_c_now = self.input_values.get("constraint", "")
        parsed_current = self._parse_constraints(txt_c_now)
        n_c = len(parsed_current)
        lbl_c_text = f"Restricciones s.a. [{n_c} detectada{'s' if n_c != 1 else ''}]:" if n_c > 0 else "Restricciones (s.a. / Sujeto a):"
        surface.blit(self.fonts["var_label"].render(lbl_c_text, True, (163, 113, 247)), (left_rect.x + 16, ly))

        # Botón Vaciar Restricciones
        self.btn_clear_constraint = pygame.Rect(left_rect.x + 355, ly - 2, 85, 20)
        pygame.draw.rect(surface, (28, 22, 38), self.btn_clear_constraint, border_radius=4)
        pygame.draw.rect(surface, (163, 113, 247), self.btn_clear_constraint, width=1, border_radius=4)
        t_clear_c = self.fonts["small"].render("✕ Vaciar", True, (220, 190, 255))
        surface.blit(t_clear_c, t_clear_c.get_rect(center=self.btn_clear_constraint.center))

        ly += 20
        r_c = pygame.Rect(left_rect.x + 16, ly, left_w - 32, 30)
        self.input_rects["constraint"] = r_c
        is_focused_c = (self.focused_field == "constraint")
        pygame.draw.rect(surface, (25, 20, 38) if is_focused_c else (15, 18, 24), r_c, border_radius=6)
        pygame.draw.rect(surface, (163, 113, 247) if is_focused_c else BORDER_COLOR, r_c, width=2 if is_focused_c else 1, border_radius=6)

        txt_c = self.input_values.get("constraint", "")
        if not txt_c and not is_focused_c:
            surface.blit(self.fonts["small"].render("(Separa con ';' o ',': ej. 480x1+300x2<=24000; x1,x2>=0)", True, TEXT_MUTED), (r_c.x + 8, r_c.y + 7))
        else:
            txt_disp = txt_c + ("|" if (is_focused_c and int(now * 2) % 2 == 0) else "")
            old_clip = surface.get_clip()
            surface.set_clip(r_c.inflate(-8, -4))
            rendered_t = self.fonts["mono"].render(txt_disp, True, (255, 255, 255))
            if rendered_t.get_width() > r_c.width - 16:
                surface.blit(rendered_t, (r_c.right - 8 - rendered_t.get_width(), r_c.y + 6))
            else:
                surface.blit(rendered_t, (r_c.x + 10, r_c.y + 6))
            surface.set_clip(old_clip)
        ly += 34

        # Visual chips para restricciones detectadas si hay más de 1
        if n_c > 1:
            chip_x = left_rect.x + 16
            for idx_p, p_item in enumerate(parsed_current[:3]):
                chip_txt = f"g{idx_p+1}: {to_pretty_math(p_item['lhs'])} {p_item['op']} {to_pretty_math(p_item['rhs'])}"
                chip_surf = self.fonts["small"].render(chip_txt[:20], True, (215, 195, 255))
                chip_w = chip_surf.get_width() + 10
                chip_rect = pygame.Rect(chip_x, ly, chip_w, 18)
                if chip_rect.right < left_rect.right - 16:
                    pygame.draw.rect(surface, (32, 22, 48), chip_rect, border_radius=4)
                    pygame.draw.rect(surface, (140, 95, 210), chip_rect, width=1, border_radius=4)
                    surface.blit(chip_surf, (chip_x + 5, ly + 2))
                    chip_x += chip_w + 6
            if n_c > 3:
                more_surf = self.fonts["small"].render(f"+{n_c - 3} más", True, (160, 140, 210))
                surface.blit(more_surf, (chip_x, ly + 2))
            ly += 22

        # Campo: Punto Inicial x0 (si es Gradiente)
        if self.custom_model_type == "gradient":
            surface.blit(self.fonts["var_label"].render("Punto Inicial x₀ (separado por coma):", True, (255, 215, 0)), (left_rect.x + 16, ly))
            ly += 20
            r_init = pygame.Rect(left_rect.x + 16, ly, left_w - 32, 28)
            self.input_rects["init_val"] = r_init
            is_focused_init = (self.focused_field == "init_val")
            pygame.draw.rect(surface, (35, 30, 20) if is_focused_init else (15, 18, 24), r_init, border_radius=6)
            pygame.draw.rect(surface, (255, 215, 0) if is_focused_init else BORDER_COLOR, r_init, width=2 if is_focused_init else 1, border_radius=6)

            txt_init = self.input_values.get("init_val", "")
            if is_focused_init and int(now * 2) % 2 == 0:
                txt_init += "|"
            surface.blit(self.fonts["mono"].render(txt_init, True, (255, 255, 255)), (r_init.x + 10, r_init.y + 5))
            ly += 34

        # 4. Presets Académicos Rápidos (1 Clic)
        surface.blit(self.fonts["small"].render("EJEMPLOS ACADÉMICOS RÁPIDOS (1 Clic para cargar):", True, TEXT_MUTED), (left_rect.x + 16, ly + 2))
        ly += 22

        self.preset_button_rects = []
        p_col_w = (left_w - 32 - 8) // 2
        for i, preset in enumerate(PRESETS_CUSTOM):
            c_idx = i % 2
            r_idx = i // 2
            px = left_rect.x + 16 + c_idx * (p_col_w + 8)
            py = ly + r_idx * 30
            p_rect = pygame.Rect(px, py, p_col_w, 26)
            self.preset_button_rects.append((p_rect, preset))

            pygame.draw.rect(surface, (20, 26, 36), p_rect, border_radius=5)
            pygame.draw.rect(surface, (35, 45, 60), p_rect, width=1, border_radius=5)

            lbl = self.fonts["small"].render(preset["title"], True, (160, 205, 250))
            surface.blit(lbl, (p_rect.x + 6, p_rect.y + 5))

        # ---------------------------------------------------------------------
        # PANEL DERECHO: RESULTADOS ANALÍTICOS Y GRÁFICO DINÁMICO
        # ---------------------------------------------------------------------
        pygame.draw.rect(surface, CARD_BG, right_rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, right_rect, width=1, border_radius=8)

        surface.blit(self.fonts["header"].render("RESULTADOS DEL ANÁLISIS MATEMÁTICO", True, ROUTER_ACTIVE), (right_rect.x + 16, right_rect.y + 14))
        surface.blit(self.fonts["small"].render("Deducción con Lagrange / KKT / Hessiano y evaluación exacta de óptimos:", True, TEXT_MUTED), (right_rect.x + 16, right_rect.y + 34))

        ry = right_rect.y + 56

        # Si hay error en la fórmula
        if self.custom_error_msg:
            err_box = pygame.Rect(right_rect.x + 16, ry, right_w - 32, 85)
            pygame.draw.rect(surface, (45, 20, 24), err_box, border_radius=8)
            pygame.draw.rect(surface, SERVER_EMERGENCY, err_box, width=1, border_radius=8)

            surface.blit(self.fonts["var_label"].render("⚠️ Error de Sintaxis o Restricción:", True, (255, 140, 140)), (err_box.x + 12, err_box.y + 10))
            surface.blit(self.fonts["small"].render(str(self.custom_error_msg)[:75], True, TEXT_PRIMARY), (err_box.x + 12, err_box.y + 32))
            surface.blit(self.fonts["small"].render("Sugerencia: Escribe funciones como 'x*y' o '2x + 3y', restricciones como 'x + 2y = 20' o 'x + y <= 10'.", True, TEXT_MUTED), (err_box.x + 12, err_box.y + 55))
            return

        sol = self.custom_solution
        if not sol:
            return

        # 1. Tarjeta Resumen del Óptimo
        opt_box = pygame.Rect(right_rect.x + 16, ry, right_w - 32, 68)
        pygame.draw.rect(surface, (18, 28, 42), opt_box, border_radius=8)
        pygame.draw.rect(surface, (56, 189, 248), opt_box, width=1, border_radius=8)

        # Badge de Objetivo
        is_max = (self.custom_goal == "maximize")
        b_color = SERVER_NORMAL if is_max else SERVER_EMERGENCY
        b_txt = "MAXIMIZACIÓN" if is_max else "MINIMIZACIÓN"
        badge_rect = pygame.Rect(opt_box.x + 12, opt_box.y + 10, 120, 20)
        pygame.draw.rect(surface, b_color, badge_rect, border_radius=4)
        t_b = self.fonts["small"].render(b_txt, True, (255, 255, 255))
        surface.blit(t_b, t_b.get_rect(center=badge_rect.center))

        # Dictamen / Veredicto
        surface.blit(self.fonts["var_label"].render(f"Dictamen: {sol['verdict']}", True, (255, 215, 0)), (opt_box.x + 145, opt_box.y + 11))

        # Punto óptimo y Valor óptimo
        pt_txt = f"Punto Óptimo: {sol['best_point_str']}"
        val_txt = f"Valor Óptimo f* = {sol['best_value_str']}"
        surface.blit(self.fonts["mono_bold"].render(pt_txt, True, (255, 255, 255)), (opt_box.x + 12, opt_box.y + 38))
        surface.blit(self.fonts["mono_bold"].render(val_txt, True, (255, 235, 120)), (opt_box.x + 280, opt_box.y + 38))
        ry += 76

        # 2. Detalles Matemáticos Analíticos
        if sol.get("has_constraints"):
            c_str = sol.get("clean_constr", "")
            if sol["type"] == "lagrange":
                l_val = f"{sol['lambda_val']:.4f}" if sol.get("lambda_val") is not None else "-"
                l_line = f"Multiplicador Sombra λ* = {l_val}  |  Restricción g: {c_str}"
                surface.blit(self.fonts["var_label"].render(l_line, True, (163, 113, 247)), (right_rect.x + 18, ry))
                surface.blit(self.fonts["small"].render("Condición de Tangencia Analítica: ∇f(x*, y*) = λ* ∇g(x*, y*)", True, TEXT_MUTED), (right_rect.x + 18, ry + 20))
                ry += 44
            else:
                slack_info = sol.get("slack_analysis", [])
                n_res = len(slack_info)
                surface.blit(self.fonts["var_label"].render(f"Restricciones s.a. ({n_res} evaluada{'s' if n_res != 1 else ''}):", True, (163, 113, 247)), (right_rect.x + 18, ry))
                ry += 20
                for si in slack_info[:3]:
                    st_col = (110, 231, 183) if si["status"] == "ACTIVA" else (148, 163, 184)
                    st_text = "[ACTIVA (Frontera)]" if si["status"] == "ACTIVA" else f"[Inactiva (Holgura: {si['slack']:.2f})]"
                    line_txt = f"• {si['label']}  ->  {st_text}"
                    surface.blit(self.fonts["small"].render(line_txt, True, st_col), (right_rect.x + 24, ry))
                    ry += 18
                if n_res > 3:
                    surface.blit(self.fonts["small"].render(f"  (+{n_res - 3} restricciones adicionales satisfechas)", True, TEXT_MUTED), (right_rect.x + 24, ry))
                    ry += 18
                ry += 4
        elif sol["type"] == "univariate":
            d1_line = f"1.ª Derivada:  f'({sol['var_name']}) = {sol['df_str']} = 0"
            d2_line = f"2.ª Derivada: f''({sol['var_name']}) = {sol['d2f_str']}"
            surface.blit(self.fonts["var_label"].render(d1_line, True, TEXT_PRIMARY), (right_rect.x + 18, ry))
            surface.blit(self.fonts["var_label"].render(d2_line, True, TEXT_SECONDARY), (right_rect.x + 18, ry + 20))
            ry += 46
        elif sol["type"] == "multivariate":
            grad_strs = [f"∂f/∂{k} = {v}" for k, v in sol["grad_dict"].items()]
            grad_line = "Gradiente ∇f: [" + ",  ".join(grad_strs) + "]"
            surface.blit(self.fonts["small"].render(grad_line, True, TEXT_PRIMARY), (right_rect.x + 18, ry))

            hess_rows = ["[" + ", ".join(row) + "]" for row in sol["hess_str"]]
            hess_line = "Matriz Hessiana H: [" + "; ".join(hess_rows) + "]"
            surface.blit(self.fonts["small"].render(hess_line, True, TEXT_SECONDARY), (right_rect.x + 18, ry + 18))
            ry += 42
        elif sol["type"] == "gradient":
            g_line = f"Iteraciones realizadas: {sol['iterations']} / 60  |  Tasa α = 0.08  |  Convergencia: {sol['converged']}"
            surface.blit(self.fonts["var_label"].render(g_line, True, (255, 215, 0)), (right_rect.x + 18, ry))
            surface.blit(self.fonts["small"].render("Regla de actualización: x^(k+1) = x^(k) ± α · ∇f(x^(k))", True, TEXT_MUTED), (right_rect.x + 18, ry + 20))
            ry += 44

        # 3. Canvas de Visualización Gráfica
        plot_h = right_rect.bottom - ry - 14
        plot_rect = pygame.Rect(right_rect.x + 16, ry, right_w - 32, max(110, plot_h))
        pygame.draw.rect(surface, (14, 18, 24), plot_rect, border_radius=6)
        pygame.draw.rect(surface, BORDER_COLOR, plot_rect, width=1, border_radius=6)

        # Dibujar Gráfica 1D o 2D
        if sol["type"] == "univariate":
            # Gráfica de curva 1D con recta tangente horizontal en el óptimo
            best_p = sol.get("best_point")
            cx_val = best_p.x if best_p else 0.0
            x_min = cx_val - 4.0
            x_max = cx_val + 4.0
            var_name = sol.get("var_name", "x")

            # Evaluar función en 60 puntos
            pts_xy: List[Tuple[float, float]] = []
            solver_inst = sol.get("solver_result")
            for step_i in range(60):
                x_eval = x_min + step_i * (x_max - x_min) / 59.0
                try:
                    y_eval = solver_inst.evaluate(x_eval)
                    if math.isfinite(y_eval):
                        pts_xy.append((x_eval, y_eval))
                except Exception:
                    pass

            if len(pts_xy) >= 2:
                all_y = [p[1] for p in pts_xy]
                min_y = min(all_y)
                max_y = max(all_y)
                if abs(max_y - min_y) < 1e-4:
                    max_y += 1.0
                    min_y -= 1.0
                pad_y = (max_y - min_y) * 0.15
                min_y -= pad_y
                max_y += pad_y

                # Coordenadas en pantalla
                def to_screen(x_val: float, y_val: float) -> Tuple[int, int]:
                    sx = plot_rect.x + 35 + int((x_val - x_min) / (x_max - x_min) * (plot_rect.width - 70))
                    sy = plot_rect.bottom - 22 - int((y_val - min_y) / (max_y - min_y) * (plot_rect.height - 44))
                    return sx, sy

                # Ejes de coordenadas
                if min_y <= 0.0 <= max_y:
                    _, zero_y = to_screen(0.0, 0.0)
                    pygame.draw.line(surface, (40, 50, 65), (plot_rect.x + 30, zero_y), (plot_rect.right - 30, zero_y), 1)
                if x_min <= 0.0 <= x_max:
                    zero_x, _ = to_screen(0.0, 0.0)
                    pygame.draw.line(surface, (40, 50, 65), (zero_x, plot_rect.y + 15), (zero_x, plot_rect.bottom - 15), 1)

                # Curva f(x)
                screen_pts = [to_screen(px, py) for px, py in pts_xy]
                pygame.draw.lines(surface, (56, 189, 248), False, screen_pts, 2)

                # Punto óptimo y tangente horizontal
                if best_p:
                    opt_x, opt_y = to_screen(best_p.x, best_p.f_x)
                    # Recta tangente horizontal
                    pygame.draw.line(surface, (255, 215, 0), (max(plot_rect.x + 10, opt_x - 45), opt_y), (min(plot_rect.right - 10, opt_x + 45), opt_y), 2)
                    # Halo y punto
                    pygame.draw.circle(surface, (255, 215, 0), (opt_x, opt_y), 6)
                    pygame.draw.circle(surface, (255, 255, 255), (opt_x, opt_y), 3)

                    lbl_pt = self.fonts["small"].render(f"({best_p.x:.2f}, {best_p.f_x:.2f})", True, (255, 235, 120))
                    surface.blit(lbl_pt, (opt_x + 8, opt_y - 14))

                # Etiqueta de la curva
                surface.blit(self.fonts["small"].render(f"f({var_name}) = {sol['expr_clean']}", True, (56, 189, 248)), (plot_rect.x + 14, plot_rect.y + 10))

        else:
            # Gráfica 2D con curvas de nivel concéntricas y trayectoria/restricción
            pcx = plot_rect.centerx
            pcy = plot_rect.centery + 10

            # Ejes en cruz
            pygame.draw.line(surface, (32, 42, 56), (plot_rect.x + 20, pcy), (plot_rect.right - 20, pcy), 1)
            pygame.draw.line(surface, (32, 42, 56), (pcx, plot_rect.y + 10), (pcx, plot_rect.bottom - 10), 1)

            # Curvas de nivel concéntricas (elipses)
            radii = [(70, 42), (120, 72), (170, 102)]
            for rx_rad, ry_rad in radii:
                ell_rect = pygame.Rect(pcx - rx_rad, pcy - ry_rad, rx_rad * 2, ry_rad * 2)
                pygame.draw.ellipse(surface, (28, 48, 70), ell_rect, 1)

            # Si es Lagrange: Dibujar recta de restricción tangente
            if sol["type"] == "lagrange":
                pygame.draw.line(surface, (163, 113, 247), (pcx - 140, pcy + 85), (pcx + 140, pcy - 85), 2)
                t_c = self.fonts["small"].render("Restricción g(x, y) = 0", True, (163, 113, 247))
                surface.blit(t_c, (pcx + 50, pcy - 80))

            # Si es Gradiente: Dibujar puntos de trayectoria
            elif sol["type"] == "gradient" and sol.get("history"):
                hist = sol["history"]
                step_points = []
                for idx_h, pt in enumerate(hist[:25]):
                    t_ratio = idx_h / max(1, min(24, len(hist) - 1))
                    sx = int(pcx - 120 + t_ratio * 120)
                    sy = int(pcy + 65 - t_ratio * 65)
                    step_points.append((sx, sy))
                    pygame.draw.circle(surface, (255, 215, 0), (sx, sy), 3)

                if len(step_points) >= 2:
                    pygame.draw.lines(surface, (255, 215, 0), False, step_points, 1)

            # Punto Óptimo Central
            pygame.draw.circle(surface, (255, 215, 0), (pcx, pcy), 7)
            pygame.draw.circle(surface, (255, 255, 255), (pcx, pcy), 3)
            lbl_o = self.fonts["var_label"].render(f"Óptimo: {sol['best_point_str']}", True, (255, 235, 120))
            surface.blit(lbl_o, (pcx + 12, pcy - 12))
            surface.blit(self.fonts["small"].render("Mapa de Curvas de Nivel y Tangencia", True, TEXT_MUTED), (plot_rect.x + 14, plot_rect.y + 10))
