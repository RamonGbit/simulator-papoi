"""
Renderizador del Laboratorio de Programación No Lineal para Pygame.
Permite explorar interactivamente los 4 modelos no lineales:
1. Univariable (Extremos e Inflexión)
2. Multivariable con Matriz Hessiana y Criterio de Sylvester
3. Método del Gradiente (Convergencia iterativa)
4. Multiplicadores de Lagrange (Restricciones analíticas)
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Optional
import pygame

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


class NonLinearLabRenderer:
    """
    Renderizador del modal/pantalla de Laboratorio No Lineal en Pygame.
    """

    def __init__(self, rect: tuple[int, int, int, int], fonts: dict[str, pygame.font.Font]):
        self.rect = pygame.Rect(rect)
        self.fonts = fonts
        self.active_tab: int = 1  # 1: 1D, 2: Hessiano, 3: Gradiente, 4: Lagrange

        # Cargar presets matemáticos pre-calculados
        self.presets = NonLinearLabSuite.get_datacenter_presets()
        self._init_models()

        # Botones de pestañas
        self.tab_rects: List[pygame.Rect] = []
        tab_w = (self.rect.width - 40) // 4
        for i in range(4):
            rx = self.rect.x + 20 + i * tab_w
            ry = self.rect.y + 60
            self.tab_rects.append(pygame.Rect(rx, ry, tab_w - 8, 36))

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
        l_preset = self.presets["cluster_lagrange"]
        self.lagr_solver = ConstrainedLagrangeOptimizer(
            l_preset["objective"],
            l_preset["constraints"],
            l_preset["vars"]
        )
        self.lagr_res = self.lagr_solver.solve(mode="maximize")

    def handle_click(self, pos: tuple[int, int]) -> bool:
        """Gestiona clics en las pestañas."""
        for i, r in enumerate(self.tab_rects):
            if r.collidepoint(pos):
                self.active_tab = i + 1
                return True
        return False

    def handle_key(self, key: int) -> bool:
        """Gestiona atajos de teclado numéricos [1-4]."""
        if key == pygame.K_1:
            self.active_tab = 1
            return True
        elif key == pygame.K_2:
            self.active_tab = 2
            return True
        elif key == pygame.K_3:
            self.active_tab = 3
            return True
        elif key == pygame.K_4:
            self.active_tab = 4
            return True
        return False

    def render(self, surface: pygame.Surface) -> None:
        """Dibuja el panel completo del Laboratorio No Lineal."""
        # Fondo oscuro del modal
        overlay = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        surface.blit(overlay, (0, 0))

        # Tarjeta principal
        pygame.draw.rect(surface, PANEL_BG, self.rect, border_radius=12)
        pygame.draw.rect(surface, BORDER_ACTIVE, self.rect, width=2, border_radius=12)

        # Encabezado
        title_surf = self.fonts["title"].render(
            "LABORATORIO DE PROGRAMACIÓN NO LINEAL — MÉTODOS CUANTITATIVOS (UJAP)",
            True,
            ROUTER_ACTIVE
        )
        surface.blit(title_surf, (self.rect.x + 24, self.rect.y + 18))

        sub_surf = self.fonts["small"].render(
            "Presiona [1-4] o haz clic en las pestañas para alternar | Presiona [N] o [ESC] para volver al Datacenter",
            True,
            TEXT_SECONDARY
        )
        surface.blit(sub_surf, (self.rect.x + 24, self.rect.y + 42))

        # Dibujar pestañas superiores
        self._render_tabs(surface)

        # Dibujar contenido según la pestaña activa
        content_rect = pygame.Rect(
            self.rect.x + 20,
            self.rect.y + 108,
            self.rect.width - 40,
            self.rect.height - 128
        )

        if self.active_tab == 1:
            self._render_tab_1d(surface, content_rect)
        elif self.active_tab == 2:
            self._render_tab_hessian(surface, content_rect)
        elif self.active_tab == 3:
            self._render_tab_gradient(surface, content_rect)
        elif self.active_tab == 4:
            self._render_tab_lagrange(surface, content_rect)

    def _render_tabs(self, surface: pygame.Surface) -> None:
        """Renderiza las 4 pestañas superiores."""
        tab_names = [
            "[1] Modelo 1D: Inflexión y Extremos",
            "[2] Multivariable: Matriz Hessiana",
            "[3] Método del Gradiente Descendente",
            "[4] Multiplicadores de Lagrange"
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

    # -------------------------------------------------------------------------
    # TAB 1: MODELO 1D (ANÁLISIS TÉRMICO E INFLEXIÓN)
    # -------------------------------------------------------------------------
    def _render_tab_1d(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        left_w = int(rect.width * 0.48)
        right_w = rect.width - left_w - 20

        # Panel Izquierdo: Fórmulas y Derivadas
        left_rect = pygame.Rect(rect.x, rect.y, left_w, rect.height)
        pygame.draw.rect(surface, CARD_BG, left_rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, left_rect, width=1, border_radius=8)

        y = left_rect.y + 16
        t_title = self.fonts["header"].render("1. Análisis Diferencial Simbólico (SymPy)", True, TEXT_PRIMARY)
        surface.blit(t_title, (left_rect.x + 16, y))
        y += 28

        lines = [
            ("Función Térmica T(L):", self.uni_res.expression_str, ROUTER_ACTIVE),
            ("Primera Derivada T'(L):", self.uni_res.first_derivative_str, SERVER_NORMAL),
            ("Segunda Derivada T''(L):", self.uni_res.second_derivative_str, CHART_LINE_COST),
        ]
        for lbl, val, col in lines:
            l_surf = self.fonts["var_label"].render(lbl, True, TEXT_SECONDARY)
            surface.blit(l_surf, (left_rect.x + 16, y))
            y += 18
            v_surf = self.fonts["mono_bold"].render(f"  {val}", True, col)
            surface.blit(v_surf, (left_rect.x + 16, y))
            y += 24

        y += 8
        crit_title = self.fonts["header"].render("Puntos Críticos e Inflexión:", True, TEXT_PRIMARY)
        surface.blit(crit_title, (left_rect.x + 16, y))
        y += 26

        if self.uni_res.inflection_points:
            ip = self.uni_res.inflection_points[0]
            txt_ip = f"• Punto de Inflexión Crítico: L = {ip.x*100:.1f}%, T = {ip.f_x:.2f}°C"
            s1 = self.fonts["regular"].render(txt_ip, True, SERVER_EMERGENCY)
            surface.blit(s1, (left_rect.x + 16, y))
            y += 20
            s2 = self.fonts["small"].render("  -> A partir de este punto, d²T/dL² > 0 (Aceleración superlineal).", True, TEXT_MUTED)
            surface.blit(s2, (left_rect.x + 16, y))
            y += 28

        txt_protocol = "• Protocolo: Activación de enfriamiento de emergencia al superar el 60% de carga."
        s3 = self.fonts["small"].render(txt_protocol, True, SERVER_WARNING)
        surface.blit(s3, (left_rect.x + 16, y))

        # Panel Derecho: Gráfica de la Curva No Lineal
        right_rect = pygame.Rect(rect.x + left_w + 20, rect.y, right_w, rect.height)
        pygame.draw.rect(surface, CARD_BG, right_rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, right_rect, width=1, border_radius=8)

        g_title = self.fonts["header"].render("Curva No Lineal de Temperatura T(L)", True, TEXT_PRIMARY)
        surface.blit(g_title, (right_rect.x + 16, right_rect.y + 16))

        # Dibujar gráfico de la curva T(L) para L in [0, 1]
        plot_rect = pygame.Rect(right_rect.x + 40, right_rect.y + 50, right_rect.width - 60, right_rect.height - 80)
        pygame.draw.rect(surface, (18, 22, 28), plot_rect)
        pygame.draw.rect(surface, CHART_GRID, plot_rect, width=1)

        # Muestrear puntos
        steps = 50
        points = []
        for i in range(steps + 1):
            L = i / steps
            T = self.uni_solver.evaluate(L)
            px = plot_rect.x + (L * plot_rect.width)
            py = plot_rect.bottom - ((T - 20) / (70 - 20)) * plot_rect.height
            points.append((px, py))

        if len(points) > 1:
            pygame.draw.lines(surface, ROUTER_ACTIVE, False, points, 3)

        # Marcar punto de inflexión L = 0.60
        inf_x = plot_rect.x + (0.60 * plot_rect.width)
        inf_t = self.uni_solver.evaluate(0.60)
        inf_y = plot_rect.bottom - ((inf_t - 20) / (70 - 20)) * plot_rect.height

        pygame.draw.circle(surface, SERVER_EMERGENCY, (int(inf_x), int(inf_y)), 7)
        pygame.draw.circle(surface, (255, 255, 255), (int(inf_x), int(inf_y)), 3)

        lbl_inf = self.fonts["small"].render(f"Inflexión (60%, {inf_t:.1f}°C)", True, SERVER_EMERGENCY)
        surface.blit(lbl_inf, (inf_x - 50, inf_y - 24))

    # -------------------------------------------------------------------------
    # TAB 2: MULTIVARIABLE CON MATRIZ HESSIANA
    # -------------------------------------------------------------------------
    def _render_tab_hessian(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        left_w = int(rect.width * 0.50)
        right_w = rect.width - left_w - 20

        # Panel Izquierdo: Modelo y Gradiente
        left_rect = pygame.Rect(rect.x, rect.y, left_w, rect.height)
        pygame.draw.rect(surface, CARD_BG, left_rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, left_rect, width=1, border_radius=8)

        y = left_rect.y + 16
        t_title = self.fonts["header"].render("2. Optimización Multivariable Sin Restricciones", True, TEXT_PRIMARY)
        surface.blit(t_title, (left_rect.x + 16, y))
        y += 28

        f_lbl = self.fonts["var_label"].render("Función de Disipación Conjunta E(f, v):", True, TEXT_SECONDARY)
        surface.blit(f_lbl, (left_rect.x + 16, y))
        y += 18
        f_val = self.fonts["mono_bold"].render(f"  {self.multi_res.expression_str}", True, ROUTER_ACTIVE)
        surface.blit(f_val, (left_rect.x + 16, y))
        y += 26

        g_lbl = self.fonts["var_label"].render("Vector Gradiente ∇E = [∂E/∂f, ∂E/∂v]:", True, TEXT_SECONDARY)
        surface.blit(g_lbl, (left_rect.x + 16, y))
        y += 18
        for k, expr in self.multi_res.gradient_str.items():
            g_item = self.fonts["mono"].render(f"  • ∂E/∂{k} = {expr}", True, SERVER_NORMAL)
            surface.blit(g_item, (left_rect.x + 16, y))
            y += 20

        y += 10
        cond_lbl = self.fonts["small"].render("Condición de Primer Orden: ∇E = [0, 0] para puntos críticos.", True, TEXT_MUTED)
        surface.blit(cond_lbl, (left_rect.x + 16, y))
        y += 26

        if self.multi_res.critical_points:
            cp = self.multi_res.critical_points[0]
            cp_title = self.fonts["header"].render(f"Punto Crítico Hallado: {cp.point_dict}", True, TEXT_PRIMARY)
            surface.blit(cp_title, (left_rect.x + 16, y))
            y += 22
            val_txt = self.fonts["regular"].render(f"Valor Óptimo E(f*, v*) = {cp.f_val} unidades de disipación", True, SERVER_NORMAL)
            surface.blit(val_txt, (left_rect.x + 16, y))

        # Panel Derecho: Matriz Hessiana y Criterio de Sylvester
        right_rect = pygame.Rect(rect.x + left_w + 20, rect.y, right_w, rect.height)
        pygame.draw.rect(surface, CARD_BG, right_rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, right_rect, width=1, border_radius=8)

        ry = right_rect.y + 16
        h_title = self.fonts["header"].render("Matriz Hessiana y Criterio de Sylvester", True, TEXT_PRIMARY)
        surface.blit(h_title, (right_rect.x + 16, ry))
        ry += 30

        if self.multi_res.critical_points:
            cp = self.multi_res.critical_points[0]
            H = cp.hessian_matrix_evaluated
            
            # Dibujar Matriz Hessiana visualmente
            matrix_txt = [
                f"       [ {H[0][0]:>5.1f}   {H[0][1]:>5.1f} ]",
                f"  H  = [                     ]",
                f"       [ {H[1][0]:>5.1f}   {H[1][1]:>5.1f} ]"
            ]
            for m_line in matrix_txt:
                s_m = self.fonts["mono_bold"].render(m_line, True, CHART_LINE_COST)
                surface.blit(s_m, (right_rect.x + 24, ry))
                ry += 20

            ry += 16
            minors_title = self.fonts["var_label"].render("Menores Principales Líderes (Sylvester):", True, TEXT_SECONDARY)
            surface.blit(minors_title, (right_rect.x + 16, ry))
            ry += 20

            m1_txt = f"  • Δ1 = |H11| = {cp.principal_minors[0]:.2f} > 0"
            m2_txt = f"  • Δ2 = det(H) = {cp.principal_minors[1]:.2f} > 0"
            surface.blit(self.fonts["mono"].render(m1_txt, True, SERVER_NORMAL), (right_rect.x + 20, ry))
            ry += 20
            surface.blit(self.fonts["mono"].render(m2_txt, True, SERVER_NORMAL), (right_rect.x + 20, ry))
            ry += 26

            clf_box = pygame.Rect(right_rect.x + 16, ry, right_w - 32, 44)
            pygame.draw.rect(surface, (20, 45, 30), clf_box, border_radius=6)
            pygame.draw.rect(surface, SERVER_NORMAL, clf_box, width=1, border_radius=6)

            clf_txt = self.fonts["header"].render(
                f"Clasificación: {cp.classification.upper()}",
                True,
                SERVER_NORMAL
            )
            surface.blit(clf_txt, clf_txt.get_rect(center=clf_box.center))

    # -------------------------------------------------------------------------
    # TAB 3: MÉTODO DEL GRADIENTE DESCENDENTE
    # -------------------------------------------------------------------------
    def _render_tab_gradient(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        left_w = int(rect.width * 0.45)
        right_w = rect.width - left_w - 20

        # Panel Izquierdo: Parámetros e Iteraciones
        left_rect = pygame.Rect(rect.x, rect.y, left_w, rect.height)
        pygame.draw.rect(surface, CARD_BG, left_rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, left_rect, width=1, border_radius=8)

        y = left_rect.y + 16
        t_title = self.fonts["header"].render("3. Método del Gradiente Descendente", True, TEXT_PRIMARY)
        surface.blit(t_title, (left_rect.x + 16, y))
        y += 28

        info_lines = [
            f"• Punto Inicial P0: {self.grad_res.initial_point['P']} W",
            f"• Tasa de Aprendizaje α: 0.08",
            f"• Iteraciones Realizadas: {self.grad_res.iterations}",
            f"• Convergencia Exitosa: {self.grad_res.converged}",
            f"• Potencia Óptima P*: {self.grad_res.optimal_point['P']} W",
            f"• Costo Mínimo J(P*): {self.grad_res.optimal_val:.4f}"
        ]
        for line in info_lines:
            surface.blit(self.fonts["regular"].render(line, True, TEXT_SECONDARY), (left_rect.x + 16, y))
            y += 22

        y += 10
        hist_title = self.fonts["header"].render("Tabla de Convergencia (Primeros y Últimos pasos):", True, TEXT_PRIMARY)
        surface.blit(hist_title, (left_rect.x + 16, y))
        y += 24

        table_header = f"{'Paso':<6} {'Potencia (W)':<14} {'Costo J':<12} {'||∇J||':<10}"
        surface.blit(self.fonts["mono_bold"].render(table_header, True, ROUTER_ACTIVE), (left_rect.x + 16, y))
        y += 18

        sample_steps = self.grad_res.history[:4] + self.grad_res.history[-3:]
        for it in sample_steps:
            row_txt = f"{it.step:<6} {it.coords['P']:<14.2f} {it.f_val:<12.2f} {it.grad_norm:<10.4f}"
            surface.blit(self.fonts["mono"].render(row_txt, True, TEXT_MUTED), (left_rect.x + 16, y))
            y += 18

        # Panel Derecho: Curva de Convergencia
        right_rect = pygame.Rect(rect.x + left_w + 20, rect.y, right_w, rect.height)
        pygame.draw.rect(surface, CARD_BG, right_rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, right_rect, width=1, border_radius=8)

        g_title = self.fonts["header"].render("Trayectoria de Costo J(P) vs Iteración", True, TEXT_PRIMARY)
        surface.blit(g_title, (right_rect.x + 16, right_rect.y + 16))

        plot_rect = pygame.Rect(right_rect.x + 40, right_rect.y + 50, right_rect.width - 60, right_rect.height - 80)
        pygame.draw.rect(surface, (18, 22, 28), plot_rect)
        pygame.draw.rect(surface, CHART_GRID, plot_rect, width=1)

        # Graficar descenso de costo
        hist = self.grad_res.history
        if len(hist) > 1:
            max_c = max(it.f_val for it in hist)
            min_c = min(it.f_val for it in hist)
            c_range = max(1.0, max_c - min_c)

            pts = []
            for i, it in enumerate(hist):
                px = plot_rect.x + (i / (len(hist) - 1)) * plot_rect.width
                py = plot_rect.bottom - ((it.f_val - min_c) / c_range) * (plot_rect.height - 20) - 10
                pts.append((px, py))

            pygame.draw.lines(surface, CHART_LINE_GRADIENT, False, pts, 3)

            # Punto final
            pygame.draw.circle(surface, SERVER_NORMAL, (int(pts[-1][0]), int(pts[-1][1])), 6)

    # -------------------------------------------------------------------------
    # TAB 4: MULTIPLICADORES DE LAGRANGE
    # -------------------------------------------------------------------------
    def _render_tab_lagrange(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        left_w = int(rect.width * 0.50)
        right_w = rect.width - left_w - 20

        # Panel Izquierdo: Formulación Simbólica
        left_rect = pygame.Rect(rect.x, rect.y, left_w, rect.height)
        pygame.draw.rect(surface, CARD_BG, left_rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, left_rect, width=1, border_radius=8)

        y = left_rect.y + 16
        t_title = self.fonts["header"].render("4. Multiplicadores de Lagrange (Con Restricciones)", True, TEXT_PRIMARY)
        surface.blit(t_title, (left_rect.x + 16, y))
        y += 28

        obj_lbl = self.fonts["var_label"].render("Función Objetivo a Maximizar R(P):", True, TEXT_SECONDARY)
        surface.blit(obj_lbl, (left_rect.x + 16, y))
        y += 18
        obj_txt = self.fonts["mono"].render("  R = (8P1 - 0.025P1²) + (12P2 - 0.04P2²) + (15P3 - 0.02P3²)", True, ROUTER_ACTIVE)
        surface.blit(obj_txt, (left_rect.x + 16, y))
        y += 26

        res_lbl = self.fonts["var_label"].render("Restricción de Igualdad g(P) = 0:", True, TEXT_SECONDARY)
        surface.blit(res_lbl, (left_rect.x + 16, y))
        y += 18
        res_txt = self.fonts["mono"].render("  P1 + P2 + P3 - 300 = 0  (Presupuesto = 300 kW)", True, SERVER_WARNING)
        surface.blit(res_txt, (left_rect.x + 16, y))
        y += 26

        lagr_lbl = self.fonts["var_label"].render("Función Lagrangiana Ensamblada:", True, TEXT_SECONDARY)
        surface.blit(lagr_lbl, (left_rect.x + 16, y))
        y += 18
        lagr_txt = self.fonts["mono"].render("  L(P, λ) = R(P) - λ · (P1 + P2 + P3 - 300)", True, CHART_LINE_COST)
        surface.blit(lagr_txt, (left_rect.x + 16, y))
        y += 30

        if self.lagr_res.best_point:
            opt_title = self.fonts["header"].render(f"Multiplicador Óptimo λ* = 6.6957", True, SERVER_NORMAL)
            surface.blit(opt_title, (left_rect.x + 16, y))
            y += 22
            rend_txt = self.fonts["regular"].render(f"Rendimiento Máximo R* = {self.lagr_res.best_value} unidades", True, TEXT_PRIMARY)
            surface.blit(rend_txt, (left_rect.x + 16, y))

        # Panel Derecho: Gráfico de Asignación por Clúster
        right_rect = pygame.Rect(rect.x + left_w + 20, rect.y, right_w, rect.height)
        pygame.draw.rect(surface, CARD_BG, right_rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, right_rect, width=1, border_radius=8)

        ry = right_rect.y + 16
        b_title = self.fonts["header"].render("Distribución Óptima de Potencia por Clúster", True, TEXT_PRIMARY)
        surface.blit(b_title, (right_rect.x + 16, ry))
        ry += 40

        if self.lagr_res.best_point:
            clusters_info = [
                ("Clúster WEB (P1)", self.lagr_res.best_point.get("P1", 26.09), ROUTER_ACTIVE),
                ("Clúster DB (P2)", self.lagr_res.best_point.get("P2", 66.30), SERVER_NORMAL),
                ("Clúster AI (P3)", self.lagr_res.best_point.get("P3", 207.61), (163, 113, 247))
            ]

            max_p = 300.0
            for name, val, col in clusters_info:
                surface.blit(self.fonts["var_label"].render(f"{name}: {val:.2f} kW ({val/max_p*100:.1f}%)", True, TEXT_PRIMARY), (right_rect.x + 20, ry))
                ry += 22

                bar_bg = pygame.Rect(right_rect.x + 20, ry, right_w - 40, 22)
                pygame.draw.rect(surface, (18, 22, 28), bar_bg, border_radius=4)

                fill_w = int((val / max_p) * (right_w - 40))
                bar_fill = pygame.Rect(right_rect.x + 20, ry, fill_w, 22)
                pygame.draw.rect(surface, col, bar_fill, border_radius=4)
                ry += 36
