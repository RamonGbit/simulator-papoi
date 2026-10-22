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
    Renderizador interactivo del modal de Laboratorio No Lineal con navegación paso a paso y animaciones.
    Utiliza iconos vectoriales dibujados con primitivas geométricas para garantizar compatibilidad universal.
    """

    def __init__(self, rect: tuple[int, int, int, int], fonts: dict[str, pygame.font.Font]):
        self.rect = pygame.Rect(rect)
        self.fonts = fonts
        self.active_tab: int = 1  # 1: 1D, 2: Hessiano, 3: Gradiente, 4: Lagrange

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

        # Botones de pestañas superiores
        self.tab_rects: List[pygame.Rect] = []
        tab_w = (self.rect.width - 40) // 4
        for i in range(4):
            rx = self.rect.x + 20 + i * tab_w
            ry = self.rect.y + 60
            self.tab_rects.append(pygame.Rect(rx, ry, tab_w - 8, 34))

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

        return False

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
            "Navegacion paso a paso: Usa [<< Anterior] / [Siguiente >>] o teclas [A]/[D] | Presiona [N] o [ESC] para volver al Datacenter",
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

    def _get_current_step_info(self) -> tuple[int, int]:
        """Retorna (paso_actual, total_pasos) según la pestaña activa."""
        if self.active_tab == 1:
            return self.step_1d, 4
        elif self.active_tab == 2:
            return self.step_hessian, 5
        elif self.active_tab == 3:
            return self.step_grad, len(self.grad_res.history) - 1
        else:
            return self.step_lagr, 5

    def _render_tabs(self, surface: pygame.Surface) -> None:
        """Renderiza las 4 pestañas superiores."""
        tab_names = [
            "[1] Modelo 1D: Inflexión Térmica",
            "[2] Multivariable: Matriz Hessiana",
            "[3] Descenso de Gradiente",
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

    def _render_step_controls(self, surface: pygame.Surface) -> None:
        """Renderiza la barra de control interactivo dibujando iconos vectoriales limpios."""
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

        # --- Título descriptivo del paso actual (Sin símbolos exóticos) ---
        step_titles_1d = [
            "Paso 1/4: Funcion de Temperatura T(L) y Comportamiento Base",
            "Paso 2/4: Primera Derivada T'(L) y Tasa de Variacion Positiva (Creciente)",
            "Paso 3/4: Segunda Derivada d^2T/dL^2 y Determinacion del Punto de Inflexion (60%)",
            "Paso 4/4: Delimitacion de Regimenes y Protocolo de Enfriamiento de Emergencia"
        ]
        step_titles_hess = [
            "Paso 1/5: Funcion Objetivo de Disipacion Energetica E(f, v)",
            "Paso 2/5: Vector Gradiente grad(E) = [0, 0] (Condicion de 1er Orden)",
            "Paso 3/5: Resolucion Analitica y Localizacion del Punto Critico (f*, v*)",
            "Paso 4/5: Construccion de la Matriz Hessiana Simbolica H",
            "Paso 5/5: Criterio de Sylvester (Menores Principales) y Veredicto de Minimo"
        ]
        step_titles_lagr = [
            "Paso 1/5: Funcion de Rendimiento Global R(P) con Rendimientos Decrecientes",
            "Paso 2/5: Restriccion de Presupuesto Electrico Fijo g(P) = 0",
            "Paso 3/5: Ensamblaje de la Funcion Lagrangiana L(P, lambda)",
            "Paso 4/5: Condiciones Necesarias de Primer Orden grad(L) = 0",
            "Paso 5/5: Simulador de Sensibilidad: Variacion de Presupuesto y Multiplicador lambda*"
        ]

        if self.active_tab == 1:
            lbl_desc = step_titles_1d[curr - 1]
        elif self.active_tab == 2:
            lbl_desc = step_titles_hess[curr - 1]
        elif self.active_tab == 3:
            lbl_desc = f"Paso {curr}/{total}: Iteracion k={curr} del Gradiente Descendente | P={self.grad_res.history[curr].coords['P']:.2f} W"
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
        surface.blit(self.fonts["header"].render("1. Funcion de Temperatura T(L):", True, p1_col), (left_rect.x + 16, y))
        y += 22
        surface.blit(self.fonts["mono_bold"].render(f"  {self.uni_res.expression_str}", True, ROUTER_ACTIVE), (left_rect.x + 16, y))
        y += 20
        surface.blit(self.fonts["small"].render("  Donde L en [0, 1] es el factor de carga de CPU/RAM.", True, TEXT_MUTED), (left_rect.x + 16, y))
        y += 28

        # Paso 2: Derivada 1
        p2_col = (255, 215, 0) if self.step_1d == 2 else TEXT_PRIMARY
        surface.blit(self.fonts["header"].render("2. Primera Derivada T'(L) (Crecimiento):", True, p2_col), (left_rect.x + 16, y))
        y += 22
        surface.blit(self.fonts["mono_bold"].render(f"  {self.uni_res.first_derivative_str}", True, SERVER_NORMAL), (left_rect.x + 16, y))
        y += 20
        surface.blit(self.fonts["small"].render("  T'(L) > 0 para todo L en [0, 1] ==> Calentamiento monotono positivo.", True, TEXT_MUTED), (left_rect.x + 16, y))
        y += 28

        # Paso 3: Derivada 2 e Inflexión
        p3_col = (255, 215, 0) if self.step_1d == 3 else TEXT_PRIMARY
        surface.blit(self.fonts["header"].render("3. Segunda Derivada d^2T/dL^2 y Punto de Inflexion:", True, p3_col), (left_rect.x + 16, y))
        y += 22
        surface.blit(self.fonts["mono_bold"].render(f"  {self.uni_res.second_derivative_str} = 0", True, CHART_LINE_COST), (left_rect.x + 16, y))
        y += 20
        surface.blit(self.fonts["mono_bold"].render("  ==> 180L = 108  ==>  L_inflexion = 0.60 (60% de Carga)", True, SERVER_EMERGENCY), (left_rect.x + 16, y))
        y += 20
        surface.blit(self.fonts["small"].render("  Temperatura en la inflexion: T(0.60) = 48.04 C", True, TEXT_SECONDARY), (left_rect.x + 16, y))
        y += 28

        # Paso 4: Dictamen y Protocolo
        p4_col = (255, 215, 0) if self.step_1d == 4 else TEXT_PRIMARY
        surface.blit(self.fonts["header"].render("4. Regimen Superlineal y Alerta Operativa:", True, p4_col), (left_rect.x + 16, y))
        y += 22
        surface.blit(self.fonts["regular"].render("  * Si L < 60%: d^2T/dL^2 < 0 (Convexa hacia abajo, regimen seguro).", True, SERVER_NORMAL), (left_rect.x + 16, y))
        y += 18
        surface.blit(self.fonts["regular"].render("  * Si L >= 60%: d^2T/dL^2 > 0 (Aceleracion superlineal descontrolada).", True, SERVER_EMERGENCY), (left_rect.x + 16, y))
        y += 22
        box_prot = pygame.Rect(left_rect.x + 16, y, left_w - 32, 34)
        pygame.draw.rect(surface, (50, 15, 18), box_prot, border_radius=6)
        pygame.draw.rect(surface, SERVER_EMERGENCY, box_prot, width=1, border_radius=6)
        t_em = self.fonts["mono_bold"].render("ALERTA: Protocolo de Enfriamiento de Emergencia activo en L >= 60%", True, (255, 180, 180))
        surface.blit(t_em, t_em.get_rect(center=box_prot.center))

        # Panel Derecho: Gráfica Dinámica
        right_rect = pygame.Rect(rect.x + left_w + 20, rect.y, right_w, rect.height)
        pygame.draw.rect(surface, CARD_BG, right_rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, right_rect, width=1, border_radius=8)

        g_title = self.fonts["header"].render("Visualizacion Dinamica de la Curva Termica T(L)", True, TEXT_PRIMARY)
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
            surface.blit(self.fonts["small"].render("ZONA CRITICA SUPERLINEAL (>= 60%)", True, SERVER_EMERGENCY), (split_x + 10, plot_rect.y + 10))

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

            lbl_tang = self.fonts["small"].render(f"Tangente T'({tang_L:.2f}) = +{slope:.1f} > 0", True, SERVER_NORMAL)
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

            surface.blit(self.fonts["mono_bold"].render("PUNTO DE INFLEXION (60%, 48.0 C)", True, SERVER_EMERGENCY), (inf_px - 85, inf_py - 24))

        # Etiquetas de Ejes
        surface.blit(self.fonts["small"].render("70 C", True, TEXT_MUTED), (right_rect.x + 12, plot_rect.y))
        surface.blit(self.fonts["small"].render("48 C", True, SERVER_EMERGENCY), (right_rect.x + 12, plot_rect.bottom - int(((48.04 - 20.0) / (70.0 - 20.0)) * plot_rect.height) - 6))
        surface.blit(self.fonts["small"].render("20 C", True, TEXT_MUTED), (right_rect.x + 12, plot_rect.bottom - 10))

        surface.blit(self.fonts["small"].render("0% Carga", True, TEXT_MUTED), (plot_rect.x, plot_rect.bottom + 4))
        surface.blit(self.fonts["small"].render("60% Inflexion", True, SERVER_EMERGENCY), (plot_rect.x + int(0.60 * plot_rect.width) - 25, plot_rect.bottom + 4))
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
        surface.blit(self.fonts["header"].render("1. Funcion de Disipacion de CPU E(f, v):", True, p1_col), (left_rect.x + 16, y))
        y += 22
        surface.blit(self.fonts["mono_bold"].render(f"  {self.multi_res.expression_str}", True, ROUTER_ACTIVE), (left_rect.x + 16, y))
        y += 20
        surface.blit(self.fonts["small"].render("  f = Frecuencia de reloj (GHz), v = Voltaje de nucleo (V)", True, TEXT_MUTED), (left_rect.x + 16, y))
        y += 28

        # Paso 2: Gradiente
        p2_col = (255, 215, 0) if self.step_hessian == 2 else TEXT_PRIMARY
        surface.blit(self.fonts["header"].render("2. Vector Gradiente grad(E) = [0, 0] (1er Orden):", True, p2_col), (left_rect.x + 16, y))
        y += 22
        surface.blit(self.fonts["mono_bold"].render("  * dE/df = 8f - 2v - 24 = 0", True, SERVER_NORMAL), (left_rect.x + 16, y))
        y += 18
        surface.blit(self.fonts["mono_bold"].render("  * dE/dv = 12v - 2f - 30 = 0", True, SERVER_NORMAL), (left_rect.x + 16, y))
        y += 28

        # Paso 3: Punto Crítico
        p3_col = (255, 215, 0) if self.step_hessian == 3 else TEXT_PRIMARY
        surface.blit(self.fonts["header"].render("3. Resolucion Analitica y Punto Estacionario:", True, p3_col), (left_rect.x + 16, y))
        y += 22
        surface.blit(self.fonts["mono_bold"].render("  ==> f* = 3.80 GHz  |  v* = 3.13 V", True, (255, 215, 0)), (left_rect.x + 16, y))
        y += 18
        surface.blit(self.fonts["regular"].render("  Disipacion en el optimo: E(3.80, 3.13) = 7.39 unidades", True, TEXT_SECONDARY), (left_rect.x + 16, y))

        # Panel Derecho: Matriz Hessiana y Sylvester (Pasos 4 y 5)
        right_rect = pygame.Rect(rect.x + left_w + 20, rect.y, right_w, rect.height)
        pygame.draw.rect(surface, CARD_BG, right_rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, right_rect, width=1, border_radius=8)

        ry = right_rect.y + 14

        # Paso 4: Matriz Hessiana
        p4_col = (255, 215, 0) if self.step_hessian == 4 else TEXT_PRIMARY
        surface.blit(self.fonts["header"].render("4. Matriz Hessiana H (2do Orden):", True, p4_col), (right_rect.x + 16, ry))
        ry += 24

        box_h = pygame.Rect(right_rect.x + 24, ry, right_w - 48, 64)
        pygame.draw.rect(surface, (18, 22, 28), box_h, border_radius=6)
        pygame.draw.rect(surface, (255, 166, 87) if self.step_hessian == 4 else BORDER_COLOR, box_h, width=1, border_radius=6)

        surface.blit(self.fonts["mono_bold"].render("       [  d^2E/df^2     d^2E/dfdv  ]       [  8   -2  ]", True, CHART_LINE_COST), (box_h.x + 20, box_h.y + 12))
        surface.blit(self.fonts["mono_bold"].render("  H =  [                           ]  =    [          ]", True, CHART_LINE_COST), (box_h.x + 20, box_h.y + 26))
        surface.blit(self.fonts["mono_bold"].render("       [  d^2E/dvdf     d^2E/dv^2  ]       [ -2   12  ]", True, CHART_LINE_COST), (box_h.x + 20, box_h.y + 40))
        ry += 80

        # Paso 5: Criterio de Sylvester
        p5_col = (255, 215, 0) if self.step_hessian == 5 else TEXT_PRIMARY
        surface.blit(self.fonts["header"].render("5. Criterio de Sylvester (Menores Principales):", True, p5_col), (right_rect.x + 16, ry))
        ry += 24

        surface.blit(self.fonts["mono_bold"].render("  * Menor Principal 1:  Delta 1 = |H11| = 8 > 0", True, SERVER_NORMAL), (right_rect.x + 20, ry))
        ry += 20
        surface.blit(self.fonts["mono_bold"].render("  * Menor Principal 2:  Delta 2 = det(H) = (8)(12) - (-2)^2 = 92 > 0", True, SERVER_NORMAL), (right_rect.x + 20, ry))
        ry += 26

        # Veredicto de Sylvester animado
        verdict_box = pygame.Rect(right_rect.x + 16, ry, right_w - 32, 54)
        is_highlight = (self.step_hessian == 5)
        bg_v = (20, 50, 32) if is_highlight else (18, 22, 28)
        border_v = SERVER_NORMAL if is_highlight else BORDER_COLOR
        pygame.draw.rect(surface, bg_v, verdict_box, border_radius=8)
        pygame.draw.rect(surface, border_v, verdict_box, width=2 if is_highlight else 1, border_radius=8)

        v_t1 = self.fonts["header"].render("VEREDICTO: MATRIZ DEFINIDA POSITIVA (Delta 1 > 0 y Delta 2 > 0)", True, SERVER_NORMAL)
        v_t2 = self.fonts["mono_bold"].render("==> EL PUNTO (3.80 GHz, 3.13 V) ES UN MINIMO LOCAL ESTRICTO", True, (255, 235, 120))
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
        surface.blit(self.fonts["header"].render(f"Iteracion Actual: Paso k = {k}", True, (255, 215, 0)), (left_rect.x + 16, y))
        y += 24

        # Estado en vivo
        stats = [
            ("Potencia del Ventilador P:", f"{curr_it.coords['P']:.2f} W", ROUTER_ACTIVE),
            ("Valor de Costo J(P):", f"{curr_it.f_val:.4f}", CHART_LINE_COST),
            ("Norma del Gradiente ||grad(J)||:", f"{curr_it.grad_norm:.6f}", SERVER_NORMAL),
            ("Tasa de Aprendizaje alpha:", "0.08", TEXT_SECONDARY),
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

        surface.blit(self.fonts["small"].render("Ecuacion de Actualizacion Iterativa:", True, TEXT_MUTED), (box_eq.x + 10, box_eq.y + 6))
        eq_txt = f"P(k+1) = {curr_it.coords['P']:.2f} - (0.08 * grad(J))"
        surface.blit(self.fonts["mono_bold"].render(eq_txt, True, (255, 215, 0)), (box_eq.x + 10, box_eq.y + 24))
        y += 64

        # Tabla de los últimos pasos
        surface.blit(self.fonts["header"].render("Historial Reciente de Convergencia:", True, TEXT_PRIMARY), (left_rect.x + 16, y))
        y += 22

        header_txt = f"{'Paso':<6} {'P (W)':<10} {'Costo J':<12} {'||grad||':<10}"
        surface.blit(self.fonts["mono_bold"].render(header_txt, True, ROUTER_ACTIVE), (left_rect.x + 16, y))
        y += 16

        start_show = max(0, k - 4)
        for it in hist[start_show:k + 1]:
            is_active = (it.step == k)
            row_col = (255, 215, 0) if is_active else TEXT_MUTED
            row_txt = f"{it.step:<6} {it.coords['P']:<10.2f} {it.f_val:<12.2f} {it.grad_norm:<10.4f}"
            surface.blit(self.fonts["mono"].render(row_txt, True, row_col), (left_rect.x + 16, y))
            y += 16

        # Panel Derecho: Curva de la Parábola de Costo y Bolita Rodante
        right_rect = pygame.Rect(rect.x + left_w + 20, rect.y, right_w, rect.height)
        pygame.draw.rect(surface, CARD_BG, right_rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, right_rect, width=1, border_radius=8)

        g_title = self.fonts["header"].render("Trayectoria Dinamica sobre la Parabola de Costo J(P)", True, TEXT_PRIMARY)
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

        lbl_ball = self.fonts["small"].render(f"P({k}) = {curr_p:.1f}W", True, (255, 215, 0))
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
        surface.blit(self.fonts["header"].render("1. Funcion Objetivo a Maximizar R(P):", True, p1_col), (left_rect.x + 16, y))
        y += 22
        surface.blit(self.fonts["mono"].render("  R = (8P1 - 0.025P1^2) + (12P2 - 0.04P2^2) + (15P3 - 0.02P3^2)", True, ROUTER_ACTIVE), (left_rect.x + 16, y))
        y += 20
        surface.blit(self.fonts["small"].render("  Rendimientos marginales decrecientes (terminos cuadraticos -0.5*beta*P^2).", True, TEXT_MUTED), (left_rect.x + 16, y))
        y += 26

        # Paso 2: Restricción
        p2_col = (255, 215, 0) if self.step_lagr == 2 else TEXT_PRIMARY
        surface.blit(self.fonts["header"].render("2. Restriccion de Presupuesto Global g(P) = 0:", True, p2_col), (left_rect.x + 16, y))
        y += 22
        surface.blit(self.fonts["mono"].render(f"  P1 + P2 + P3 - {self.lab_budget:.0f} = 0  (Presupuesto = {self.lab_budget:.0f} kW)", True, SERVER_WARNING), (left_rect.x + 16, y))
        y += 26

        # Paso 3: Lagrangiano
        p3_col = (255, 215, 0) if self.step_lagr == 3 else TEXT_PRIMARY
        surface.blit(self.fonts["header"].render("3. Funcion Lagrangiana Ensamblada L(P, lambda):", True, p3_col), (left_rect.x + 16, y))
        y += 22
        surface.blit(self.fonts["mono"].render(f"  L = R(P) - lambda * (P1 + P2 + P3 - {self.lab_budget:.0f})", True, CHART_LINE_COST), (left_rect.x + 16, y))
        y += 26

        # Paso 4: Condiciones
        p4_col = (255, 215, 0) if self.step_lagr == 4 else TEXT_PRIMARY
        surface.blit(self.fonts["header"].render("4. Condiciones Necesarias de Primer Orden grad(L) = 0:", True, p4_col), (left_rect.x + 16, y))
        y += 20
        surface.blit(self.fonts["mono"].render("  * dL/dP1 = 8 - 0.05P1 - lambda = 0 ==> P1 = (8 - lambda) / 0.05", True, SERVER_NORMAL), (left_rect.x + 16, y))
        y += 18
        surface.blit(self.fonts["mono"].render("  * dL/dP2 = 12 - 0.08P2 - lambda = 0 ==> P2 = (12 - lambda) / 0.08", True, SERVER_NORMAL), (left_rect.x + 16, y))
        y += 18
        surface.blit(self.fonts["mono"].render("  * dL/dP3 = 15 - 0.04P3 - lambda = 0 ==> P3 = (15 - lambda) / 0.04", True, SERVER_NORMAL), (left_rect.x + 16, y))
        y += 18
        surface.blit(self.fonts["mono"].render(f"  * dL/dlambda = -(P1 + P2 + P3 - {self.lab_budget:.0f}) = 0", True, (255, 215, 0)), (left_rect.x + 16, y))

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

        txt_l1 = f"Multiplicador Optimo lambda* = {lambda_val:.4f}"
        txt_l2 = f"(Rendimiento marginal: +{lambda_val:.2f} unidades por cada kW extra)"
        surface.blit(self.fonts["mono_bold"].render(txt_l1, True, (163, 113, 247)), (box_lam.x + 12, box_lam.y + 6))
        surface.blit(self.fonts["small"].render(txt_l2, True, TEXT_MUTED), (box_lam.x + 12, box_lam.y + 22))
        ry += 48

        # Barras de Asignación por Clúster
        if self.lagr_res.best_point:
            clusters_info = [
                ("Cluster WEB (P1)", self.lagr_res.best_point.get("P1", 26.0), ROUTER_ACTIVE),
                ("Cluster DB (P2)", self.lagr_res.best_point.get("P2", 66.0), SERVER_NORMAL),
                ("Cluster IA (P3)", self.lagr_res.best_point.get("P3", 208.0), (163, 113, 247))
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
            surface.blit(self.fonts["mono_bold"].render(f"Rendimiento Global Maximo R* = {val_r:.2f} unidades", True, (255, 235, 120)), (right_rect.x + 16, ry + 6))
