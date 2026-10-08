"""
Renderizador de Gráficos en Tiempo Real para el Método de Gradiente y Multiplicadores de Lagrange.
Dibuja curvas dinámicas de convergencia y barras comparativas de asignación de energía.
"""

from __future__ import annotations
import pygame
from typing import Dict, List, Tuple
from src.core.state import DataCenterSnapshot
from src.ui.theme import (
    CARD_BG, BORDER_COLOR, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    CHART_LINE_GRADIENT, CHART_LINE_COST, CHART_GRID, CHART_AXIS,
    CLUSTER_COLORS
)


class RealtimeChartsRenderer:
    """
    Gestiona el renderizado de dos sub-gráficos en vivo:
    1. Descenso de Gradiente (Potencia de ventilador vs Costo)
    2. Multiplicadores de Lagrange (Asignación de energía por clúster)
    """

    def __init__(self, rect: Tuple[int, int, int, int], fonts: Dict[str, pygame.font.Font]):
        self.rect = pygame.Rect(rect)
        self.fonts = fonts

        # Dividir el área en 2 subpaneles verticales
        pad = 8
        half_h = (self.rect.height - (3 * pad)) // 2
        self.rect_grad = pygame.Rect(self.rect.x, self.rect.y + pad, self.rect.width, half_h)
        self.rect_lagrange = pygame.Rect(self.rect.x, self.rect.y + (2 * pad) + half_h, self.rect.width, half_h)

    def draw(self, surface: pygame.Surface, snapshot: DataCenterSnapshot) -> None:
        self._draw_gradient_chart(surface, snapshot)
        self._draw_lagrange_chart(surface, snapshot)

    def _draw_gradient_chart(self, surface: pygame.Surface, snapshot: DataCenterSnapshot) -> None:
        """Dibuja la gráfica temporal de la potencia del ventilador ajustada por Gradiente Descendente."""
        r = self.rect_grad
        pygame.draw.rect(surface, (18, 22, 28), r, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, r, width=1, border_radius=8)

        # Título
        title = self.fonts["header"].render("MÉTODO DE GRADIENTE: CONTROL DE ENFRIAMIENTO", True, TEXT_PRIMARY)
        surface.blit(title, (r.x + 12, r.y + 10))

        # Área de trazado
        margin_l, margin_r, margin_t, margin_b = 40, 16, 36, 26
        plot_x = r.x + margin_l
        plot_y = r.y + margin_t
        plot_w = r.width - margin_l - margin_r
        plot_h = r.height - margin_t - margin_b

        # Fondo del plot
        pygame.draw.rect(surface, (14, 17, 22), (plot_x, plot_y, plot_w, plot_h))
        pygame.draw.rect(surface, CHART_AXIS, (plot_x, plot_y, plot_w, plot_h), width=1)

        # Tomar historial de potencia del primer servidor activo o promedio
        hist_p: List[float] = []
        for s_id, h in snapshot.cooling_power_history.items():
            if h:
                hist_p = h
                break

        if len(hist_p) >= 2:
            max_val = max(120.0, max(hist_p) * 1.1)
            min_val = 0.0

            # Líneas de cuadrícula horizontal
            for i in range(1, 4):
                gy = plot_y + int((plot_h / 4) * i)
                pygame.draw.line(surface, CHART_GRID, (plot_x, gy), (plot_x + plot_w, gy), 1)

            # Trazo de puntos
            n_pts = len(hist_p)
            points = []
            for idx, val in enumerate(hist_p):
                px = plot_x + int((idx / (n_pts - 1)) * plot_w)
                norm_y = (val - min_val) / (max_val - min_val)
                py = plot_y + plot_h - int(norm_y * plot_h)
                points.append((px, py))

            if len(points) >= 2:
                pygame.draw.lines(surface, CHART_LINE_GRADIENT, False, points, width=2)

            curr_p = hist_p[-1]
            legend_txt = f"Potencia P_fan: {curr_p:.1f} W (grad J -> min)"
        else:
            legend_txt = "Optimizando gradiente en tiempo real..."

        lbl = self.fonts["small"].render(legend_txt, True, CHART_LINE_GRADIENT)
        surface.blit(lbl, (plot_x + 8, plot_y + 6))

        # Etiquetas de eje
        surface.blit(self.fonts["small"].render("120W", True, TEXT_MUTED), (r.x + 8, plot_y - 2))
        surface.blit(self.fonts["small"].render("0W", True, TEXT_MUTED), (r.x + 12, plot_y + plot_h - 10))

    def _draw_lagrange_chart(self, surface: pygame.Surface, snapshot: DataCenterSnapshot) -> None:
        """Dibuja el gráfico de barras de distribución de presupuesto eléctrico de Lagrange."""
        r = self.rect_lagrange
        pygame.draw.rect(surface, (18, 22, 28), r, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, r, width=1, border_radius=8)

        # Título conciso que no invade el badge derecho
        title = self.fonts["header"].render("ASIGNACIÓN ENERGÉTICA (LAGRANGE)", True, TEXT_PRIMARY)
        surface.blit(title, (r.x + 12, r.y + 10))

        # Multiplicador lambda renderizado como badge compacto
        lam_txt = f"lambda* = {snapshot.lagrange_lambda:.4f}"
        lam_surf = self.fonts["mono_bold"].render(lam_txt, True, (255, 215, 0))
        badge_w = lam_surf.get_width() + 16
        badge_rect = pygame.Rect(r.x + r.width - badge_w - 12, r.y + 8, badge_w, 22)
        pygame.draw.rect(surface, (38, 32, 16), badge_rect, border_radius=4)
        pygame.draw.rect(surface, (210, 153, 34), badge_rect, width=1, border_radius=4)
        surface.blit(lam_surf, (badge_rect.x + 8, badge_rect.y + 3))

        # Área de trazado
        margin_l, margin_r, margin_t, margin_b = 40, 16, 40, 30
        plot_x = r.x + margin_l
        plot_y = r.y + margin_t
        plot_w = r.width - margin_l - margin_r
        plot_h = r.height - margin_t - margin_b

        pygame.draw.rect(surface, (14, 17, 22), (plot_x, plot_y, plot_w, plot_h))
        pygame.draw.rect(surface, CHART_AXIS, (plot_x, plot_y, plot_w, plot_h), width=1)

        allocs = snapshot.lagrange_allocation
        if allocs:
            max_budget = 300.0
            n_bars = len(allocs)
            slot_w = plot_w / n_bars
            bar_w = int(slot_w * 0.55)

            for idx, (cid, p_val) in enumerate(allocs.items()):
                bx = plot_x + int(idx * slot_w + (slot_w - bar_w) / 2)
                bar_h = int((p_val / max_budget) * plot_h)
                by = plot_y + plot_h - bar_h

                bar_col = CLUSTER_COLORS.get(cid, (88, 166, 255))
                pygame.draw.rect(surface, bar_col, (bx, by, bar_w, bar_h), border_radius=4)

                # Etiqueta valor kW
                val_surf = self.fonts["small"].render(f"{p_val:.1f}kW", True, TEXT_PRIMARY)
                surface.blit(val_surf, (bx, max(plot_y + 4, by - 14)))

                # Etiqueta nombre clúster
                c_lbl = self.fonts["small"].render(cid.replace("cluster_", "").upper(), True, TEXT_SECONDARY)
                surface.blit(c_lbl, (bx + 2, plot_y + plot_h + 8))

        # Línea de referencia total
        lbl_tot = self.fonts["small"].render("Capacidad: 300 kW", True, TEXT_MUTED)
        surface.blit(lbl_tot, (plot_x + 8, plot_y + 6))
