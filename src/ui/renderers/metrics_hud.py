"""
Renderizador del HUD superior de Métricas Operativas y Consola de Alertas inferior.
"""

from __future__ import annotations
import pygame
from typing import Dict, List, Tuple
from src.core.state import DataCenterSnapshot
from src.ui.theme import (
    CARD_BG, BORDER_COLOR, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    SERVER_EMERGENCY, ROUTER_ACTIVE
)


class MetricsHUDRenderer:
    """
    Dibuja los paneles de métricas globales y la consola de eventos en tiempo real.
    """

    def __init__(
        self,
        top_rect: Tuple[int, int, int, int],
        bottom_rect: Tuple[int, int, int, int],
        fonts: Dict[str, pygame.font.Font]
    ):
        self.top_rect = pygame.Rect(top_rect)
        self.bottom_rect = pygame.Rect(bottom_rect)
        self.fonts = fonts

    def draw_top_hud(self, surface: pygame.Surface, snapshot: DataCenterSnapshot) -> None:
        """Dibuja la barra de KPIs principales en la parte superior."""
        r = self.top_rect
        pygame.draw.rect(surface, (18, 22, 28), r, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, r, width=1, border_radius=8)

        # Título del Centro de Datos
        t_title = self.fonts["title"].render("DATASTREAM NETWORKS — CLOUD OPERATIONS CENTER", True, (88, 166, 255))
        surface.blit(t_title, (r.x + 16, r.y + 12))

        # Estado de Simulación y Reloj
        status_txt = "PAUSADO" if snapshot.is_paused else "EN VIVO (SimPy)"
        status_col = (210, 153, 34) if snapshot.is_paused else (63, 185, 80)
        time_txt = f"[{status_txt}] Tiempo Simulado: {snapshot.sim_time:.1f} s"
        t_time = self.fonts["mono_bold"].render(time_txt, True, status_col)
        surface.blit(t_time, (r.x + 16, r.y + 36))

        # Tarjetas de Métricas Rápidas (KPIs)
        kpi_x = r.x + 580
        kpis = [
            ("PAQUETES ENTREGADOS", f"{snapshot.metrics.total_packets_delivered}/{snapshot.metrics.total_packets_created}", (88, 166, 255)),
            ("LATENCIA MEDIA", f"{snapshot.metrics.average_latency:.2f} ms", (255, 215, 0)),
            ("VALOR PROCESADO (SLA)", f"{snapshot.metrics.total_value_processed:.1f} pts", (63, 185, 80)),
            ("PERDIDOS / CONGESTIÓN", f"{snapshot.metrics.total_packets_dropped}", (248, 81, 73) if snapshot.metrics.total_packets_dropped > 0 else TEXT_MUTED),
            ("RE-ENRUTAMIENTOS DP", f"{snapshot.metrics.path_reroute_count}", (163, 113, 247))
        ]

        slot_w = 155
        for idx, (label, val, col) in enumerate(kpis):
            bx = kpi_x + (idx * slot_w)
            pygame.draw.line(surface, BORDER_COLOR, (bx - 10, r.y + 8), (bx - 10, r.y + r.height - 8), 1)
            lbl_surf = self.fonts["small"].render(label, True, TEXT_MUTED)
            val_surf = self.fonts["header"].render(val, True, col)
            surface.blit(lbl_surf, (bx, r.y + 12))
            surface.blit(val_surf, (bx, r.y + 32))

    def draw_bottom_console(self, surface: pygame.Surface, snapshot: DataCenterSnapshot) -> None:
        """Dibuja la consola de eventos, re-enrutamientos y alertas en la parte inferior."""
        r = self.bottom_rect
        pygame.draw.rect(surface, (14, 17, 22), r, border_radius=6)
        pygame.draw.rect(surface, BORDER_COLOR, r, width=1, border_radius=6)

        # Encabezado de la consola
        lbl = self.fonts["small"].render("LOG DE EVENTOS Y DECISIONES DE OPTIMIZACIÓN EN TIEMPO REAL:", True, TEXT_MUTED)
        surface.blit(lbl, (r.x + 12, r.y + 6))

        # Mostrar las últimas 3 alertas
        alerts = snapshot.alert_messages[-3:] if snapshot.alert_messages else ["Simulación operando en condiciones normales."]
        start_y = r.y + 22
        for idx, msg in enumerate(alerts):
            col = SERVER_EMERGENCY if "ALERTA" in msg or "EMERGENCIA" in msg else ((255, 215, 0) if "REROUTE" in msg else TEXT_SECONDARY)
            msg_surf = self.fonts["mono"].render(f"• {msg}", True, col)
            surface.blit(msg_surf, (r.x + 14, start_y + (idx * 16)))
