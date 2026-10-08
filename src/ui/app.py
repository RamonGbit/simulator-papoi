"""
Bucle principal y coordinador de la Interfaz Gráfica Pygame (Dashboard NOC).
Integra en tiempo real la topología de red, gráficas dinámicas de optimización,
controles interactivos de simulación y exportación de reportes PDF con IA.
"""

from __future__ import annotations
import os
import sys
import time
import random
import pygame

from config.settings import AppConfig
from src.simulation.engine import DataCenterSimulation
from src.reporting.pdf_generator import PDFReportGenerator
from src.ui.theme import (
    WINDOW_WIDTH, WINDOW_HEIGHT, FPS, BG_DARK, get_fonts
)
from src.ui.components.button import UIActionButton
from src.ui.renderers.network_view import NetworkTopologyRenderer
from src.ui.renderers.chart_view import RealtimeChartsRenderer
from src.ui.renderers.metrics_hud import MetricsHUDRenderer


class DataCenterDashboardApp:
    """
    Aplicación interactiva de escritorio en Pygame.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        pygame.init()
        pygame.display.set_caption("DataStream Networks — Simulador de Optimización para Datacenters (POO)")

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.is_running = True
        self.fonts = get_fonts()

        # 1. Instanciar Motor de Simulación SimPy
        self.simulation = DataCenterSimulation(config)

        # 2. Generador de Reportes PDF
        self.pdf_generator = PDFReportGenerator()
        self.status_toast: str = ""
        self.status_toast_timer: float = 0.0

        # 3. Componentes Visuales y Renderizadores
        # HUD Superior
        self.hud_renderer = MetricsHUDRenderer(
            top_rect=(20, 15, WINDOW_WIDTH - 40, 65),
            bottom_rect=(20, 625, 830, 80),
            fonts=self.fonts
        )

        # Lienzo de Topología
        self.network_renderer = NetworkTopologyRenderer(
            rect=(20, 90, 830, 520),
            config=config,
            fonts=self.fonts
        )

        # Gráficas en tiempo real (Gradiente y Lagrange)
        self.charts_renderer = RealtimeChartsRenderer(
            rect=(870, 90, 510, 520),
            fonts=self.fonts
        )

        # 4. Botones de Control Interactivo
        self.buttons: list[UIActionButton] = []
        self._init_buttons()

    def _init_buttons(self) -> None:
        """Inicializa los botones de acción ubicados en la barra inferior derecha."""
        start_x = 870
        btn_y = 625
        w1, w2, h = 160, 160, 36

        # Fila 1
        btn_pause = UIActionButton(
            rect=(start_x, btn_y, w1, h),
            text="Pausa / Reanudar [ESP]",
            font=self.fonts["regular"],
            callback=self._toggle_pause,
            bg_color=(33, 38, 45)
        )

        btn_fail = UIActionButton(
            rect=(start_x + w1 + 10, btn_y, w1, h),
            text="Falla Router [F]",
            font=self.fonts["regular"],
            callback=self._trigger_failure,
            bg_color=(50, 20, 25)
        )

        btn_burst = UIActionButton(
            rect=(start_x + (w1 * 2) + 20, btn_y, 160, h),
            text="Ráfaga Tráfico [T]",
            font=self.fonts["regular"],
            callback=self._inject_traffic_burst,
            bg_color=(35, 45, 60)
        )

        # Fila 2
        btn_recover = UIActionButton(
            rect=(start_x, btn_y + h + 8, w1, h),
            text="Reparar Nodos [R]",
            font=self.fonts["regular"],
            callback=self._recover_nodes,
            bg_color=(25, 45, 30)
        )

        btn_pdf = UIActionButton(
            rect=(start_x + w1 + 10, btn_y + h + 8, 330, h),
            text="EXPORTAR REPORTE PDF + IA [P]",
            font=self.fonts["header"],
            callback=self._generate_pdf_report,
            bg_color=(60, 30, 75)
        )

        self.buttons.extend([btn_pause, btn_fail, btn_burst, btn_recover, btn_pdf])

    def _toggle_pause(self) -> None:
        paused = self.simulation.toggle_pause()
        self._set_toast("Simulación Pausada" if paused else "Simulación Reanudada")

    def _trigger_failure(self) -> None:
        """Inyecta una falla manual en un router para forzar el re-enrutamiento dinámico con DP."""
        vulnerable = [r.id for r in self.config.routers if r.id != "R_IN"]
        active = [nid for nid in vulnerable if nid not in self.simulation.failure_model.failed_nodes]
        if active:
            target = random.choice(active)
            self.simulation.failure_model.trigger_manual_failure(target)
            self._set_toast(f"¡Falla forzada en {target}! DP recalculando ruta...")
        else:
            self._set_toast("Todos los routers vulnerables ya están fuera de servicio.")

    def _recover_nodes(self) -> None:
        """Restaura todos los nodos caídos."""
        failed = list(self.simulation.failure_model.failed_nodes)
        for nid in failed:
            self.simulation.failure_model.trigger_manual_recovery(nid)
        self._set_toast("Todos los nodos han sido reparados y reconectados.")

    def _inject_traffic_burst(self) -> None:
        """Dispara una ráfaga de paquetes entrantes."""
        for _ in range(8):
            self.simulation.traffic_generator.action = self.simulation.env.process(
                self.simulation.traffic_generator._run_generator()
            )
        self._set_toast("Ráfaga de tráfico inyectada (+8 paquetes/s)")

    def _generate_pdf_report(self) -> None:
        """Genera el reporte PDF y consulta la IA."""
        self._set_toast("Generando auditoría PDF y consultando IA...")
        try:
            snapshot = self.simulation.get_snapshot()
            pdf_path = self.pdf_generator.generate_report(snapshot)
            filename = os.path.basename(pdf_path)
            self._set_toast(f"Reporte generado: reports/{filename}")
        except Exception as e:
            self._set_toast(f"Error al exportar PDF: {e}")

    def _set_toast(self, msg: str) -> None:
        self.status_toast = msg
        self.status_toast_timer = time.time()

    def run(self) -> None:
        """Bucle interactivo principal."""
        while self.is_running:
            dt = self.clock.tick(FPS) / 1000.0  # Delta time en segundos reales

            # 1. Manejo de Eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.is_running = False
                    elif event.key == pygame.K_SPACE:
                        self._toggle_pause()
                    elif event.key == pygame.K_f:
                        self._trigger_failure()
                    elif event.key == pygame.K_r:
                        self._recover_nodes()
                    elif event.key == pygame.K_t:
                        self._inject_traffic_burst()
                    elif event.key == pygame.K_p:
                        self._generate_pdf_report()

                # Delegar a botones
                for btn in self.buttons:
                    btn.handle_event(event)

            # 2. Actualización de la Simulación SimPy
            # Avanzamos la simulación temporal en proporción al delta time real
            self.simulation.advance_time(dt * 1.5)
            snapshot = self.simulation.get_snapshot()

            # 3. Renderizado Gráfico
            self.screen.fill(BG_DARK)

            # HUD y Consola
            self.hud_renderer.draw_top_hud(self.screen, snapshot)
            self.hud_renderer.draw_bottom_console(self.screen, snapshot)

            # Topología de Red y Enrutamiento DP
            self.network_renderer.draw(self.screen, snapshot)

            # Gráficas de Gradiente y Lagrange
            self.charts_renderer.draw(self.screen, snapshot)

            # Botones de Acción
            for btn in self.buttons:
                btn.draw(self.screen)

            # Toast flotante de estado
            if self.status_toast and (time.time() - self.status_toast_timer < 4.0):
                toast_rect = pygame.Rect(20, WINDOW_HEIGHT - 30, WINDOW_WIDTH - 40, 24)
                pygame.draw.rect(self.screen, (24, 30, 42), toast_rect, border_radius=4)
                t_surf = self.fonts["mono_bold"].render(f"🔔 {self.status_toast}", True, (255, 215, 0))
                self.screen.blit(t_surf, (toast_rect.x + 10, toast_rect.y + 4))

            pygame.display.flip()

        pygame.quit()
