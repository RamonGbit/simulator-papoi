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
    WINDOW_WIDTH, WINDOW_HEIGHT, FPS, BG_DARK, get_fonts,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, BORDER_COLOR, CARD_BG
)
from src.ui.components.button import UIActionButton
from src.ui.renderers.network_view import NetworkTopologyRenderer
from src.ui.renderers.chart_view import RealtimeChartsRenderer
from src.ui.renderers.metrics_hud import MetricsHUDRenderer
from src.ui.renderers.nonlinear_lab_view import NonLinearLabRenderer



class DataCenterDashboardApp:
    """
    Aplicación interactiva de escritorio en Pygame.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        pygame.init()
        pygame.display.set_caption("DataStream Papoi Networks — Simulador de Optimización para Datacenters (POO)")

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

        # 3. Componentes Visuales y Renderizadores Escalados a 1600x940
        # HUD Superior de KPIs
        self.hud_renderer = MetricsHUDRenderer(
            top_rect=(20, 12, WINDOW_WIDTH - 40, 68),
            bottom_rect=(20, 660, 750, 230),
            fonts=self.fonts
        )

        # Lienzo de Topología Ampliado (Izquierda)
        self.network_renderer = NetworkTopologyRenderer(
            rect=(20, 88, 930, 560),
            config=config,
            fonts=self.fonts
        )

        # Gráficas en tiempo real Ampliadas (Derecha)
        self.charts_renderer = RealtimeChartsRenderer(
            rect=(970, 88, 610, 560),
            fonts=self.fonts
        )

        # Panel de Control Inferior Derecho
        self.controls_rect = pygame.Rect(790, 660, 790, 230)

        # 4. Laboratorio de Programación No Lineal (Modal / Vista HUD)
        self.show_nonlinear_lab: bool = False
        self.nonlinear_lab_renderer = NonLinearLabRenderer(
            rect=(80, 40, WINDOW_WIDTH - 160, WINDOW_HEIGHT - 80),
            fonts=self.fonts
        )

        # 5. Botones de Control Interactivo (Variables y Operaciones)
        self.buttons: list[UIActionButton] = []
        self._init_buttons()

    def _init_buttons(self) -> None:
        """Inicializa los botones de control de variables y operaciones."""
        cx = self.controls_rect.x
        cy = self.controls_rect.y

        # FILA 1: Botones de Variables Numéricas (x = cx + 16, y = cy + 58)
        # Grupo A: Presupuesto Lagrange
        btn_bud_dec = UIActionButton(
            rect=(cx + 16, cy + 58, 115, 34),
            text="[-] 25 kW",
            font=self.fonts["regular"],
            callback=self._dec_budget,
            bg_color=(35, 42, 54)
        )
        btn_bud_inc = UIActionButton(
            rect=(cx + 137, cy + 58, 115, 34),
            text="[+] 25 kW",
            font=self.fonts["regular"],
            callback=self._inc_budget,
            bg_color=(35, 42, 54)
        )

        # Grupo B: Tráfico Poisson
        btn_traf_dec = UIActionButton(
            rect=(cx + 276, cy + 58, 115, 34),
            text="[-] 2 pkt/s",
            font=self.fonts["regular"],
            callback=self._dec_traffic,
            bg_color=(35, 42, 54)
        )
        btn_traf_inc = UIActionButton(
            rect=(cx + 397, cy + 58, 115, 34),
            text="[+] 2 pkt/s",
            font=self.fonts["regular"],
            callback=self._inc_traffic,
            bg_color=(35, 42, 54)
        )

        # Grupo C: Temperatura Target
        btn_temp_dec = UIActionButton(
            rect=(cx + 532, cy + 58, 115, 34),
            text="[-] 2 C",
            font=self.fonts["regular"],
            callback=self._dec_target_temp,
            bg_color=(35, 42, 54)
        )
        btn_temp_inc = UIActionButton(
            rect=(cx + 653, cy + 58, 115, 34),
            text="[+] 2 C",
            font=self.fonts["regular"],
            callback=self._inc_target_temp,
            bg_color=(35, 42, 54)
        )

        # FILA 2: Acciones del Sistema y Contingencia (y = cy + 104)
        act_w = 175
        btn_pause = UIActionButton(
            rect=(cx + 16, cy + 104, act_w, 36),
            text="Pausa / Reanudar [ESP]",
            font=self.fonts["regular"],
            callback=self._toggle_pause,
            bg_color=(33, 38, 45)
        )

        btn_fail = UIActionButton(
            rect=(cx + 203, cy + 104, act_w, 36),
            text="Falla Router [F]",
            font=self.fonts["regular"],
            callback=self._trigger_failure,
            bg_color=(54, 24, 28)
        )

        btn_burst = UIActionButton(
            rect=(cx + 390, cy + 104, act_w, 36),
            text="Rafaga Trafico [T]",
            font=self.fonts["regular"],
            callback=self._inject_traffic_burst,
            bg_color=(32, 45, 62)
        )

        btn_recover = UIActionButton(
            rect=(cx + 577, cy + 104, 180, 36),
            text="Reparar Nodos [R]",
            font=self.fonts["regular"],
            callback=self._recover_nodes,
            bg_color=(25, 48, 32)
        )

        # FILA 3: Laboratorio No Lineal, Test de Estrés y Reporte PDF (y = cy + 152)
        btn_nonlin = UIActionButton(
            rect=(cx + 16, cy + 152, 240, 44),
            text="LABORATORIO NO LINEAL [N]",
            font=self.fonts["header"],
            callback=self._toggle_nonlinear_lab,
            bg_color=(20, 60, 95),
            text_color=(120, 220, 255)
        )

        btn_stress = UIActionButton(
            rect=(cx + 266, cy + 152, 230, 44),
            text="TEST ESTRÉS TÉRMICO [S]",
            font=self.fonts["header"],
            callback=self._trigger_thermal_stress,
            bg_color=(60, 20, 25),
            text_color=(255, 160, 160)
        )

        btn_pdf = UIActionButton(
            rect=(cx + 506, cy + 152, 251, 44),
            text="EXPORTAR REPORTE PDF [P]",
            font=self.fonts["header"],
            callback=self._generate_pdf_report,
            bg_color=(68, 34, 88),
            text_color=(255, 235, 120)
        )

        self.buttons.extend([
            btn_bud_dec, btn_bud_inc,
            btn_traf_dec, btn_traf_inc,
            btn_temp_dec, btn_temp_inc,
            btn_pause, btn_fail, btn_burst, btn_recover,
            btn_nonlin, btn_stress, btn_pdf
        ])

    def _toggle_nonlinear_lab(self) -> None:
        self.show_nonlinear_lab = not self.show_nonlinear_lab
        if self.show_nonlinear_lab:
            self._set_toast("Laboratorio No Lineal Abierto (Presiona [1-4] para alternar metodo o [N] para salir)")
        else:
            self._set_toast("Regresando a Topologia de Red")


    def _inc_budget(self) -> None:
        val = self.simulation.adjust_power_budget(+25.0)
        self._set_toast(f"Presupuesto Lagrange aumentado a {val:.1f} kW")

    def _dec_budget(self) -> None:
        val = self.simulation.adjust_power_budget(-25.0)
        self._set_toast(f"Presupuesto Lagrange reducido a {val:.1f} kW")

    def _inc_traffic(self) -> None:
        val = self.simulation.adjust_arrival_rate(+2.0)
        self._set_toast(f"Tasa de trafico lambda aumentada a {val:.1f} pkt/s")

    def _dec_traffic(self) -> None:
        val = self.simulation.adjust_arrival_rate(-2.0)
        self._set_toast(f"Tasa de trafico lambda reducida a {val:.1f} pkt/s")

    def _inc_target_temp(self) -> None:
        val = self.simulation.adjust_target_temperature(+2.0)
        self._set_toast(f"Temp objetivo de refrigeracion aumentada a {val:.1f} C")

    def _dec_target_temp(self) -> None:
        val = self.simulation.adjust_target_temperature(-2.0)
        self._set_toast(f"Temp objetivo de refrigeracion reducida a {val:.1f} C")

    def _toggle_pause(self) -> None:
        paused = self.simulation.toggle_pause()
        self._set_toast("Simulacion Pausada" if paused else "Simulacion Reanudada")

    def _trigger_failure(self) -> None:
        """Inyecta una falla manual en un router para forzar el re-enrutamiento dinámico con DP."""
        vulnerable = [r.id for r in self.config.routers if r.id != "R_IN"]
        active = [nid for nid in vulnerable if nid not in self.simulation.failure_model.failed_nodes]
        if active:
            target = random.choice(active)
            self.simulation.failure_model.trigger_manual_failure(target)
            self._set_toast(f"Falla forzada en {target}! DP recalculando ruta...")
        else:
            self._set_toast("Todos los routers vulnerables ya estan fuera de servicio.")

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
        self._set_toast("Rafaga de trafico inyectada (+8 paquetes/s)")

    def _trigger_thermal_stress(self) -> None:
        """Inyecta un choque térmico brusco en los servidores para forzar el régimen superlineal."""
        target_s = self.simulation.trigger_thermal_stress("S_WEB", duration=6.0)
        self._set_toast(f"¡TEST ESTRÉS en {target_s}! Carga 95%, cruza inflexión 48°C y acelera ventilador.")

    def _generate_pdf_report(self) -> None:
        """Genera el reporte PDF y consulta la IA."""
        self._set_toast("Generando auditoria PDF y consultando IA...")
        try:
            snapshot = self.simulation.get_snapshot()
            custom_probs = getattr(self.nonlinear_lab_renderer, "custom_saved_problems", [])
            pdf_path = self.pdf_generator.generate_report(snapshot, custom_problems=custom_probs)
            filename = os.path.basename(pdf_path)
            self._set_toast(f"Reporte generado: reports/{filename}")
        except Exception as e:
            self._set_toast(f"Error al exportar PDF: {e}")

    def _set_toast(self, msg: str) -> None:
        self.status_toast = msg
        self.status_toast_timer = time.time()

    def _draw_controls_panel(self, surface: pygame.Surface) -> None:
        """Dibuja el panel de control inferior derecho con etiquetas en vivo más grandes y destacadas."""
        r = self.controls_rect
        pygame.draw.rect(surface, (18, 22, 28), r, border_radius=8)
        pygame.draw.rect(surface, (48, 54, 61), r, width=1, border_radius=8)

        # Título de la sección
        lbl_title = self.fonts["header"].render("PANEL DE CONTROL DE VARIABLES Y OPERACIONES", True, TEXT_PRIMARY)
        surface.blit(lbl_title, (r.x + 16, r.y + 8))

        # Etiquetas en vivo de las variables más grandes (var_label 14pt bold)
        lbl_bud = self.fonts["var_label"].render(
            f"Presupuesto Lagrange: {self.config.power_budget_total:.0f} kW",
            True,
            (255, 215, 0)
        )
        surface.blit(lbl_bud, (r.x + 16, r.y + 34))

        lbl_traf = self.fonts["var_label"].render(
            f"Trafico lambda: {self.config.simulation_defaults.arrival_rate_lambda:.1f} pkt/s",
            True,
            (88, 166, 255)
        )
        surface.blit(lbl_traf, (r.x + 276, r.y + 34))

        lbl_temp = self.fonts["var_label"].render(
            f"Temp Target: {self.config.servers[0].target_temp:.1f} C",
            True,
            (219, 97, 162)
        )
        surface.blit(lbl_temp, (r.x + 532, r.y + 34))

    def run(self) -> None:
        """Bucle interactivo principal."""
        while self.is_running:
            dt = self.clock.tick(FPS) / 1000.0  # Delta time en segundos reales

            # 1. Manejo de Eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                elif self.show_nonlinear_lab:
                    # Modo Modal: Laboratorio No Lineal activo
                    if event.type == pygame.KEYDOWN:
                        handled = self.nonlinear_lab_renderer.handle_key_event(event)
                        if not handled:
                            if event.key in (pygame.K_ESCAPE, pygame.K_n):
                                self.show_nonlinear_lab = False
                                self._set_toast("Regresando al Datacenter")
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self.nonlinear_lab_renderer.handle_click(event.pos)
                        if getattr(self.nonlinear_lab_renderer, "should_close", False):
                            self.show_nonlinear_lab = False
                            self.nonlinear_lab_renderer.should_close = False
                            self._set_toast("Regresando al Datacenter")
                else:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.is_running = False
                        elif event.key == pygame.K_n:
                            self._toggle_nonlinear_lab()
                        elif event.key == pygame.K_SPACE:
                            self._toggle_pause()
                        elif event.key == pygame.K_f:
                            self._trigger_failure()
                        elif event.key == pygame.K_r:
                            self._recover_nodes()
                        elif event.key == pygame.K_t:
                            self._inject_traffic_burst()
                        elif event.key == pygame.K_s:
                            self._trigger_thermal_stress()
                        elif event.key == pygame.K_p:
                            self._generate_pdf_report()
                        # Modificación dinámica de variables por teclado
                        elif event.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
                            self._inc_budget()
                        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                            self._dec_budget()
                        elif event.key == pygame.K_UP:
                            self._inc_traffic()
                        elif event.key == pygame.K_DOWN:
                            self._dec_traffic()
                        elif event.key == pygame.K_PAGEUP:
                            self._inc_target_temp()
                        elif event.key == pygame.K_PAGEDOWN:
                            self._dec_target_temp()

                    # Delegar a botones
                    for btn in self.buttons:
                        btn.handle_event(event)

            # 2. Actualización de la Simulación SimPy
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

            # Panel de Control y Botones
            self._draw_controls_panel(self.screen)
            for btn in self.buttons:
                btn.draw(self.screen)

            # Si el laboratorio no lineal está activo, renderizar el overlay
            if self.show_nonlinear_lab:
                self.nonlinear_lab_renderer.render(self.screen)

            # Barra informativa permanente de atajos para cambiar variables
            footer_rect = pygame.Rect(20, WINDOW_HEIGHT - 30, WINDOW_WIDTH - 40, 24)
            pygame.draw.rect(self.screen, (16, 20, 28), footer_rect, border_radius=4)
            pygame.draw.rect(self.screen, (38, 48, 64), footer_rect, width=1, border_radius=4)
            footer_txt = "CONTROLES: [N] Laboratorio No Lineal | [+/-] Presupuesto | [UP/DOWN] Trafico | [PgUp/PgDn] Temp | [F] Falla | [R] Reparar | [P] PDF"
            f_surf = self.fonts["small"].render(footer_txt, True, (139, 148, 158))
            self.screen.blit(f_surf, (footer_rect.x + 12, footer_rect.y + 4))

            # Toast flotante de estado por encima de la barra
            if self.status_toast and (time.time() - self.status_toast_timer < 4.0):
                toast_rect = pygame.Rect(20, WINDOW_HEIGHT - 60, WINDOW_WIDTH - 40, 26)
                pygame.draw.rect(self.screen, (24, 30, 42), toast_rect, border_radius=4)
                pygame.draw.rect(self.screen, (255, 215, 0), toast_rect, width=1, border_radius=4)
                t_surf = self.fonts["mono_bold"].render(f"[*] {self.status_toast}", True, (255, 215, 0))
                self.screen.blit(t_surf, (toast_rect.x + 12, toast_rect.y + 5))

            pygame.display.flip()

        pygame.quit()
