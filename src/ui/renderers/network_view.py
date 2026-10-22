"""
Renderizador visual del Mapa de Topología de Red y Tráfico de Paquetes en Pygame.
Dibuja routers, servidores, enlaces, paquetes en movimiento y resalta la ruta óptima de DP.
"""

from __future__ import annotations
import math
import time
import pygame
from typing import Dict, List, Tuple, Any

from config.settings import AppConfig
from src.core.state import DataCenterSnapshot
from src.ui.theme import (
    ROUTER_ACTIVE, ROUTER_FAILED, SERVER_NORMAL, SERVER_WARNING, SERVER_EMERGENCY,
    LINK_IDLE, LINK_ACTIVE_PATH, LINK_GLOW, PACKET_COLOR, PACKET_GLOW,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, CARD_BG, BORDER_COLOR
)


class NetworkTopologyRenderer:
    """
    Renderiza la topología del centro de datos sobre un área rectangular dedicada.
    """

    def __init__(self, rect: Tuple[int, int, int, int], config: AppConfig, fonts: Dict[str, pygame.font.Font]):
        self.rect = pygame.Rect(rect)
        self.config = config
        self.fonts = fonts

        # Coordenadas relativas de nodos dentro del área del canvas
        # Mapeamos desde coordenadas lógicas de configuración
        self.node_positions: Dict[str, Tuple[int, int]] = {}
        self._calculate_node_positions()

        # Efecto de alerta visual de re-enrutamiento
        self.last_path_change_time: float = 0.0
        self.previous_active_path: List[str] = []

    def _calculate_node_positions(self) -> None:
        """Adapta las posiciones lógicas de la configuración al tamaño del panel visual."""
        x_min = min([r.pos[0] for r in self.config.routers] + [s.pos[0] for s in self.config.servers])
        x_max = max([r.pos[0] for r in self.config.routers] + [s.pos[0] for s in self.config.servers])
        y_min = min([r.pos[1] for r in self.config.routers] + [s.pos[1] for s in self.config.servers])
        y_max = max([r.pos[1] for r in self.config.routers] + [s.pos[1] for s in self.config.servers])

        margin_left = 65
        margin_right = 95
        margin_top = 55
        margin_bottom = 75
        draw_w = self.rect.width - margin_left - margin_right
        draw_h = self.rect.height - margin_top - margin_bottom

        span_x = max(1, x_max - x_min)
        span_y = max(1, y_max - y_min)

        for r in self.config.routers:
            px = self.rect.x + margin_left + int(((r.pos[0] - x_min) / span_x) * draw_w)
            py = self.rect.y + margin_top + int(((r.pos[1] - y_min) / span_y) * draw_h)
            self.node_positions[r.id] = (px, py)

        for s in self.config.servers:
            px = self.rect.x + margin_left + int(((s.pos[0] - x_min) / span_x) * draw_w)
            py = self.rect.y + margin_top + int(((s.pos[1] - y_min) / span_y) * draw_h)
            self.node_positions[s.id] = (px, py)

    def draw(self, surface: pygame.Surface, snapshot: DataCenterSnapshot) -> None:
        # Fondo y marco del canvas de red
        pygame.draw.rect(surface, (18, 22, 28), self.rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, self.rect, width=1, border_radius=8)

        # Título del panel
        title_surf = self.fonts["header"].render("TOPOLOGÍA DE RED Y ENRUTAMIENTO DINÁMICO (DP)", True, TEXT_PRIMARY)
        surface.blit(title_surf, (self.rect.x + 16, self.rect.y + 12))

        # Detectar si cambió la ruta óptima de DP para animar indicador
        active_path = snapshot.metrics.active_shortest_path
        if active_path and active_path != self.previous_active_path:
            self.previous_active_path = list(active_path)
            self.last_path_change_time = time.time()

        # Construir conjunto de aristas que pertenecen a la ruta activa
        active_edges = set()
        if len(active_path) >= 2:
            for i in range(len(active_path) - 1):
                active_edges.add((active_path[i], active_path[i + 1]))

        # 1. Dibujar Enlaces
        for link in self.config.links:
            u, v = link.source, link.target
            if u in self.node_positions and v in self.node_positions:
                pos_u = self.node_positions[u]
                pos_v = self.node_positions[v]
                is_on_shortest_path = (u, v) in active_edges

                if is_on_shortest_path:
                    # Enlace activo en la ruta más corta: Brillo dorado/amarillo
                    pygame.draw.line(surface, LINK_ACTIVE_PATH, pos_u, pos_v, width=4)
                    # Línea delgada interior
                    pygame.draw.line(surface, LINK_GLOW, pos_u, pos_v, width=2)
                else:
                    # Enlace normal inactivo
                    pygame.draw.line(surface, LINK_IDLE, pos_u, pos_v, width=2)

                # Mostrar latencia base
                mid_x = (pos_u[0] + pos_v[0]) // 2
                mid_y = (pos_u[1] + pos_v[1]) // 2
                lat_txt = f"{link.base_latency:.1f}ms"
                lbl = self.fonts["small"].render(lat_txt, True, TEXT_MUTED)
                surface.blit(lbl, (mid_x - 12, mid_y - 10))

        # 2. Dibujar Paquetes en Movimiento
        for pkt in snapshot.active_packets:
            if pkt.path and pkt.path_index < len(pkt.path) - 1:
                u = pkt.path[pkt.path_index]
                v = pkt.path[pkt.path_index + 1]
                if u in self.node_positions and v in self.node_positions:
                    p1 = self.node_positions[u]
                    p2 = self.node_positions[v]
                    prog = max(0.0, min(1.0, pkt.progress))
                    pkt_x = int(p1[0] + (p2[0] - p1[0]) * prog)
                    pkt_y = int(p1[1] + (p2[1] - p1[1]) * prog)

                    # Halo y partícula
                    pygame.draw.circle(surface, PACKET_GLOW, (pkt_x, pkt_y), 5)
                    pygame.draw.circle(surface, PACKET_COLOR, (pkt_x, pkt_y), 3)

        # 3. Dibujar Routers
        for r_cfg in self.config.routers:
            pos = self.node_positions.get(r_cfg.id)
            if not pos:
                continue

            r_state = snapshot.routers.get(r_cfg.id)
            is_active = (r_state.is_active if r_state else True) and (r_cfg.id not in snapshot.failed_nodes)
            queue_len = r_state.queue_size if r_state else 0

            # Color del nodo según estado
            node_color = ROUTER_ACTIVE if is_active else ROUTER_FAILED
            pygame.draw.circle(surface, (25, 30, 40), pos, 24)
            pygame.draw.circle(surface, node_color, pos, 24, width=2)

            # Icono/Texto ID
            id_surf = self.fonts["mono_bold"].render(r_cfg.id, True, TEXT_PRIMARY)
            surface.blit(id_surf, id_surf.get_rect(center=pos))

            # Indicador de Cola (Buffer)
            q_color = (63, 185, 80) if queue_len < 5 else ((210, 153, 34) if queue_len < 15 else (248, 81, 73))
            q_txt = f"Cola: {queue_len}" if is_active else "OFFLINE"
            q_surf = self.fonts["small"].render(q_txt, True, q_color)
            surface.blit(q_surf, (pos[0] - 26, pos[1] + 28))

        # 4. Dibujar Servidores con Emulación Física (Ventiladores Giratorios, Aura Térmica y Termómetro)
        now_t = time.time()
        dt_frame = min(0.1, now_t - getattr(self, "_last_draw_time", now_t))
        self._last_draw_time = now_t
        if not hasattr(self, "fan_angles"):
            self.fan_angles = {s.id: 0.0 for s in self.config.servers}

        for s_cfg in self.config.servers:
            pos = self.node_positions.get(s_cfg.id)
            if not pos:
                continue

            s_state = snapshot.servers.get(s_cfg.id)
            temp = s_state.temperature if s_state else s_cfg.ambient_temp
            load = s_state.current_load if s_state else 0.0
            fan_p = s_state.fan_power if s_state else 10.0
            p_alloc = s_state.allocated_power if s_state else 50.0
            is_emerg = s_state.is_emergency_cooling if s_state else (temp >= 48.04 or load >= 0.60)

            # Actualizar rotación física de las aspas del ventilador según la potencia W
            rpm_factor = 2.5 + (fan_p / 100.0) * 22.0  # rad/s proporcional a W
            self.fan_angles[s_cfg.id] = (self.fan_angles.get(s_cfg.id, 0.0) + rpm_factor * dt_frame) % (2.0 * math.pi)

            # Dimensiones del bloque del servidor
            box_w, box_h = 168, 88
            box_rect = pygame.Rect(pos[0] - box_w // 2, pos[1] - box_h // 2, box_w, box_h)

            # --- A. RESPLANDOR TÉRMICO / HEATMAP AURA ---
            # Se activa y pulsa con fuerza al superar el Punto de Inflexión (48.04°C)
            if is_emerg or temp >= 48.04:
                pulse = int(abs(math.sin(now_t * 8.0)) * 6) + 3
                aura_surf = pygame.Surface((box_w + 24 + pulse * 2, box_h + 24 + pulse * 2), pygame.SRCALPHA)
                aura_color = (248, 81, 73, 55 + pulse * 8)
                pygame.draw.rect(aura_surf, aura_color, aura_surf.get_rect(), border_radius=12)
                surface.blit(aura_surf, (box_rect.x - 12 - pulse, box_rect.y - 12 - pulse))
                s_border = SERVER_EMERGENCY
                bg_color = (48, 16, 20)
            elif temp >= 38.0:
                # Zona de calentamiento intermedio (Ámbar)
                aura_surf = pygame.Surface((box_w + 14, box_h + 14), pygame.SRCALPHA)
                pygame.draw.rect(aura_surf, (210, 153, 34, 40), aura_surf.get_rect(), border_radius=10)
                surface.blit(aura_surf, (box_rect.x - 7, box_rect.y - 7))
                s_border = (210, 153, 34)
                bg_color = CARD_BG
            else:
                # Régimen frío / estable (Cian)
                s_border = SERVER_NORMAL
                bg_color = CARD_BG

            # Caja del Servidor
            pygame.draw.rect(surface, bg_color, box_rect, border_radius=8)
            pygame.draw.rect(surface, s_border, box_rect, width=2 if is_emerg else 1, border_radius=8)

            # --- B. TERMÓMETRO ANALÓGICO VERTICAL (Con marca de Inflexión) ---
            t_meter_x = box_rect.x + 8
            t_meter_y = box_rect.y + 12
            t_meter_w = 7
            t_meter_h = 64

            # Fondo del tubo de mercurio
            pygame.draw.rect(surface, (15, 18, 24), (t_meter_x, t_meter_y, t_meter_w, t_meter_h), border_radius=3)
            pygame.draw.rect(surface, (60, 68, 80), (t_meter_x, t_meter_y, t_meter_w, t_meter_h), width=1, border_radius=3)

            # Altura del mercurio (escala de 20°C a 85°C)
            temp_ratio = max(0.0, min(1.0, (temp - 20.0) / (85.0 - 20.0)))
            fill_h = int(t_meter_h * temp_ratio)
            col_mercury = (248, 81, 73) if temp >= 48.04 else ((210, 153, 34) if temp >= 38.0 else (88, 166, 255))
            if fill_h > 0:
                pygame.draw.rect(
                    surface,
                    col_mercury,
                    (t_meter_x + 1, t_meter_y + t_meter_h - fill_h, t_meter_w - 2, fill_h),
                    border_radius=2
                )

            # Marca roja del Punto de Inflexión (48.04°C = ~43% de la altura)
            infl_ratio = (48.04 - 20.0) / (85.0 - 20.0)
            infl_y = t_meter_y + t_meter_h - int(t_meter_h * infl_ratio)
            pygame.draw.line(surface, (255, 60, 60), (t_meter_x - 3, infl_y), (t_meter_x + t_meter_w + 2, infl_y), 2)

            # --- C. VENTILADOR FÍSICO GIRATORIO ---
            fan_cx = box_rect.right - 22
            fan_cy = box_rect.y + 24
            fan_rad = 14

            # Carcasa circular del ventilador
            pygame.draw.circle(surface, (12, 16, 22), (fan_cx, fan_cy), fan_rad)
            pygame.draw.circle(surface, (70, 85, 105), (fan_cx, fan_cy), fan_rad, width=1)

            # Aspas rotatorias
            base_ang = self.fan_angles[s_cfg.id]
            for blade_i in range(4):
                ang = base_ang + blade_i * (math.pi / 2.0)
                bx = fan_cx + math.cos(ang) * (fan_rad - 2)
                by = fan_cy + math.sin(ang) * (fan_rad - 2)
                blade_col = (255, 230, 100) if is_emerg else (180, 200, 220)
                pygame.draw.line(surface, blade_col, (fan_cx, fan_cy), (bx, by), 2)
            pygame.draw.circle(surface, (210, 220, 235), (fan_cx, fan_cy), 3)

            # Etiqueta de potencia del ventilador debajo
            fan_lbl = self.fonts["small"].render(f"{fan_p:.0f}W", True, (140, 165, 190))
            surface.blit(fan_lbl, (fan_cx - fan_lbl.get_width() // 2, fan_cy + fan_rad + 2))

            # --- D. INFORMACIÓN Y CARGA DE SERVIDOR ---
            content_x = box_rect.x + 22
            # Nombre y estado
            s_title = self.fonts["mono_bold"].render(s_cfg.id, True, TEXT_PRIMARY)
            surface.blit(s_title, (content_x, box_rect.y + 6))

            # Barra de carga CPU/RAM
            bar_w = box_rect.width - 58
            bar_fill = int(bar_w * max(0.0, min(1.0, load)))
            pygame.draw.rect(surface, (20, 24, 30), (content_x, box_rect.y + 25, bar_w, 6), border_radius=3)
            load_col = (248, 81, 73) if load >= 0.60 else (88, 166, 255)
            pygame.draw.rect(surface, load_col, (content_x, box_rect.y + 25, bar_fill, 6), border_radius=3)

            # Texto de Temperatura y Estado Térmico
            t_txt = f"{temp:.1f}°C"
            if is_emerg:
                t_txt += " (CRÍTICO 60%!)"
            t_surf = self.fonts["small"].render(t_txt, True, col_mercury)
            surface.blit(t_surf, (content_x, box_rect.y + 36))

            # Texto Energía asignada por Lagrange
            e_txt = f"Lagrange: {p_alloc:.1f} kW"
            e_surf = self.fonts["small"].render(e_txt, True, (163, 113, 247))
            surface.blit(e_surf, (content_x, box_rect.y + 53))

            # Indicador de Carga numérica
            l_txt = f"Carga: {load * 100:.0f}%"
            l_surf = self.fonts["small"].render(l_txt, True, TEXT_MUTED)
            surface.blit(l_surf, (content_x, box_rect.y + 68))

        # 5. Banner de Alerta Visual de Cambio Dinámico de Ruta (DP)
        time_since_reroute = time.time() - self.last_path_change_time
        if time_since_reroute < 4.5 and active_path:
            banner_w = min(620, self.rect.width - 260)
            banner_rect = pygame.Rect(self.rect.x + 20, self.rect.y + self.rect.height - 40, banner_w, 28)
            pygame.draw.rect(surface, (40, 32, 10), banner_rect, border_radius=4)
            pygame.draw.rect(surface, (255, 215, 0), banner_rect, width=1, border_radius=4)
            msg = f">> RUTA OPTIMA DP ADAPTADA: {' -> '.join(active_path)}"
            msg_surf = self.fonts["mono_bold"].render(msg, True, (255, 225, 100))
            surface.blit(msg_surf, msg_surf.get_rect(center=banner_rect.center))
