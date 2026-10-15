"""
Tema visual y paleta de diseño Cyber-NOC para el Dashboard Pygame.
"""

from __future__ import annotations
import pygame

# Dimensiones de pantalla recomendadas (Ampliada para alta visibilidad 1080p)
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 940
FPS = 60

# Paleta de colores Dark-NOC
BG_DARK = (13, 17, 23)
PANEL_BG = (22, 27, 34)
CARD_BG = (33, 38, 45)
BORDER_COLOR = (48, 54, 61)
BORDER_ACTIVE = (88, 166, 255)

TEXT_PRIMARY = (240, 246, 252)
TEXT_SECONDARY = (139, 148, 158)
TEXT_MUTED = (110, 118, 129)

# Nodos y Estado
ROUTER_ACTIVE = (88, 166, 255)       # Azul cian
ROUTER_FAILED = (248, 81, 73)        # Rojo carmesí
SERVER_NORMAL = (46, 160, 67)        # Verde esmeralda
SERVER_WARNING = (210, 153, 34)      # Amarillo advertencia
SERVER_EMERGENCY = (248, 81, 73)     # Rojo emergencia

# Enlaces y Rutas
LINK_IDLE = (48, 54, 61)             # Gris oscuro
LINK_ACTIVE_PATH = (255, 215, 0)     # Dorado brillante (Ruta DP Óptima)
LINK_GLOW = (255, 235, 100)

# Paquetes de datos
PACKET_COLOR = (56, 139, 253)        # Azul brillante
PACKET_GLOW = (121, 192, 255)

# Gráficas
CHART_LINE_GRADIENT = (219, 97, 162) # Rosa magenta
CHART_LINE_COST = (255, 166, 87)     # Naranja
CHART_GRID = (40, 46, 54)
CHART_AXIS = (110, 118, 129)

# Clústeres Lagrange
CLUSTER_COLORS = {
    "cluster_web": (88, 166, 255),   # Azul
    "cluster_db": (63, 185, 80),     # Verde
    "cluster_ai": (163, 113, 247),   # Púrpura
}

# Inicializador de fuentes con tipografía escalada y de alta legibilidad
def get_fonts():
    pygame.font.init()
    sans_stack = "dejavusans,cantarell,liberationsans,segoeui,arial,helvetica"
    mono_stack = "dejavusansmono,hack,jetbrainsmononf,consolas,monospace"
    try:
        title_font = pygame.font.SysFont(sans_stack, 20, bold=True)
        header_font = pygame.font.SysFont(sans_stack, 15, bold=True)
        var_font = pygame.font.SysFont(sans_stack, 14, bold=True)
        regular_font = pygame.font.SysFont(sans_stack, 13)
        small_font = pygame.font.SysFont(sans_stack, 11)
        mono_font = pygame.font.SysFont(mono_stack, 12)
        mono_bold = pygame.font.SysFont(mono_stack, 12, bold=True)
    except Exception:
        title_font = pygame.font.Font(None, 26)
        header_font = pygame.font.Font(None, 20)
        var_font = pygame.font.Font(None, 19)
        regular_font = pygame.font.Font(None, 17)
        small_font = pygame.font.Font(None, 14)
        mono_font = pygame.font.Font(None, 15)
        mono_bold = pygame.font.Font(None, 15)

    return {
        "title": title_font,
        "header": header_font,
        "var_label": var_font,
        "regular": regular_font,
        "small": small_font,
        "mono": mono_font,
        "mono_bold": mono_bold
    }
