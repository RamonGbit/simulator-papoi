"""
Componente visual de Botón interactivo para Pygame con hover y callback.
"""

from __future__ import annotations
import pygame
from typing import Callable, Optional, Tuple
from src.ui.theme import CARD_BG, BORDER_COLOR, BORDER_ACTIVE, TEXT_PRIMARY, TEXT_MUTED


class UIActionButton:
    """Botón con estados idle, hover y callback al hacer clic."""

    def __init__(
        self,
        rect: Tuple[int, int, int, int],
        text: str,
        font: pygame.font.Font,
        callback: Callable[[], None],
        bg_color: Tuple[int, int, int] = CARD_BG,
        text_color: Tuple[int, int, int] = TEXT_PRIMARY
    ):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.callback = callback
        self.bg_color = bg_color
        self.text_color = text_color
        self.is_hovered = False
        self.is_enabled = True

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.is_enabled:
            return False

        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()
                return True
        return False

    def draw(self, surface: pygame.Surface) -> None:
        # Fondo
        color = (min(255, self.bg_color[0] + 25), min(255, self.bg_color[1] + 25), min(255, self.bg_color[2] + 25)) if self.is_hovered else self.bg_color
        border = BORDER_ACTIVE if self.is_hovered else BORDER_COLOR
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, border, self.rect, width=1, border_radius=6)

        # Texto centrado
        txt_surf = self.font.render(self.text, True, self.text_color if self.is_enabled else TEXT_MUTED)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        surface.blit(txt_surf, txt_rect)
