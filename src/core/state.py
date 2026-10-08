"""
DTOs y contenedor de estado sincronizado entre el motor SimPy y el Dashboard Pygame.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from src.core.entities import Packet, RouterState, ServerState


@dataclass
class SimulationMetrics:
    """Métricas operativas acumuladas del centro de datos."""
    total_packets_created: int = 0
    total_packets_delivered: int = 0
    total_packets_dropped: int = 0
    average_latency: float = 0.0
    total_value_processed: float = 0.0
    total_tasks_scheduled: int = 0
    total_tasks_rejected: int = 0
    active_shortest_path: List[str] = field(default_factory=list)
    path_reroute_count: int = 0


@dataclass
class DataCenterSnapshot:
    """Instantánea inmutable del estado del datacenter para renderizado seguro en Pygame."""
    sim_time: float
    is_paused: bool
    routers: Dict[str, RouterState]
    servers: Dict[str, ServerState]
    active_packets: List[Packet]
    metrics: SimulationMetrics
    cooling_power_history: Dict[str, List[float]]  # Servidor -> Lista de potencias de ventilador
    cooling_cost_history: Dict[str, List[float]]
    lagrange_allocation: Dict[str, float]
    lagrange_lambda: float
    failed_nodes: set[str]
    alert_messages: List[str]
