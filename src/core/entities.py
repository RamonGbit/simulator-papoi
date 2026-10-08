"""
Entidades de dominio del Centro de Datos (DataStream Networks).
Modela paquetes de datos, tareas computacionales, estado de routers y servidores.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import time
from src.math_models.dp_knapsack import TaskItem


@dataclass
class Packet:
    """Modela un paquete de datos viajando por la red."""
    id: str
    source_node: str
    target_server: str
    current_node: str
    path: List[str]
    path_index: int = 0
    created_at: float = 0.0
    delivered_at: Optional[float] = None
    task_payload: Optional[TaskItem] = None
    progress: float = 0.0  # Progreso en el enlace actual [0.0, 1.0] para animación visual
    is_dropped: bool = False

    @property
    def next_node(self) -> Optional[str]:
        if self.path_index + 1 < len(self.path):
            return self.path[self.path_index + 1]
        return None


@dataclass
class RouterState:
    """Estado en tiempo real de un nodo de enrutamiento."""
    id: str
    name: str
    is_active: bool = True
    queue_size: int = 0
    buffer_capacity: int = 50
    packets_forwarded: int = 0
    packets_dropped: int = 0
    failure_countdown: float = 0.0


@dataclass
class ServerState:
    """Estado en tiempo real de un servidor de cómputo terminal."""
    id: str
    name: str
    cluster_id: str
    cpu_capacity: int
    ram_capacity: int
    current_load: float = 0.0          # Factor de carga [0.0, 1.0]
    temperature: float = 22.0          # Temperatura actual en °C
    fan_power: float = 10.0            # Potencia de enfriamiento actual en W
    allocated_power: float = 50.0      # Presupuesto eléctrico asignado por Lagrange (kW)
    is_emergency_cooling: bool = False # Flag si supera el punto de inflexión crítico
    tasks_processed: int = 0
    tasks_rejected: int = 0
    active_tasks: List[TaskItem] = field(default_factory=list)
