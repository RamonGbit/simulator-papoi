"""
Modelo estocástico de fallas y recuperación de nodos en SimPy.
Simula tiempos medios entre fallas (MTBF) y tiempos de reparación (MTTR) con distribuciones exponenciales.
"""

from __future__ import annotations
import random
import simpy
from typing import Callable, List, Set


class StochasticFailureModel:
    """
    Gestiona los eventos estocásticos de caída y restablecimiento de routers.
    """

    def __init__(
        self,
        env: simpy.Environment,
        target_nodes: List[str],
        mtbf: float = 45.0,
        mttr: float = 8.0,
        on_node_failed: Callable[[str], None] = None,
        on_node_recovered: Callable[[str], None] = None
    ):
        self.env = env
        self.target_nodes = target_nodes
        self.mtbf = max(5.0, float(mtbf))
        self.mttr = max(1.0, float(mttr))
        self.on_node_failed = on_node_failed
        self.on_node_recovered = on_node_recovered
        self.failed_nodes: Set[str] = set()
        self.is_running = True

        # Iniciar un proceso de falla independiente por cada nodo vulnerable
        self.processes = [env.process(self._node_lifecycle(node)) for node in self.target_nodes]

    def _node_lifecycle(self, node_id: str):
        """Ciclo de vida estocástico de un nodo: Operativo -> Falla -> Reparación -> Operativo."""
        while self.is_running:
            # Tiempo hasta la próxima falla (exponencial)
            time_to_fail = random.expovariate(1.0 / self.mtbf)
            yield self.env.timeout(time_to_fail)

            # Ocurre la falla
            self.failed_nodes.add(node_id)
            if self.on_node_failed:
                self.on_node_failed(node_id)

            # Tiempo de reparación (exponencial)
            time_to_repair = random.expovariate(1.0 / self.mttr)
            yield self.env.timeout(time_to_repair)

            # Se recupera el nodo
            self.failed_nodes.discard(node_id)
            if self.on_node_recovered:
                self.on_node_recovered(node_id)

    def trigger_manual_failure(self, node_id: str) -> None:
        """Permite inyectar una falla manual desde la interfaz de Pygame para pruebas."""
        if node_id in self.target_nodes and node_id not in self.failed_nodes:
            self.failed_nodes.add(node_id)
            if self.on_node_failed:
                self.on_node_failed(node_id)

    def trigger_manual_recovery(self, node_id: str) -> None:
        """Permite recuperar manualmente un nodo caído desde la interfaz."""
        if node_id in self.failed_nodes:
            self.failed_nodes.discard(node_id)
            if self.on_node_recovered:
                self.on_node_recovered(node_id)

    def stop(self) -> None:
        self.is_running = False
