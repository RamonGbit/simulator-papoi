"""
Módulo 1.1: Determinación de la Ruta Más Corta mediante Programación Dinámica.
Implementa el algoritmo de optimización multi-etapa de Bellman-Ford con costos dinámicos
sensibles a la congestión de colas de paquetes en tiempo real.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any


@dataclass
class RouteResult:
    """Resultado del cálculo de la ruta óptima por Programación Dinámica."""
    source: str
    target: str
    path: List[str]
    total_cost: float
    stages_evaluated: int
    dp_table: Dict[int, Dict[str, float]] = field(default_factory=dict)
    is_reachable: bool = True


class DPRoutingOptimizer:
    """
    Optimizador de enrutamiento basado en el Principio de Optimalidad de Bellman (Programación Dinámica).
    
    Ecuación de Recurrencia:
        V_k(u) = min_{v in Vecinos(u)} [ Costo(u, v, t) + V_{k-1}(v) ]
    donde:
        Costo(u, v, t) = latencia_base(u, v) + factor_congestion * cola(v, t)
    """

    def __init__(self, links_data: List[Dict[str, Any]]):
        """
        Inicializa el optimizador con la topología de enlaces.
        
        Args:
            links_data: Lista de diccionarios con keys 'source', 'target', 'base_latency',
                        'bandwidth', 'congestion_factor'.
        """
        self.links = links_data
        self.nodes = sorted(list(self._extract_unique_nodes()))
        self.adjacency: Dict[str, List[Dict[str, Any]]] = {node: [] for node in self.nodes}
        self._build_adjacency()

    def _extract_unique_nodes(self) -> set[str]:
        nodes = set()
        for link in self.links:
            nodes.add(link["source"])
            nodes.add(link["target"])
        return nodes

    def _build_adjacency(self) -> None:
        """Construye lista de adyacencia dirigida."""
        for link in self.links:
            src = link["source"]
            self.adjacency[src].append(link)

    def calculate_link_cost(
        self,
        link: Dict[str, Any],
        queue_lengths: Optional[Dict[str, int]] = None,
        failed_nodes: Optional[set[str]] = None
    ) -> float:
        """
        Calcula el costo dinámico de un enlace considerando congestión y fallos.
        """
        target = link["target"]
        if failed_nodes and (target in failed_nodes or link["source"] in failed_nodes):
            return float("inf")  # Nodo inactivo

        base_lat = float(link.get("base_latency", 1.0))
        cong_factor = float(link.get("congestion_factor", 0.1))
        queue_len = 0
        if queue_lengths and target in queue_lengths:
            queue_len = max(0, queue_lengths[target])

        # Costo = latencia base + penalización por retraso de cola
        return base_lat + (cong_factor * queue_len)

    def find_shortest_path(
        self,
        source: str,
        target: str,
        queue_lengths: Optional[Dict[str, int]] = None,
        failed_nodes: Optional[set[str]] = None
    ) -> RouteResult:
        """
        Halla el camino de menor latencia entre source y target aplicando Programación Dinámica por etapas.
        
        Args:
            source: Nodo origen (ej. 'R_IN')
            target: Nodo destino (ej. 'S_WEB')
            queue_lengths: Diccionario con la cantidad actual de paquetes en la cola de cada nodo
            failed_nodes: Conjunto de IDs de nodos caídos por fallas estocásticas
            
        Returns:
            RouteResult con la ruta óptima, costo acumulado y la tabla de DP generada.
        """
        if failed_nodes is None:
            failed_nodes = set()
        if queue_lengths is None:
            queue_lengths = {}

        if source in failed_nodes or target in failed_nodes:
            return RouteResult(source, target, [], float("inf"), 0, {}, is_reachable=False)

        num_nodes = len(self.nodes)
        max_stages = num_nodes - 1

        # dp[k][u] almacena la distancia mínima desde u hasta target usando a lo sumo k aristas
        dp: Dict[int, Dict[str, float]] = {}
        successor: Dict[int, Dict[str, Optional[str]]] = {}

        # Inicialización etapa 0 (0 aristas hasta target)
        dp[0] = {node: float("inf") for node in self.nodes}
        dp[0][target] = 0.0
        successor[0] = {node: None for node in self.nodes}

        # Iteración por etapas de Programación Dinámica (Bellman)
        stages_evaluated = 0
        for k in range(1, max_stages + 1):
            stages_evaluated = k
            dp[k] = dp[k - 1].copy()
            successor[k] = successor[k - 1].copy()
            changed = False

            # Para cada nodo u, evaluamos relajar a través de vecinos v
            for u in self.nodes:
                if u in failed_nodes:
                    continue

                for link in self.adjacency[u]:
                    v = link["target"]
                    if v in failed_nodes:
                        continue

                    cost_uv = self.calculate_link_cost(link, queue_lengths, failed_nodes)
                    if cost_uv == float("inf"):
                        continue

                    candidate_cost = cost_uv + dp[k - 1][v]
                    if candidate_cost < dp[k][u] - 1e-9:
                        dp[k][u] = candidate_cost
                        successor[k][u] = v
                        changed = True

            # Si no hubo cambios en esta etapa, el principio de optimalidad converge tempranamente
            if not changed:
                break

        final_stage = stages_evaluated
        total_cost = dp[final_stage].get(source, float("inf"))

        if total_cost == float("inf"):
            return RouteResult(source, target, [], float("inf"), stages_evaluated, dp, is_reachable=False)

        # Reconstrucción de la ruta óptima
        path = [source]
        curr = source
        visited = {source}
        curr_stage = final_stage

        while curr != target and curr_stage > 0:
            next_node = successor[curr_stage].get(curr)
            if next_node is None or next_node in visited:
                break  # Evita ciclos
            path.append(next_node)
            visited.add(next_node)
            curr = next_node
            curr_stage -= 1

        if path[-1] != target:
            # Si la reconstrucción no llegó al destino debido al etiquetado por etapas, intentamos reconstruir vorazmente con dp final
            path = self._reconstruct_greedy_from_dp(source, target, dp[final_stage], queue_lengths, failed_nodes)

        return RouteResult(
            source=source,
            target=target,
            path=path,
            total_cost=total_cost,
            stages_evaluated=stages_evaluated,
            dp_table=dp,
            is_reachable=(len(path) > 1 and path[-1] == target)
        )

    def _reconstruct_greedy_from_dp(
        self,
        source: str,
        target: str,
        final_costs: Dict[str, float],
        queue_lengths: Dict[str, int],
        failed_nodes: set[str]
    ) -> List[str]:
        path = [source]
        curr = source
        visited = {source}
        while curr != target:
            best_neighbor = None
            best_val = float("inf")
            for link in self.adjacency[curr]:
                v = link["target"]
                if v not in visited and v not in failed_nodes:
                    cost_uv = self.calculate_link_cost(link, queue_lengths, failed_nodes)
                    tot = cost_uv + final_costs.get(v, float("inf"))
                    if tot < best_val:
                        best_val = tot
                        best_neighbor = v
            if best_neighbor is None:
                break
            path.append(best_neighbor)
            visited.add(best_neighbor)
            curr = best_neighbor
        return path


if __name__ == "__main__":
    # Prueba independiente del módulo
    sample_links = [
        {"source": "R_IN", "target": "R1", "base_latency": 3.0, "congestion_factor": 0.2},
        {"source": "R_IN", "target": "R2", "base_latency": 5.0, "congestion_factor": 0.1},
        {"source": "R1", "target": "R3", "base_latency": 4.0, "congestion_factor": 0.15},
        {"source": "R2", "target": "R3", "base_latency": 2.0, "congestion_factor": 0.15},
        {"source": "R3", "target": "S_WEB", "base_latency": 1.0, "congestion_factor": 0.05},
    ]
    optimizer = DPRoutingOptimizer(sample_links)
    
    # Caso 1: Red sin congestión
    res_normal = optimizer.find_shortest_path("R_IN", "S_WEB")
    print(f"[Caso Normal] Ruta óptima: {' -> '.join(res_normal.path)} | Latencia: {res_normal.total_cost:.2f} ms")

    # Caso 2: R1 congestionado (15 paquetes en cola)
    res_congested = optimizer.find_shortest_path("R_IN", "S_WEB", queue_lengths={"R1": 15})
    print(f"[Caso Congestionado R1=15] Ruta óptima: {' -> '.join(res_congested.path)} | Latencia: {res_congested.total_cost:.2f} ms")
