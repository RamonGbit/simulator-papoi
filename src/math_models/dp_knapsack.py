"""
Módulo 1.2: Modelo de Volumen-Carga (Problema de la Mochila mediante Programación Dinámica).
Optimiza la asignación de tareas entrantes a servidores computacionales terminales,
maximizando el valor total (prioridad/SLA) sin exceder la capacidad finita de RAM/CPU.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple


@dataclass
class TaskItem:
    """Representa una tarea computacional o carga entrante."""
    task_id: str
    name: str
    cpu_required: int
    ram_required: int
    value: float  # Recompensa económica, SLA o nivel de prioridad

    @property
    def composite_weight(self) -> int:
        """
        Calcula un peso compuesto entero normalizado para la mochila.
        Pondera CPU (unidades) y RAM (GB) de forma equilibrada.
        """
        return max(1, int(self.cpu_required + (self.ram_required // 2)))


@dataclass
class KnapsackResult:
    """Resultado de la optimización del modelo volumen-carga."""
    server_id: str
    capacity_limit: int
    total_value: float
    total_weight_used: int
    selected_tasks: List[TaskItem]
    rejected_tasks: List[TaskItem]
    utilization_percentage: float
    dp_matrix_shape: Tuple[int, int]


class DPKnapsackScheduler:
    """
    Optimizador de Asignación Servidor-Carga basado en Programación Dinámica (Mochila 0/1).
    
    Ecuación de Recurrencia de Bellman:
        DP[i][w] = DP[i-1][w]                                    si w_i > w
        DP[i][w] = max( DP[i-1][w], DP[i-1][w - w_i] + v_i )    si w_i <= w
    """

    def __init__(self, capacity_limit: int):
        """
        Args:
            capacity_limit: Capacidad máxima entera disponible en el servidor (RAM/CPU ponderado).
        """
        if capacity_limit <= 0:
            raise ValueError("La capacidad del servidor debe ser estrictamente positiva.")
        self.capacity_limit = capacity_limit

    def optimize_allocation(self, server_id: str, tasks: List[TaskItem]) -> KnapsackResult:
        """
        Ejecuta la programación dinámica y reconstruye el subconjunto óptimo de tareas.
        
        Args:
            server_id: Identificador del servidor.
            tasks: Lista de tareas candidatas esperando procesamiento.
            
        Returns:
            KnapsackResult con tareas seleccionadas, valor acumulado y métricas de utilización.
        """
        if not tasks:
            return KnapsackResult(
                server_id=server_id,
                capacity_limit=self.capacity_limit,
                total_value=0.0,
                total_weight_used=0,
                selected_tasks=[],
                rejected_tasks=[],
                utilization_percentage=0.0,
                dp_matrix_shape=(0, self.capacity_limit + 1)
            )

        n = len(tasks)
        W = self.capacity_limit

        # Matriz DP de dimensiones (n + 1) x (W + 1)
        # dp[i][w] = valor óptimo considerando las primeras i tareas y capacidad w
        dp = [[0.0 for _ in range(W + 1)] for _ in range(n + 1)]

        for i in range(1, n + 1):
            task = tasks[i - 1]
            wt = task.composite_weight
            val = task.value

            for w in range(W + 1):
                if wt <= w:
                    dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - wt] + val)
                else:
                    dp[i][w] = dp[i - 1][w]

        # Reconstrucción del subconjunto óptimo de tareas mediante retroceso (backtracking)
        selected_tasks: List[TaskItem] = []
        w_curr = W
        for i in range(n, 0, -1):
            # Si el valor difiere del nivel anterior, la tarea fue seleccionada
            if abs(dp[i][w_curr] - dp[i - 1][w_curr]) > 1e-6:
                task = tasks[i - 1]
                selected_tasks.append(task)
                w_curr -= task.composite_weight

        selected_tasks.reverse()
        selected_ids = {t.task_id for t in selected_tasks}
        rejected_tasks = [t for t in tasks if t.task_id not in selected_ids]

        total_weight_used = sum(t.composite_weight for t in selected_tasks)
        total_value = dp[n][W]
        utilization = (total_weight_used / self.capacity_limit) * 100.0 if self.capacity_limit > 0 else 0.0

        return KnapsackResult(
            server_id=server_id,
            capacity_limit=self.capacity_limit,
            total_value=round(total_value, 2),
            total_weight_used=total_weight_used,
            selected_tasks=selected_tasks,
            rejected_tasks=rejected_tasks,
            utilization_percentage=round(utilization, 2),
            dp_matrix_shape=(n + 1, W + 1)
        )


if __name__ == "__main__":
    # Demostración independiente del Modelo Volumen-Carga
    print("=== Prueba Independiente: Problema de la Mochila (DP) ===")
    capacidad_servidor = 25  # Unidades compuestas de cómputo (CPU + RAM/2)
    scheduler = DPKnapsackScheduler(capacity_limit=capacidad_servidor)

    sample_tasks = [
        TaskItem("T1", "Microservicio Auth", cpu_required=4, ram_required=4, value=40.0),    # wt = 4 + 2 = 6
        TaskItem("T2", "Motor de Indexación", cpu_required=8, ram_required=8, value=75.0),   # wt = 8 + 4 = 12
        TaskItem("T3", "Ingesta de Telemetría", cpu_required=2, ram_required=2, value=25.0), # wt = 2 + 1 = 3
        TaskItem("T4", "Inferencia Neuronal", cpu_required=12, ram_required=16, value=110.0), # wt = 12 + 8 = 20
        TaskItem("T5", "Backup de Logs", cpu_required=4, ram_required=8, value=15.0),        # wt = 4 + 4 = 8
    ]

    resultado = scheduler.optimize_allocation("S_WEB_01", sample_tasks)
    print(f"Capacidad del Servidor: {resultado.capacity_limit} unidades")
    print(f"Valor Máximo Obtenido: {resultado.total_value} pts")
    print(f"Capacidad Utilizada: {resultado.total_weight_used}/{resultado.capacity_limit} ({resultado.utilization_percentage}%)")
    print("Tareas Aceptadas:")
    for t in resultado.selected_tasks:
        print(f"  [+] {t.task_id} ({t.name}): Peso={t.composite_weight}, Valor={t.value}")
    print("Tareas Pospuestas/Rechazadas:")
    for t in resultado.rejected_tasks:
        print(f"  [-] {t.task_id} ({t.name}): Peso={t.composite_weight}, Valor={t.value}")
