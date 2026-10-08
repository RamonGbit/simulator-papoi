"""
Generador de tráfico estocástico para el centro de datos (Teoría de Colas y Procesos de Poisson).
Genera llegadas aleatorias con distribución exponencial para modelar colas M/M/c.
"""

from __future__ import annotations
import random
import simpy
from typing import Callable, List, Optional
from src.core.entities import Packet
from src.math_models.dp_knapsack import TaskItem


class TrafficGenerator:
    """
    Generador de paquetes según proceso de Poisson con tasa media lambda.
    Intervalo entre llegadas: Delta t ~ Exp(lambda)
    """

    def __init__(
        self,
        env: simpy.Environment,
        arrival_rate_lambda: float,
        ingress_node: str,
        target_servers: List[str],
        on_packet_generated: Callable[[Packet], None]
    ):
        self.env = env
        self.arrival_rate_lambda = max(0.1, float(arrival_rate_lambda))
        self.ingress_node = ingress_node
        self.target_servers = target_servers
        self.on_packet_generated = on_packet_generated
        self.is_running = True
        self.packet_counter = 0

        # Iniciar proceso en el entorno SimPy
        self.action = env.process(self._run_generator())

    def _generate_task_payload(self, packet_id: str) -> TaskItem:
        """Genera una tarea computacional con requerimientos y valor comercial sintético."""
        task_types = [
            ("AuthService", (2, 4), 30.0),
            ("DBQuery", (4, 8), 60.0),
            ("AnalyticsETL", (8, 16), 95.0),
            ("AIInference", (16, 32), 150.0),
            ("MediaStream", (6, 12), 70.0),
        ]
        name, (cpu_range, ram_range), base_val = random.choice(task_types)
        cpu = random.randint(max(1, cpu_range // 2), cpu_range)
        ram = random.randint(max(2, ram_range // 2), ram_range)
        val = base_val + random.uniform(-10.0, 15.0)

        return TaskItem(
            task_id=f"TASK_{packet_id}",
            name=name,
            cpu_required=cpu,
            ram_required=ram,
            value=round(val, 2)
        )

    def _run_generator(self):
        """Bucle generador de eventos estocásticos SimPy."""
        while self.is_running:
            # Tiempo inter-llegada exponencial: 1.0 / lambda
            inter_arrival = random.expovariate(self.arrival_rate_lambda)
            yield self.env.timeout(inter_arrival)

            if not self.target_servers:
                continue

            self.packet_counter += 1
            pkt_id = f"PKT_{self.packet_counter:04d}"
            target_server = random.choice(self.target_servers)
            payload = self._generate_task_payload(pkt_id)

            pkt = Packet(
                id=pkt_id,
                source_node=self.ingress_node,
                target_server=target_server,
                current_node=self.ingress_node,
                path=[],
                created_at=self.env.now,
                task_payload=payload
            )

            self.on_packet_generated(pkt)

    def stop(self) -> None:
        self.is_running = False
