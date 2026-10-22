"""
Motor central de simulación de eventos discretos (SimPy) para DataStream Networks.
Orquesta el tráfico de red, teoría de colas M/M/c, enrutamiento por Programación Dinámica,
planificación por Mochila DP, control térmico por Gradiente y presupuesto de Lagrange.
"""

from __future__ import annotations
import threading
import time
from typing import Dict, List, Optional, Any, Set
import simpy

from config.settings import AppConfig
from src.core.entities import Packet, RouterState, ServerState
from src.core.state import DataCenterSnapshot, SimulationMetrics
from src.math_models.dp_routing import DPRoutingOptimizer, RouteResult
from src.math_models.dp_knapsack import DPKnapsackScheduler, TaskItem, KnapsackResult
from src.math_models.thermal_analysis import ThermalAnalyzer1D, ThermalCriticalPoints
from src.math_models.cooling_gradient import GradientDescentCooling
from src.math_models.power_lagrange import LagrangePowerBudget, ClusterEnergySpec, LagrangeOptimizationResult
from src.simulation.traffic_generator import TrafficGenerator
from src.simulation.failure_model import StochasticFailureModel


class DataCenterSimulation:
    """
    Simulador integral del centro de datos coordinado por SimPy.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.lock = threading.Lock()
        self.is_paused = False
        self.is_running = True

        # 1. Entorno SimPy
        self.env = simpy.Environment()

        # 2. Inicialización de Nodos y Recursos SimPy
        self.routers: Dict[str, RouterState] = {}
        self.router_resources: Dict[str, simpy.Resource] = {}
        for r_cfg in config.routers:
            self.routers[r_cfg.id] = RouterState(
                id=r_cfg.id,
                name=r_cfg.name,
                buffer_capacity=r_cfg.buffer_capacity
            )
            # Router como recurso con capacidad de reenvío simultáneo
            self.router_resources[r_cfg.id] = simpy.Resource(self.env, capacity=4)

        self.servers: Dict[str, ServerState] = {}
        self.server_resources: Dict[str, simpy.Resource] = {}
        self.server_task_buffers: Dict[str, List[TaskItem]] = {}
        for s_cfg in config.servers:
            self.servers[s_cfg.id] = ServerState(
                id=s_cfg.id,
                name=s_cfg.name,
                cluster_id=s_cfg.cluster_id,
                cpu_capacity=s_cfg.cpu_capacity,
                ram_capacity=s_cfg.ram_capacity,
                temperature=s_cfg.ambient_temp
            )
            self.server_resources[s_cfg.id] = simpy.Resource(self.env, capacity=s_cfg.cpu_capacity // 4)
            self.server_task_buffers[s_cfg.id] = []

        # 3. Modelos Matemáticos de Optimización
        # 3.1 Enrutador DP
        links_data = [link.model_dump() for link in config.links]
        self.routing_optimizer = DPRoutingOptimizer(links_data)

        # 3.2 Planificadores Mochila DP por servidor
        self.knapsack_schedulers: Dict[str, DPKnapsackScheduler] = {}
        for s_cfg in config.servers:
            composite_cap = max(10, s_cfg.cpu_capacity // 2 + s_cfg.ram_capacity // 4)
            self.knapsack_schedulers[s_cfg.id] = DPKnapsackScheduler(capacity_limit=composite_cap)

        # 3.3 Analizadores Térmicos (1D Simbólico)
        self.thermal_analyzers: Dict[str, ThermalAnalyzer1D] = {}
        self.thermal_critical_points: Dict[str, ThermalCriticalPoints] = {}
        for s_cfg in config.servers:
            analyzer = ThermalAnalyzer1D(
                t_ambient=s_cfg.ambient_temp,
                c1=s_cfg.thermal_params.a,
                c2=s_cfg.thermal_params.b,
                c3=s_cfg.thermal_params.c
            )
            self.thermal_analyzers[s_cfg.id] = analyzer
            self.thermal_critical_points[s_cfg.id] = analyzer.analyze()

        # 3.4 Controladores de Refrigeración por Descenso de Gradiente
        self.coolers: Dict[str, GradientDescentCooling] = {}
        for s_cfg in config.servers:
            self.coolers[s_cfg.id] = GradientDescentCooling(
                k_fan=s_cfg.cooling_params.k_fan,
                k_penalty=s_cfg.cooling_params.k_penalty,
                target_temp=s_cfg.target_temp,
                min_power=s_cfg.cooling_params.min_fan_power,
                max_power=s_cfg.cooling_params.max_fan_power,
                learning_rate=config.simulation_defaults.gradient_learning_rate
            )

        # 3.5 Optimizador de Presupuesto Eléctrico de Lagrange
        cluster_specs = [
            ClusterEnergySpec(
                cluster_id=s_cfg.cluster_id,
                name=s_cfg.name,
                alpha=s_cfg.lagrange_params.weight * 4.0,
                beta=s_cfg.lagrange_params.efficiency
            )
            for s_cfg in config.servers
        ]
        self.lagrange_optimizer = LagrangePowerBudget(cluster_specs)
        self.current_lagrange_result: Optional[LagrangeOptimizationResult] = None
        self._update_lagrange_allocation()

        # 4. Estado de Tráfico y Métricas
        self.active_packets: List[Packet] = []
        self.metrics = SimulationMetrics()
        self.alert_messages: List[str] = []
        self.last_shortest_paths: Dict[str, List[str]] = {}

        # Historial para gráficas en tiempo real
        self.cooling_power_history: Dict[str, List[float]] = {s_id: [10.0] for s_id in self.servers}
        self.cooling_cost_history: Dict[str, List[float]] = {s_id: [0.0] for s_id in self.servers}
        self.thermal_stress_timer: Dict[str, float] = {s_id: 0.0 for s_id in self.servers}

        # 5. Generador de Tráfico y Fallas
        target_server_ids = list(self.servers.keys())
        self.traffic_generator = TrafficGenerator(
            env=self.env,
            arrival_rate_lambda=config.simulation_defaults.arrival_rate_lambda,
            ingress_node=config.routers[0].id,
            target_servers=target_server_ids,
            on_packet_generated=self._handle_new_packet
        )

        vulnerable_routers = [r.id for r in config.routers if r.id != config.routers[0].id]
        self.failure_model = StochasticFailureModel(
            env=self.env,
            target_nodes=vulnerable_routers,
            mtbf=config.simulation_defaults.failure_mean_time,
            mttr=config.simulation_defaults.repair_mean_time,
            on_node_failed=self._handle_node_failed,
            on_node_recovered=self._handle_node_recovered
        )

        # 6. Procesos periódicos en SimPy
        self.env.process(self._periodic_server_knapsack_processing())
        self.env.process(self._periodic_thermal_and_cooling_loop())
        self.env.process(self._periodic_lagrange_budget_update())

    def _update_lagrange_allocation(self) -> None:
        """Calcula y aplica la distribución óptima de energía según Lagrange."""
        try:
            self.current_lagrange_result = self.lagrange_optimizer.optimize_symbolic(
                total_budget=self.config.power_budget_total
            )
            for s in self.servers.values():
                if s.cluster_id in self.current_lagrange_result.cluster_allocations:
                    s.allocated_power = self.current_lagrange_result.cluster_allocations[s.cluster_id]
        except Exception as e:
            self.alert_messages.append(f"Error en Lagrange: {e}")

    def _handle_node_failed(self, node_id: str) -> None:
        with self.lock:
            if node_id in self.routers:
                self.routers[node_id].is_active = False
            self.alert_messages.append(f"ALERTA: Falla estocástica en nodo {node_id}")
            self.metrics.path_reroute_count += 1

    def _handle_node_recovered(self, node_id: str) -> None:
        with self.lock:
            if node_id in self.routers:
                self.routers[node_id].is_active = True
            self.alert_messages.append(f"INFO: Nodo {node_id} reparado y operativo")
            self.metrics.path_reroute_count += 1

    def _handle_new_packet(self, pkt: Packet) -> None:
        """Recibe un nuevo paquete generado e inicia su tránsito."""
        with self.lock:
            self.metrics.total_packets_created += 1

            # Calcular ruta más corta por DP considerando congestión y fallos
            q_lens = {r_id: r.queue_size for r_id, r in self.routers.items()}
            route_res = self.routing_optimizer.find_shortest_path(
                source=pkt.source_node,
                target=pkt.target_server,
                queue_lengths=q_lens,
                failed_nodes=self.failure_model.failed_nodes
            )

            if not route_res.is_reachable:
                pkt.is_dropped = True
                self.metrics.total_packets_dropped += 1
                return

            pkt.path = route_res.path
            pkt.path_index = 0
            self.active_packets.append(pkt)

            # Notificar cambio de ruta
            old_path = self.last_shortest_paths.get(pkt.target_server, [])
            if old_path and old_path != pkt.path:
                self.metrics.path_reroute_count += 1
                self.alert_messages.append(
                    f"REROUTE DP: {'->'.join(old_path)} ==> {'->'.join(pkt.path)}"
                )
            self.last_shortest_paths[pkt.target_server] = pkt.path
            self.metrics.active_shortest_path = pkt.path

        # Iniciar proceso de tránsito en SimPy
        self.env.process(self._packet_transit_process(pkt))

    def _packet_transit_process(self, pkt: Packet):
        """Modela el viaje del paquete a través de cada arista de la red."""
        while pkt.path_index < len(pkt.path) - 1:
            curr_node = pkt.path[pkt.path_index]
            next_node = pkt.path[pkt.path_index + 1]

            # Si el siguiente nodo está caído, re-enrutar dinámicamente con DP
            if next_node in self.failure_model.failed_nodes:
                with self.lock:
                    q_lens = {r_id: r.queue_size for r_id, r in self.routers.items()}
                    reroute = self.routing_optimizer.find_shortest_path(
                        source=curr_node,
                        target=pkt.target_server,
                        queue_lengths=q_lens,
                        failed_nodes=self.failure_model.failed_nodes
                    )
                    if not reroute.is_reachable:
                        pkt.is_dropped = True
                        self.metrics.total_packets_dropped += 1
                        if pkt in self.active_packets:
                            self.active_packets.remove(pkt)
                        return
                    pkt.path = [curr_node] + reroute.path[1:]
                    pkt.path_index = 0
                    next_node = pkt.path[1]

            # Buscar latencia del enlace
            edge_latency = 2.0  # ms
            for l in self.config.links:
                if l.source == curr_node and l.target == next_node:
                    edge_latency = l.base_latency + (l.congestion_factor * self.routers.get(next_node, RouterState(next_node, next_node)).queue_size)
                    break

            sim_delay = max(0.05, edge_latency / 10.0)

            # Aumentar cola en el router actual
            if curr_node in self.routers:
                with self.lock:
                    self.routers[curr_node].queue_size += 1

            # Simular propagación en el canal
            steps = 8
            for step_i in range(steps):
                yield self.env.timeout(sim_delay / steps)
                with self.lock:
                    pkt.progress = (step_i + 1) / steps

            if curr_node in self.routers:
                with self.lock:
                    self.routers[curr_node].queue_size = max(0, self.routers[curr_node].queue_size - 1)
                    self.routers[curr_node].packets_forwarded += 1

            pkt.path_index += 1
            pkt.current_node = next_node
            pkt.progress = 0.0

        # Llegada al servidor final
        with self.lock:
            pkt.delivered_at = self.env.now
            if pkt in self.active_packets:
                self.active_packets.remove(pkt)

            self.metrics.total_packets_delivered += 1
            latency = (pkt.delivered_at - pkt.created_at) * 10.0
            n_deliv = self.metrics.total_packets_delivered
            self.metrics.average_latency = ((self.metrics.average_latency * (n_deliv - 1)) + latency) / n_deliv

            if pkt.task_payload and pkt.target_server in self.server_task_buffers:
                self.server_task_buffers[pkt.target_server].append(pkt.task_payload)

    def _periodic_server_knapsack_processing(self):
        """Aplica periódicamente el Problema de la Mochila (DP) a los buffers de tareas en los servidores."""
        while self.is_running:
            yield self.env.timeout(1.5)  # Cada 1.5 unidades de tiempo simulado
            with self.lock:
                for s_id, buffer in list(self.server_task_buffers.items()):
                    if not buffer:
                        continue
                    scheduler = self.knapsack_schedulers[s_id]
                    result: KnapsackResult = scheduler.optimize_allocation(s_id, buffer)

                    # Actualizar métricas
                    self.metrics.total_value_processed += result.total_value
                    self.metrics.total_tasks_scheduled += len(result.selected_tasks)
                    self.metrics.total_tasks_rejected += len(result.rejected_tasks)

                    server = self.servers[s_id]
                    server.tasks_processed += len(result.selected_tasks)
                    server.tasks_rejected += len(result.rejected_tasks)
                    server.current_load = min(1.0, result.utilization_percentage / 100.0)

                    # Vaciar buffer procesado
                    self.server_task_buffers[s_id] = []

    def _periodic_thermal_and_cooling_loop(self):
        """Calcula el modelo no lineal T(L) y ajusta ventiladores con Gradiente Descendente."""
        while self.is_running:
            yield self.env.timeout(0.5)
            with self.lock:
                for s_id, server in self.servers.items():
                    analyzer = self.thermal_analyzers[s_id]
                    crit_pts = self.thermal_critical_points[s_id]
                    cooler = self.coolers[s_id]
                    # 1. Aplicar estrés térmico si está activo
                    effective_load = server.current_load
                    if self.thermal_stress_timer.get(s_id, 0.0) > 0.0:
                        self.thermal_stress_timer[s_id] = max(0.0, self.thermal_stress_timer[s_id] - 0.5)
                        effective_load = max(effective_load, 0.95)
                        server.current_load = effective_load

                    # Temperatura base no lineal producida por la carga
                    t_base = analyzer.evaluate_temp(effective_load)
                    if self.thermal_stress_timer.get(s_id, 0.0) > 0.0:
                        t_base += 18.0  # Impulso adicional de choque térmico

                    # 2. Paso de descenso de gradiente para ventilador
                    fan_p, cost, grad = cooler.step(t_base)
                    server.fan_power = fan_p
                    server.temperature = cooler.compute_temperature(t_base, fan_p)

                    # 3. Detectar si sobrepasa el punto crítico de inflexión para enfriamiento de emergencia
                    if server.current_load >= crit_pts.emergency_threshold_load or server.temperature >= crit_pts.emergency_threshold_temp:
                        server.is_emergency_cooling = True
                        if not any(f"EMERGENCIA TÉRMICA en {s_id}" in msg for msg in self.alert_messages[-3:]):
                            self.alert_messages.append(
                                f"EMERGENCIA TÉRMICA en {s_id}: Carga={server.current_load*100:.1f}%, Temp={server.temperature:.1f}°C"
                            )
                    else:
                        server.is_emergency_cooling = False

                    # Guardar historial para gráficas
                    self.cooling_power_history[s_id].append(fan_p)
                    self.cooling_cost_history[s_id].append(cost)
                    if len(self.cooling_power_history[s_id]) > 60:
                        self.cooling_power_history[s_id].pop(0)
                        self.cooling_cost_history[s_id].pop(0)

    def _periodic_lagrange_budget_update(self):
        """Actualiza periódicamente la optimización de Lagrange."""
        while self.is_running:
            yield self.env.timeout(5.0)
            with self.lock:
                self._update_lagrange_allocation()

    def advance_time(self, dt: float) -> None:
        """Avanza la simulación SimPy en pequeños incrementos de tiempo."""
        if not self.is_paused and self.is_running:
            try:
                self.env.run(until=self.env.now + dt)
            except Exception as e:
                with self.lock:
                    self.alert_messages.append(f"Error SimPy: {e}")

    def toggle_pause(self) -> bool:
        with self.lock:
            self.is_paused = not self.is_paused
            return self.is_paused

    def adjust_power_budget(self, delta: float) -> float:
        """Ajusta dinámicamente el presupuesto eléctrico total de Lagrange."""
        with self.lock:
            new_val = max(50.0, min(1000.0, self.config.power_budget_total + delta))
            self.config.power_budget_total = round(new_val, 1)
            self._update_lagrange_allocation()
            self.alert_messages.append(f"PRESUPUESTO LAGRANGE: {self.config.power_budget_total:.1f} kW (lambda*={self.current_lagrange_result.lambda_multiplier:.4f})")
            return self.config.power_budget_total

    def adjust_arrival_rate(self, delta: float) -> float:
        """Ajusta dinámicamente la tasa de llegada de tráfico Poisson (lambda)."""
        with self.lock:
            new_rate = max(1.0, min(80.0, self.config.simulation_defaults.arrival_rate_lambda + delta))
            self.config.simulation_defaults.arrival_rate_lambda = round(new_rate, 1)
            self.traffic_generator.arrival_rate_lambda = new_rate
            self.alert_messages.append(f"TRAFICO POISSON lambda: {new_rate:.1f} pkt/s")
            return new_rate

    def adjust_target_temperature(self, delta: float) -> float:
        """Ajusta la temperatura objetivo del sistema de refrigeración por gradiente."""
        with self.lock:
            new_temp = max(35.0, min(80.0, self.config.servers[0].target_temp + delta))
            new_temp = round(new_temp, 1)
            for s in self.config.servers:
                s.target_temp = new_temp
            for cooler in self.coolers.values():
                cooler.target_temp = new_temp
            self.alert_messages.append(f"OBJETIVO TERMICO REFRIGERACION: {new_temp:.1f} C")
            return new_temp

    def trigger_thermal_stress(self, server_id: Optional[str] = None, duration: float = 6.0) -> str:
        """Inyecta un choque térmico brusco para demostrar en vivo el régimen superlineal y el gradiente."""
        with self.lock:
            if server_id is None or server_id not in self.servers:
                server_id = list(self.servers.keys())[0]
            self.thermal_stress_timer[server_id] = duration
            server = self.servers[server_id]
            server.current_load = 0.95
            self.alert_messages.append(f"TEST DE ESTRÉS TÉRMICO en {server_id} (+95% carga / +18°C)")
            return server_id

    def get_snapshot(self) -> DataCenterSnapshot:
        """Retorna una instantánea segura del estado para renderizar en Pygame."""
        with self.lock:
            routers_copy = {k: RouterState(**vars(v)) for k, v in self.routers.items()}
            servers_copy = {k: ServerState(**vars(v)) for k, v in self.servers.items()}
            packets_copy = [Packet(**vars(p)) for p in self.active_packets]
            metrics_copy = SimulationMetrics(**vars(self.metrics))

            alloc_copy = (
                self.current_lagrange_result.cluster_allocations.copy()
                if self.current_lagrange_result else {}
            )
            lam_val = (
                self.current_lagrange_result.lambda_multiplier
                if self.current_lagrange_result else 0.0
            )

            alerts = list(self.alert_messages[-8:])

            return DataCenterSnapshot(
                sim_time=round(self.env.now, 2),
                is_paused=self.is_paused,
                routers=routers_copy,
                servers=servers_copy,
                active_packets=packets_copy,
                metrics=metrics_copy,
                cooling_power_history={k: list(v) for k, v in self.cooling_power_history.items()},
                cooling_cost_history={k: list(v) for k, v in self.cooling_cost_history.items()},
                lagrange_allocation=alloc_copy,
                lagrange_lambda=lam_val,
                failed_nodes=set(self.failure_model.failed_nodes),
                alert_messages=alerts
            )
