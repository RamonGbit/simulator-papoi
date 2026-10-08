"""
Gestor de configuraciones y validaciones estrictas del sistema DataStream Networks.
Garantiza que todos los datos numéricos y topológicos respeten restricciones físicas y lógicas.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class ThermalParams(BaseModel):
    """Parámetros para la función no lineal de temperatura: T(L) = T_amb + a*L^2 + b*L^3 - c*L."""
    a: float = Field(..., gt=0, description="Coeficiente cuadrático térmico")
    b: float = Field(..., gt=0, description="Coeficiente cúbico térmico")
    c: float = Field(..., ge=0, description="Coeficiente lineal de disipación activa")


class CoolingParams(BaseModel):
    """Parámetros para el modelo de refrigeración y optimización por gradiente."""
    k_fan: float = Field(..., gt=0, description="Ponderación del coste energético del ventilador")
    k_penalty: float = Field(..., gt=0, description="Penalización cuadrática por exceso de temperatura")
    min_fan_power: float = Field(0.0, ge=0.0, description="Potencia mínima de ventilación")
    max_fan_power: float = Field(100.0, gt=0.0, description="Potencia máxima de ventilación")

    @model_validator(mode="after")
    def validate_power_range(self) -> CoolingParams:
        if self.min_fan_power >= self.max_fan_power:
            raise ValueError("min_fan_power debe ser estrictamente menor que max_fan_power")
        return self


class LagrangeParams(BaseModel):
    """Parámetros de rendimiento para asignación presupuestaria multivariable."""
    weight: float = Field(..., gt=0, description="Peso de prioridad de rendimiento (omega)")
    efficiency: float = Field(..., gt=0, description="Factor de ganancia logarítmica (beta)")


class RouterConfig(BaseModel):
    """Especificación de un Router dentro de la topología."""
    id: str = Field(..., min_length=1, description="Identificador único del router")
    name: str = Field(..., min_length=1, description="Nombre legible del router")
    buffer_capacity: int = Field(50, gt=0, description="Capacidad máxima de cola de paquetes")
    pos: Tuple[int, int] = Field(..., description="Coordenadas (x, y) para renderizado en Pygame")
    failure_probability: float = Field(0.005, ge=0.0, le=1.0, description="Probabilidad de falla estocástica")


class ServerConfig(BaseModel):
    """Especificación de un Servidor de cómputo terminal."""
    id: str = Field(..., min_length=1, description="Identificador único del servidor")
    name: str = Field(..., min_length=1, description="Nombre del servidor o clúster")
    cluster_id: str = Field(..., min_length=1, description="Clúster al que pertenece")
    connected_router: str = Field(..., min_length=1, description="ID del router al que está conectado")
    cpu_capacity: int = Field(..., gt=0, description="Capacidad total de núcleos CPU")
    ram_capacity: int = Field(..., gt=0, description="Capacidad total de memoria RAM (GB)")
    pos: Tuple[int, int] = Field(..., description="Coordenadas (x, y) para renderizado en Pygame")
    ambient_temp: float = Field(22.0, ge=10.0, le=45.0, description="Temperatura ambiente base (°C)")
    target_temp: float = Field(60.0, gt=20.0, description="Temperatura objetivo (°C)")
    emergency_temp: float = Field(85.0, gt=40.0, description="Umbral crítico de enfriamiento de emergencia (°C)")
    thermal_params: ThermalParams
    cooling_params: CoolingParams
    lagrange_params: LagrangeParams

    @model_validator(mode="after")
    def validate_temperatures(self) -> ServerConfig:
        if self.target_temp <= self.ambient_temp:
            raise ValueError(f"target_temp ({self.target_temp}) debe ser mayor que ambient_temp ({self.ambient_temp})")
        if self.emergency_temp <= self.target_temp:
            raise ValueError(f"emergency_temp ({self.emergency_temp}) debe ser mayor que target_temp ({self.target_temp})")
        return self


class LinkConfig(BaseModel):
    """Especificación de un enlace de red bidireccional o unidireccional."""
    source: str = Field(..., min_length=1, description="Nodo origen")
    target: str = Field(..., min_length=1, description="Nodo destino")
    base_latency: float = Field(..., gt=0.0, description="Latencia base de propagación en milisegundos")
    bandwidth: float = Field(..., gt=0.0, description="Ancho de banda nominal en Gbps")
    congestion_factor: float = Field(0.1, ge=0.0, description="Factor de retardo por cada paquete en cola")


class SimulationDefaults(BaseModel):
    """Parámetros predeterminados del motor de eventos SimPy y optimizadores."""
    arrival_rate_lambda: float = Field(12.0, gt=0.0, description="Tasa media de llegada de paquetes (Poisson)")
    simulation_duration: float = Field(120.0, gt=0.0, description="Duración estándar en segundos simulados")
    failure_mean_time: float = Field(40.0, gt=0.0, description="Tiempo medio entre fallas (MTBF)")
    repair_mean_time: float = Field(8.0, gt=0.0, description="Tiempo medio de reparación (MTTR)")
    gradient_learning_rate: float = Field(0.08, gt=0.0, le=1.0, description="Tasa de aprendizaje alpha")
    gradient_iterations: int = Field(60, gt=0, le=1000, description="Iteraciones máximas del gradiente")


class AppConfig(BaseModel):
    """Modelo raíz de configuración del centro de datos."""
    datacenter_name: str = Field("DataStream Networks Datacenter", min_length=1)
    version: str = Field("1.0.0")
    power_budget_total: float = Field(..., gt=0.0, description="Límite máximo de presupuesto eléctrico (kW)")
    routers: List[RouterConfig]
    servers: List[ServerConfig]
    links: List[LinkConfig]
    simulation_defaults: SimulationDefaults

    @model_validator(mode="after")
    def validate_topology_integrity(self) -> AppConfig:
        router_ids = {r.id for r in self.routers}
        server_ids = {s.id for s in self.servers}
        all_node_ids = router_ids.union(server_ids)

        # Validar colisión de IDs
        if len(router_ids) != len(self.routers):
            raise ValueError("Existen IDs de routers duplicados en la configuración")
        if len(server_ids) != len(self.servers):
            raise ValueError("Existen IDs de servidores duplicados en la configuración")
        if router_ids.intersection(server_ids):
            raise ValueError("Conflicto de IDs entre routers y servidores")

        # Validar enlaces a nodos existentes
        for link in self.links:
            if link.source not in all_node_ids:
                raise ValueError(f"El nodo origen '{link.source}' en el enlace no existe")
            if link.target not in all_node_ids:
                raise ValueError(f"El nodo destino '{link.target}' en el enlace no existe")

        # Validar routers conectados a servidores
        for s in self.servers:
            if s.connected_router not in router_ids:
                raise ValueError(
                    f"El servidor '{s.id}' apunta a un router inexistente '{s.connected_router}'"
                )

        return self


def load_config(file_path: Optional[str | Path] = None) -> AppConfig:
    """Carga y valida la configuración desde un archivo JSON, con fallback al predeterminado."""
    if file_path is None:
        base_dir = Path(__file__).resolve().parent
        file_path = base_dir / "default_topology.json"
    else:
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return AppConfig.model_validate(data)
