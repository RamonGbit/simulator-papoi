"""
Pruebas de integración para el motor de eventos discretos SimPy y optimizadores.
"""

import pytest
from config.settings import load_config
from src.simulation.engine import DataCenterSimulation


def test_datacenter_simulation_step_and_snapshot():
    config = load_config()
    sim = DataCenterSimulation(config)

    # Avanzar 10 segundos de simulación
    for _ in range(20):
        sim.advance_time(0.5)

    snapshot = sim.get_snapshot()
    assert snapshot.sim_time >= 9.9
    assert snapshot.metrics.total_packets_created > 0
    assert len(snapshot.routers) == len(config.routers)
    assert len(snapshot.servers) == len(config.servers)
    assert snapshot.lagrange_lambda > 0.0
    assert sum(snapshot.lagrange_allocation.values()) == pytest.approx(config.power_budget_total, 1.0)
