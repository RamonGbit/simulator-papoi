"""
Pruebas unitarias completas para los Módulos 1 y 2 (Modelos Matemáticos).
Valida la independencia, ausencia de errores y exactitud conceptual requerida para los 10 puntos.
"""

import pytest
from src.math_models.dp_routing import DPRoutingOptimizer
from src.math_models.dp_knapsack import DPKnapsackScheduler, TaskItem
from src.math_models.thermal_analysis import ThermalAnalyzer1D
from src.math_models.cooling_gradient import GradientDescentCooling
from src.math_models.power_lagrange import LagrangePowerBudget, ClusterEnergySpec


# =========================================================================
# 1. Pruebas Módulo 1.1: Ruta Más Corta (Programación Dinámica)
# =========================================================================
def test_dp_routing_optimal_path_and_congestion_reroute():
    links = [
        {"source": "R_IN", "target": "R1", "base_latency": 2.0, "congestion_factor": 0.5},
        {"source": "R_IN", "target": "R2", "base_latency": 5.0, "congestion_factor": 0.1},
        {"source": "R1", "target": "S_DEST", "base_latency": 2.0, "congestion_factor": 0.5},
        {"source": "R2", "target": "S_DEST", "base_latency": 1.0, "congestion_factor": 0.1},
    ]
    optimizer = DPRoutingOptimizer(links)

    # Sin congestión: R_IN -> R1 -> S_DEST cuesta 2.0 + 2.0 = 4.0 ms (vs R2 que cuesta 5.0 + 1.0 = 6.0 ms)
    res_normal = optimizer.find_shortest_path("R_IN", "S_DEST")
    assert res_normal.is_reachable is True
    assert res_normal.path == ["R_IN", "R1", "S_DEST"]
    assert res_normal.total_cost == pytest.approx(4.0, 1e-4)

    # Con congestión en R1 (cola = 10 paquetes): Costo R_IN -> R1 pasa a ser 2 + 0.5*10 = 7 ms
    # La nueva ruta óptima debe ser R_IN -> R2 -> S_DEST (costo 6.0 ms)
    res_congested = optimizer.find_shortest_path("R_IN", "S_DEST", queue_lengths={"R1": 10})
    assert res_congested.is_reachable is True
    assert res_congested.path == ["R_IN", "R2", "S_DEST"]
    assert res_congested.total_cost == pytest.approx(6.0, 1e-4)


def test_dp_routing_node_failure():
    links = [
        {"source": "A", "target": "B", "base_latency": 1.0, "congestion_factor": 0.0},
        {"source": "B", "target": "C", "base_latency": 1.0, "congestion_factor": 0.0},
        {"source": "A", "target": "D", "base_latency": 10.0, "congestion_factor": 0.0},
        {"source": "D", "target": "C", "base_latency": 10.0, "congestion_factor": 0.0},
    ]
    optimizer = DPRoutingOptimizer(links)
    res = optimizer.find_shortest_path("A", "C", failed_nodes={"B"})
    assert res.is_reachable is True
    assert res.path == ["A", "D", "C"]
    assert res.total_cost == pytest.approx(20.0, 1e-4)


# =========================================================================
# 2. Pruebas Módulo 1.2: Modelo Volumen-Carga (Mochila DP)
# =========================================================================
def test_dp_knapsack_maximization():
    scheduler = DPKnapsackScheduler(capacity_limit=15)
    tasks = [
        TaskItem("T1", "A", cpu_required=4, ram_required=4, value=30.0),   # wt = 6
        TaskItem("T2", "B", cpu_required=6, ram_required=6, value=50.0),   # wt = 9
        TaskItem("T3", "C", cpu_required=10, ram_required=10, value=70.0), # wt = 15
        TaskItem("T4", "D", cpu_required=2, ram_required=2, value=20.0),   # wt = 3
    ]
    res = scheduler.optimize_allocation("SERVER_1", tasks)
    # Selecciones óptimas con límite 15:
    # Opción 1: T1 (6) + T2 (9) = wt 15, val = 80.0
    # Opción 2: T3 (15) = val 70.0
    # Opción 3: T2 (9) + T4 (3) = wt 12, val 70.0
    # El óptimo global es 80.0 (T1 + T2)
    assert res.total_value == pytest.approx(80.0, 1e-2)
    selected_ids = {t.task_id for t in res.selected_tasks}
    assert "T1" in selected_ids
    assert "T2" in selected_ids
    assert res.total_weight_used == 15
    assert res.utilization_percentage == pytest.approx(100.0, 1e-2)


# =========================================================================
# 3. Pruebas Módulo 2.1: Análisis Térmico Simbólico (1 Variable)
# =========================================================================
def test_thermal_analysis_critical_inflection_point():
    analyzer = ThermalAnalyzer1D(t_ambient=22.0, c1=65.0, c2=54.0, c3=30.0)
    analysis = analyzer.analyze()

    assert analysis.emergency_threshold_load == pytest.approx(0.60, 1e-2)
    assert analysis.emergency_threshold_temp == pytest.approx(48.04, 1e-1)
    assert analysis.is_superlinear_risk is True
    assert len(analysis.inflection_points) > 0
    # La temperatura al 100% de carga debe ser superior que a carga idle
    temp_idle = analyzer.evaluate_temp(0.0)
    temp_full = analyzer.evaluate_temp(1.0)
    assert temp_idle == pytest.approx(22.0, 1e-2)
    assert temp_full > 60.0


# =========================================================================
# 4. Pruebas Módulo 2.2: Descenso de Gradiente para Ventiladores
# =========================================================================
def test_gradient_cooling_convergence():
    cooler = GradientDescentCooling(
        k_fan=0.04,
        k_penalty=2.0,
        eta_cooling=0.45,
        target_temp=60.0,
        min_power=5.0,
        max_power=120.0,
        learning_rate=0.10,
        tolerance=1e-3,
        max_iterations=150
    )
    # Servidor sobrecalentado a 85°C
    res = cooler.optimize_full(t_base=85.0, initial_power=5.0)
    assert res.optimal_cost < res.initial_cost
    assert res.optimal_power > res.initial_power
    assert res.final_temp < 85.0
    assert res.converged is True


# =========================================================================
# 5. Pruebas Módulo 2.3: Multiplicadores de Lagrange Multivariable
# =========================================================================
def test_lagrange_power_allocation_satisfies_budget():
    clusters = [
        ClusterEnergySpec("c1", "Web", alpha=10.0, beta=0.05),
        ClusterEnergySpec("c2", "DB", alpha=14.0, beta=0.08),
        ClusterEnergySpec("c3", "AI", alpha=18.0, beta=0.04),
    ]
    lagrange_opt = LagrangePowerBudget(clusters)
    budget = 250.0
    res = lagrange_opt.optimize_symbolic(budget)

    total_allocated = sum(res.cluster_allocations.values())
    assert total_allocated == pytest.approx(budget, 1e-1)
    assert res.is_budget_satisfied is True
    assert res.lambda_multiplier > 0.0
    # El cluster AI (con mayor alpha) debe recibir una asignación significativa
    assert res.cluster_allocations["c3"] > res.cluster_allocations["c1"]
