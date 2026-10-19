"""
Punto de Entrada Principal (Main Entry Point) del Simulador de Optimización DataStream Networks.
Universidad José Antonio Páez — Escuela de Ingeniería en Computación.

Permite ejecutar:
  1. El Dashboard gráfico interactivo en Pygame (Modo por defecto).
  2. Demostración en consola de todos los módulos matemáticos aislados (Modo --cli / --demo).
  3. Generación directa de reportes PDF con Inteligencia Artificial (Modo --report).
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path

# Configurar salida UTF-8 para compatibilidad universal de caracteres matemáticos en terminales Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Asegurar que el directorio raíz esté en sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config.settings import load_config, AppConfig
from src.math_models.dp_routing import DPRoutingOptimizer
from src.math_models.dp_knapsack import DPKnapsackScheduler, TaskItem
from src.math_models.thermal_analysis import ThermalAnalyzer1D
from src.math_models.cooling_gradient import GradientDescentCooling
from src.math_models.power_lagrange import LagrangePowerBudget, ClusterEnergySpec
from src.math_models.nonlinear_solver import (
    UnivariateOptimizer,
    MultivariateHessianOptimizer,
    GradientDescentOptimizer,
    ConstrainedLagrangeOptimizer,
    NonLinearLabSuite
)
from src.simulation.engine import DataCenterSimulation

from src.reporting.pdf_generator import PDFReportGenerator


def run_cli_demo(config: AppConfig) -> None:
    """Ejecuta una demostración de consola paso a paso de cada módulo matemático."""
    print("=" * 75)
    print("  DATASTREAM NETWORKS: DEMOSTRACIÓN DE MODELOS MATEMÁTICOS (POO)")
    print("  Universidad José Antonio Páez - Métodos Cuantitativos")
    print("=" * 75)

    # ---------------------------------------------------------
    # MÓDULO 1.1: Ruta Más Corta por Programación Dinámica
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("[MÓDULO 1.1] RUTA MÁS CORTA POR PROGRAMACIÓN DINÁMICA (5 PTOS)")
    print("=" * 50)
    links_data = [link.model_dump() for link in config.links]
    router = DPRoutingOptimizer(links_data)
    
    # Caso 1: Normal
    r_norm = router.find_shortest_path("R_IN", "S_WEB")
    print(f"• Caso Normal (Sin congestión):")
    print(f"  Ruta óptima calculada (DP): {' -> '.join(r_norm.path)}")
    print(f"  Latencia mínima acumulada:  {r_norm.total_cost:.2f} ms en {r_norm.stages_evaluated} etapas.")

    # Caso 2: Congestión inducida
    r_cong = router.find_shortest_path("R_IN", "S_WEB", queue_lengths={"R1": 25})
    print(f"\n• Caso Adaptativo (Router R1 con 25 paquetes en cola):")
    print(f"  Nueva ruta óptima adaptada: {' -> '.join(r_cong.path)}")
    print(f"  Latencia mínima acumulada:  {r_cong.total_cost:.2f} ms (Evitó cuello de botella en R1).")

    # ---------------------------------------------------------
    # MÓDULO 1.2: Modelo Volumen-Carga (Mochila DP)
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("[MÓDULO 1.2] MODELO VOLUMEN-CARGA (MOCHILA DP - 5 PTOS)")
    print("=" * 50)
    scheduler = DPKnapsackScheduler(capacity_limit=30)
    sample_tasks = [
        TaskItem("T1", "Auth Microservice", 4, 4, 45.0),
        TaskItem("T2", "Database Query Engine", 8, 12, 90.0),
        TaskItem("T3", "Telemetry Ingestion", 2, 4, 30.0),
        TaskItem("T4", "AI Neural Inference", 14, 20, 140.0),
        TaskItem("T5", "Backup Compression", 6, 8, 25.0)
    ]
    knap_res = scheduler.optimize_allocation("SERVER_CORE_1", sample_tasks)
    print(f"• Capacidad del Servidor: {knap_res.capacity_limit} unidades de RAM/CPU ponderadas")
    print(f"• Valor Máximo Obtenido (SLA): {knap_res.total_value} pts")
    print(f"• Utilización de Recursos: {knap_res.total_weight_used}/{knap_res.capacity_limit} ({knap_res.utilization_percentage}%)")
    print("  Tareas Seleccionadas:")
    for t in knap_res.selected_tasks:
        print(f"    [+] {t.task_id} ({t.name}): Peso={t.composite_weight}, Valor={t.value}")
    print("  Tareas Rechazadas/En cola:")
    for t in knap_res.rejected_tasks:
        print(f"    [-] {t.task_id} ({t.name}): Peso={t.composite_weight}, Valor={t.value}")

    # ---------------------------------------------------------
    # MÓDULO 2.1: Cálculo Diferencial Térmico (SymPy)
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("[MÓDULO 2.1] ANÁLISIS TÉRMICO DIFERENCIAL (1 VARIABLE - 5 PTOS)")
    print("=" * 50)
    thermal = ThermalAnalyzer1D(t_ambient=22.0, c1=65.0, c2=54.0, c3=30.0)
    t_res = thermal.analyze()
    print(f"• Función No Lineal T(L):    {t_res.expression_latex}")
    print(f"• Primera Derivada T'(L):    {t_res.first_derivative_latex}")
    print(f"• Segunda Derivada T''(L):   {t_res.second_derivative_latex}")
    print(f"• PUNTOS DE INFLEXIÓN (d²T/dL² = 0):")
    for pt in t_res.inflection_points:
        print(f"    -> Carga L = {pt['load']*100:.1f}%, Temp = {pt['temp']}°C")
    print(f"• UMBRAL CRÍTICO DE ENFRIAMIENTO: Carga >= {t_res.emergency_threshold_load*100:.1f}% ({t_res.emergency_threshold_temp}°C)")

    # ---------------------------------------------------------
    # MÓDULO 2.2: Ajuste de Refrigeración por Método de Gradiente
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("[MÓDULO 2.2] AJUSTE POR MÉTODO DE GRADIENTE DESCENDENTE (5 PTOS)")
    print("=" * 50)
    cooler = GradientDescentCooling(k_fan=0.04, k_penalty=2.0, target_temp=60.0)
    c_res = cooler.optimize_full(t_base=85.0, initial_power=10.0)
    print(f"• Temp Inicial del Servidor: 85.0°C | Potencia Inicial Fan: {c_res.initial_power} W (Costo: {c_res.initial_cost})")
    print(f"• Potencia Óptima Convergida: {c_res.optimal_power} W (Costo Mínimo: {c_res.optimal_cost})")
    print(f"• Temp Final Estabilizada:   {c_res.final_temp}°C en {c_res.iterations} iteraciones (Convergencia: {c_res.converged})")

    # ---------------------------------------------------------
    # MÓDULO 2.3: Asignación de Presupuesto con Multiplicadores de Lagrange
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("[MÓDULO 2.3] MULTIPLICADORES DE LAGRANGE MULTIVARIABLE (5 PTOS)")
    print("=" * 50)
    cluster_specs = [
        ClusterEnergySpec("web", "Clúster Web", alpha=8.0, beta=0.05),
        ClusterEnergySpec("db", "Clúster DB", alpha=12.0, beta=0.08),
        ClusterEnergySpec("ai", "Clúster IA", alpha=15.0, beta=0.04),
    ]
    lagrange = LagrangePowerBudget(cluster_specs)
    l_res = lagrange.optimize_symbolic(total_budget=config.power_budget_total)
    print(f"• Presupuesto Eléctrico Total (Restricción g): {l_res.total_budget} kW")
    print(f"• Función Lagrangiana L: {l_res.lagrangian_latex}")
    print(f"• Multiplicador de Lagrange Óptimo (lambda*): {l_res.lambda_multiplier:.4f}")
    print("• Asignaciones Óptimas por Clúster:")
    for cid, pot in l_res.cluster_allocations.items():
        print(f"    [+] Clúster {cid.upper()}: {pot} kW")
    print(f"• Rendimiento Total Máximo: {l_res.total_performance:.2f} unidades")

    # ---------------------------------------------------------
    # MÓDULO 2.4: Optimización Multivariable y Matriz Hessiana (PLUS / EXTRA)
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("[MODULO 2.4] MODELO MULTIVARIABLE Y MATRIZ HESSIANA (PLUS/EXTRA)")
    print("=" * 50)
    preset_multi = NonLinearLabSuite.get_datacenter_presets()["cpu_efficiency_multivariate"]
    multi_opt = MultivariateHessianOptimizer(preset_multi["expr"], preset_multi["vars"])
    m_res = multi_opt.analyze()
    print(f"* Funcion Multivariable E(f, v): {m_res.expression_str}")
    print(f"* Vector Gradiente grad(E) = [dE/df, dE/dv]:")
    for var, g_expr in m_res.gradient_str.items():
        print(f"    -> dE/d{var} = {g_expr}")
    print(f"* Matriz Hessiana Simbolica H:")
    for row in m_res.hessian_matrix_symbolic:
        print(f"    [ {row[0]:<12} {row[1]:<12} ]")
    if m_res.critical_points:
        cp = m_res.critical_points[0]
        print(f"* Punto Critico Analitico: f = {cp.point_dict.get('f')} GHz, v = {cp.point_dict.get('v')} V")
        print(f"* Menores Principales Lideres: Delta1 = {cp.principal_minors[0]}, Delta2 = det(H) = {cp.principal_minors[1]}")
        print(f"* Clasificacion Formal (Sylvester): {cp.classification.upper()} (Disipacion minima)")

    print("\n" + "=" * 75)
    print("  TODOS LOS MODULOS EVALUADOS Y FUNCIONANDO CON EXITO.")
    print("=" * 75)


def run_nonlinear_suite() -> None:
    """Ejecuta la suite completa de los 4 métodos de Programación No Lineal."""
    print("=" * 75)
    print("  LABORATORIO DE PROGRAMACION NO LINEAL -- METODOS CUANTITATIVOS (UJAP)")
    print("  DataStream Networks: Demostracion Completa de Modelos No Lineales")
    print("=" * 75)

    presets = NonLinearLabSuite.get_datacenter_presets()

    # 1. Univariable
    print("\n" + "#" * 60)
    print("METODO 1: MODELO DE UNA VARIABLE (EXTREMOS E INFLEXION)")
    print("#" * 60)
    u_p = presets["thermal_1d"]
    u_opt = UnivariateOptimizer(u_p["expr"], var_name=u_p["var"])
    u_res = u_opt.analyze(domain_min=0.0, domain_max=1.0)
    print(f"* Funcion Objetivo: f({u_p['var']}) = {u_res.expression_str}")
    print(f"* Primera Derivada: f'({u_p['var']}) = {u_res.first_derivative_str}")
    print(f"* Segunda Derivada: f''({u_p['var']}) = {u_res.second_derivative_str}")
    print("* Puntos de Inflexion Criticos (d2f/dx2 = 0):")
    for pt in u_res.inflection_points:
        print(f"    -> Carga {pt.x*100:.1f}%, Temp = {pt.f_x:.2f} C [{pt.classification}]")

    # 2. Multivariable con Matriz Hessiana
    print("\n" + "#" * 60)
    print("METODO 2: MODELO MULTIVARIABLE Y CLASIFICACION CON HESSIANO")
    print("#" * 60)
    m_p = presets["cpu_efficiency_multivariate"]
    m_opt = MultivariateHessianOptimizer(m_p["expr"], m_p["vars"])
    m_res = m_opt.analyze()
    print(f"* Funcion Objetivo: f({', '.join(m_p['vars'])}) = {m_res.expression_str}")
    print(f"* Vector Gradiente grad(f):")
    for v, g in m_res.gradient_str.items():
        print(f"    df/d{v} = {g}")
    print(f"* Matriz Hessiana H:")
    for row in m_res.hessian_matrix_symbolic:
        print(f"    [ {'   '.join(row)} ]")
    if m_res.critical_points:
        cp = m_res.critical_points[0]
        print(f"* Punto Critico: {cp.point_dict} con f(x*) = {cp.f_val}")
        print(f"* Menores Principales: Delta1 = {cp.principal_minors[0]}, Delta2 = det(H) = {cp.principal_minors[1]}")
        print(f"* Dictamen Sylvester: {cp.classification.upper()}")

    # 3. Método del Gradiente
    print("\n" + "#" * 60)
    print("METODO 3: METODO DEL GRADIENTE DESCENDENTE (NUMERICO)")
    print("#" * 60)
    g_p = presets["cooling_gradient"]
    g_opt = GradientDescentOptimizer(g_p["expr"], g_p["vars"], learning_rate=g_p["lr"], tolerance=1e-3, max_iterations=60)
    g_res = g_opt.optimize(g_p["initial"], mode="minimize")
    print(f"* Funcion de Costo: J(P) = {g_p['expr']}")
    print(f"* Punto Inicial: {g_res.initial_point} | Tasa alfa: {g_p['lr']}")
    print(f"* Convergencia: {g_res.converged} en {g_res.iterations} iteraciones")
    print(f"* Punto Optimo: {g_res.optimal_point} | Costo Minimo: {g_res.optimal_val}")
    print("* Muestra de Iteraciones:")
    for it in g_res.history[:3] + g_res.history[-2:]:
        print(f"    Paso {it.step:>2}: P = {it.coords['P']:.2f} W, Costo = {it.f_val:.4f}, ||grad(J)|| = {it.grad_norm:.6f}")

    # 4. Multiplicadores de Lagrange
    print("\n" + "#" * 60)
    print("METODO 4: MULTIPLICADORES DE LAGRANGE (CON RESTRICCIONES)")
    print("#" * 60)
    l_p = presets["cluster_lagrange"]
    l_opt = ConstrainedLagrangeOptimizer(l_p["objective"], l_p["constraints"], l_p["vars"])
    l_res = l_opt.solve(mode="maximize")
    print(f"* Funcion Rendimiento: R(P) = {l_p['objective']}")
    print(f"* Restriccion Presupuestaria: {l_p['constraints'][0]} = 0")
    print(f"* Punto Optimo: {l_res.best_point}")
    print(f"* Multiplicadores lambda*: {l_res.lambda_multipliers[0] if l_res.lambda_multipliers else 'N/A'}")
    print(f"* Rendimiento Maximo: {l_res.best_value} unidades")
    print("\n" + "=" * 75)


def run_interactive_calculator() -> None:
    """Calculadora interactiva en terminal para evaluar ecuaciones no lineales personalizadas."""
    print("=" * 75)
    print("  CALCULADORA DE MODELOS NO LINEALES (RESOLUCIÓN PERSONALIZADA)")
    print("=" * 75)
    print("Seleccione el tipo de operación no lineal a resolver:")
    print("  1. Función de 1 Variable (f'(x), f''(x), Extremos e Inflexión)")
    print("  2. Función Multivariable (Gradiente, Matriz Hessiana y Sylvester)")
    print("  3. Método del Gradiente Descendente (Iterativo)")
    print("  4. Multiplicadores de Lagrange (Con Restricción de Igualdad)")
    print("  0. Salir")

    choice = input("\nIngrese opción [0-4]: ").strip()

    if choice == "1":
        expr = input("Ingrese f(x) (ej: x**3 - 3*x**2 + 2): ").strip() or "x**3 - 3*x**2 + 2"
        opt = UnivariateOptimizer(expr, var_name="x")
        res = opt.analyze()
        print(f"\n[OK] Función: f(x) = {res.expression_str}")
        print(f"     f'(x)  = {res.first_derivative_str}")
        print(f"     f''(x) = {res.second_derivative_str}")
        print("\nPuntos Estacionarios:")
        for p in res.stationary_points:
            print(f"  • x = {p.x}: f(x) = {p.f_x} -> {p.classification}")
        print("\nPuntos de Inflexión:")
        for p in res.inflection_points:
            print(f"  • x = {p.x}: f(x) = {p.f_x} -> Punto de Inflexión")

    elif choice == "2":
        expr = input("Ingrese f(x, y) (ej: 4*x**2 + 6*y**2 - 2*x*y - 24*x - 30*y + 100): ").strip()
        if not expr:
            expr = "4*x**2 + 6*y**2 - 2*x*y - 24*x - 30*y + 100"
        opt = MultivariateHessianOptimizer(expr, ["x", "y"])
        res = opt.analyze()
        print(f"\n[OK] Función: {res.expression_str}")
        print(f"     ∂f/∂x = {res.gradient_str['x']}")
        print(f"     ∂f/∂y = {res.gradient_str['y']}")
        print(f"     Matriz Hessiana Simbólica:")
        for row in res.hessian_matrix_symbolic:
            print(f"       [ {row[0]:<10} {row[1]:<10} ]")
        for cp in res.critical_points:
            print(f"\nPunto Crítico: {cp.point_dict}")
            print(f"  • Matriz Hessiana Evaluada: {cp.hessian_matrix_evaluated}")
            print(f"  • Menores Principales: {cp.principal_minors}")
            print(f"  • Clasificación (Sylvester): {cp.classification.upper()}")

    elif choice == "3":
        expr = input("Ingrese f(x, y) a minimizar (ej: (x-3)**2 + (y+2)**2): ").strip() or "(x-3)**2 + (y+2)**2"
        opt = GradientDescentOptimizer(expr, ["x", "y"], learning_rate=0.1, tolerance=1e-4)
        res = opt.optimize({"x": 0.0, "y": 0.0}, mode="minimize")
        print(f"\n[OK] Convergió: {res.converged} en {res.iterations} iteraciones")
        print(f"     Punto Óptimo: {res.optimal_point}")
        print(f"     Valor Mínimo: {res.optimal_val}")

    elif choice == "4":
        obj = input("Función Objetivo f(x, y) (ej: 10*x - x**2 + 20*y - 2*y**2): ").strip() or "10*x - x**2 + 20*y - 2*y**2"
        g = input("Restricción g(x, y) = 0 (ej: x + y - 10): ").strip() or "x + y - 10"
        opt = ConstrainedLagrangeOptimizer(obj, [g], ["x", "y"])
        res = opt.solve(mode="maximize")
        print(f"\n[OK] Lagrangiano: {res.lagrangian_str}")
        print(f"     Punto Óptimo: {res.best_point}")
        print(f"     Multiplicador λ*: {res.lambda_multipliers[0] if res.lambda_multipliers else 'N/A'}")
        print(f"     Valor Máximo: {res.best_value}")



def generate_direct_report(config: AppConfig) -> None:
    """Ejecuta una simulación corta y produce el reporte PDF con IA."""
    print("Iniciando simulación previa para generar telemetría...")
    sim = DataCenterSimulation(config)
    for _ in range(30):
        sim.advance_time(0.5)

    snap = sim.get_snapshot()
    pdf_gen = PDFReportGenerator()
    print("Generando reporte PDF con auditoría de IA...")
    path = pdf_gen.generate_report(snap)
    print(f"\n¡Éxito! Reporte guardado en: {path}")


def interactive_config_prompt(config: AppConfig) -> AppConfig:
    """Solicita interactivamente valores de variables al usuario por terminal con validación estricta (Pauta 6)."""
    print("\n" + "=" * 70)
    print("  MODO INTERACTIVO: AJUSTE DE VARIABLES MATEMÁTICAS Y DE SIMULACIÓN")
    print("  (Presione ENTER en cualquier campo para conservar el valor por defecto)")
    print("=" * 70)

    # 1. Presupuesto Eléctrico (Lagrange)
    while True:
        raw = input(f"1. Presupuesto Eléctrico Global Lagrange (kW) [{config.power_budget_total}]: ").strip()
        if not raw:
            break
        try:
            val = float(raw)
            if val <= 10.0 or val > 5000.0:
                print("   [!] Error de validación: El presupuesto debe estar entre 10.0 y 5000.0 kW.")
                continue
            config.power_budget_total = val
            break
        except ValueError:
            print("   [!] Error de validación: Debe ingresar un valor numérico decimal válido.")

    # 2. Tasa de llegada de tráfico Poisson (SimPy)
    while True:
        curr_lam = config.simulation_defaults.arrival_rate_lambda
        raw = input(f"2. Tasa de arribo de tráfico Poisson λ (paquetes/s) [{curr_lam}]: ").strip()
        if not raw:
            break
        try:
            val = float(raw)
            if val <= 0.5 or val > 150.0:
                print("   [!] Error de validación: La tasa λ debe estar entre 0.5 y 150.0 paquetes/s.")
                continue
            config.simulation_defaults.arrival_rate_lambda = val
            break
        except ValueError:
            print("   [!] Error de validación: Debe ingresar un valor numérico válido.")

    # 3. Temperatura objetivo (Gradiente)
    while True:
        curr_target = config.servers[0].target_temp
        raw = input(f"3. Temperatura Objetivo de Servidores (°C) [{curr_target}]: ").strip()
        if not raw:
            break
        try:
            val = float(raw)
            if val < 30.0 or val > 80.0:
                print("   [!] Error de validación: La temperatura objetivo debe estar entre 30.0°C y 80.0°C.")
                continue
            for s in config.servers:
                s.target_temp = val
            break
        except ValueError:
            print("   [!] Error de validación: Debe ingresar un número válido.")

    # 4. Learning rate del gradiente descendente
    while True:
        curr_lr = config.simulation_defaults.gradient_learning_rate
        raw = input(f"4. Tasa de Aprendizaje del Gradiente Descendente α [{curr_lr}]: ").strip()
        if not raw:
            break
        try:
            val = float(raw)
            if val <= 0.001 or val > 1.0:
                print("   [!] Error de validación: La tasa de aprendizaje α debe estar en (0.001, 1.0].")
                continue
            config.simulation_defaults.gradient_learning_rate = val
            break
        except ValueError:
            print("   [!] Error de validación: Debe ingresar un número decimal válido.")

    print("\n[OK] Nuevas variables validadas y aplicadas exitosamente.")
    print("=" * 70 + "\n")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulador de Optimización para Redes de Centros de Datos (DataStream Networks)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Ruta a un archivo JSON con topología personalizada para validar y cargar."
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Inicia un asistente de consola para modificar variables numéricas interactivamente antes de simular."
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Modifica el presupuesto eléctrico global de Lagrange (kW)."
    )
    parser.add_argument(
        "--traffic-lambda",
        type=float,
        default=None,
        help="Modifica la tasa de llegada de tráfico Poisson λ (paquetes/segundo)."
    )
    parser.add_argument(
        "--target-temp",
        type=float,
        default=None,
        help="Modifica la temperatura objetivo (°C) para el enfriamiento por gradiente."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Modifica la tasa de aprendizaje α del descenso de gradiente."
    )
    parser.add_argument(
        "--cli", "--demo",
        action="store_true",
        help="Ejecuta la demostración de consola de todos los modelos matemáticos independientes."
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Genera un reporte técnico ejecutivo en PDF con análisis de IA y finaliza."
    )
    parser.add_argument(
        "--nonlinear", "-nl",
        action="store_true",
        help="Ejecuta la suite completa de demostración de los 4 métodos de Programación No Lineal."
    )
    parser.add_argument(
        "--calc", "--solve",
        action="store_true",
        help="Inicia la calculadora interactiva para resolver ecuaciones no lineales libres del usuario."
    )

    args = parser.parse_args()

    # Cargar y validar configuración (cumple requerimientos 6 y 7 de evaluación)
    try:
        config = load_config(args.config)
        print(f"[OK] Configuración '{config.datacenter_name}' validada exitosamente.")
    except Exception as e:
        print(f"[ERROR] Validación fallida de la configuración: {e}", file=sys.stderr)
        sys.exit(1)

    # Validaciones y aplicación de variables introducidas por comandos (Pauta 6)
    if args.budget is not None:
        if args.budget <= 10.0 or args.budget > 5000.0:
            print("[ERROR] El presupuesto eléctrico (--budget) debe estar entre 10 y 5000 kW.", file=sys.stderr)
            sys.exit(1)
        config.power_budget_total = args.budget
        print(f"[OK] Variable actualizada: Presupuesto Lagrange = {args.budget} kW")

    if args.traffic_lambda is not None:
        if args.traffic_lambda <= 0.0 or args.traffic_lambda > 200.0:
            print("[ERROR] La tasa de tráfico (--traffic-lambda) debe estar entre 0.1 y 200.0 pkt/s.", file=sys.stderr)
            sys.exit(1)
        config.simulation_defaults.arrival_rate_lambda = args.traffic_lambda
        print(f"[OK] Variable actualizada: Tráfico Poisson λ = {args.traffic_lambda} pkt/s")

    if args.target_temp is not None:
        if args.target_temp < 30.0 or args.target_temp > 80.0:
            print("[ERROR] La temperatura objetivo (--target-temp) debe estar entre 30 y 80 °C.", file=sys.stderr)
            sys.exit(1)
        for s in config.servers:
            s.target_temp = args.target_temp
        print(f"[OK] Variable actualizada: Temp Objetivo = {args.target_temp} °C")

    if args.learning_rate is not None:
        if args.learning_rate <= 0.0 or args.learning_rate > 1.0:
            print("[ERROR] La tasa de aprendizaje (--learning-rate) debe estar en (0.0, 1.0].", file=sys.stderr)
            sys.exit(1)
        config.simulation_defaults.gradient_learning_rate = args.learning_rate
        print(f"[OK] Variable actualizada: Learning Rate α = {args.learning_rate}")

    # Asistente interactivo si se solicitó -i / --interactive
    if args.interactive:
        config = interactive_config_prompt(config)

    if args.nonlinear:
        run_nonlinear_suite()
    elif args.calc:
        run_interactive_calculator()
    elif args.cli:
        run_cli_demo(config)
    elif args.report:
        generate_direct_report(config)
    else:
        # Lanzar Dashboard interactivo en Pygame
        from src.ui.app import DataCenterDashboardApp
        print("Iniciando Interfaz Gráfica interactiva en Pygame...")
        app = DataCenterDashboardApp(config)
        app.run()


if __name__ == "__main__":
    main()
