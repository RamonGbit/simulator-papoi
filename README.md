# Simulador de Optimización para Redes de Centros de Datos (Cloud Computing)
### DataStream Networks — Simulación Estocástica y Métodos Cuantitativos

**Universidad José Antonio Páez (UJAP)**  
**Facultad de Ingeniería — Escuela de Ingeniería en Computación**  
**Materia:** Métodos Cuantitativos  
**Evaluación:** 20 Puntos (Módulo 1: 5 pts | Módulo 2: 5 pts | Módulo 3: 5 pts | Defensa: 5 pts)  
**Paradigma:** Programación Orientada a Objetos (POO) Estricta  
**Lenguaje:** Python 3.10+

---

## 1. Descripción del Proyecto

La empresa proveedora de servicios en la nube **"DataStream Networks"** enfrenta desafíos críticos de latencia de red, congestión en routers y elevados costos eléctricos de refrigeración en su principal centro de datos. 

Este software implementa una herramienta de simulación visual interactiva que integra modelos matemáticos de optimización en tiempo real:
1. **Programación Dinámica:** Enrutamiento multi-etapa de mínima latencia y asignación de tareas computacionales con el Modelo de Volumen-Carga (Problema de la Mochila 0/1).
2. **Optimización No Lineal:** Cálculo simbólico de temperatura y puntos críticos de emergencia con SymPy (1 variable), control continuo de ventilación mediante Descenso de Gradiente, y distribución de presupuesto eléctrico global con Multiplicadores de Lagrange (multivariable con restricciones).
3. **Simulación Temporal y Visualización:** Motor de eventos discretos con colas M/M/c y fallas estocásticas en **SimPy**, dashboard visual en **Pygame** (estilo NOC Dark Mode), auditoría asistida por **IA (Google Gemini API)** y exportación ejecutiva en **PDF con ReportLab**.

---

## 2. Arquitectura de Software (POO)

El proyecto está diseñado bajo principios SOLID, alta cohesión y bajo acoplamiento:

```text
simulator-papoi/
├── .env.example                  # Plantilla de variables de entorno
├── .gitignore                    # Filtros de exclusión Git
├── pyproject.toml                # Metadatos y configuración de herramientas
├── requirements.txt              # Dependencias compatibles
├── README.md                     # Documentación técnica y guía de defensa
│
├── config/                       # Configuración y validaciones estrictas
│   ├── default_topology.json     # Topología predeterminada para pruebas
│   └── settings.py               # Modelos Pydantic de validación
│
├── src/                          # Código fuente modular
│   ├── main.py                   # Punto de entrada (Pygame / CLI / Reporte)
│   ├── core/                     # Dominio, Entidades y DTOs sincronizados
│   │   ├── entities.py           # Packet, RouterState, ServerState
│   │   └── state.py              # DTOs DataCenterSnapshot y SimulationMetrics
│   │
│   ├── math_models/              # Modelos matemáticos puros e independientes
│   │   ├── dp_routing.py         # Módulo 1.1: Ruta más corta con Bellman DP
│   │   ├── dp_knapsack.py        # Módulo 1.2: Mochila 0/1 (Volumen-Carga)
│   │   ├── thermal_analysis.py   # Módulo 2.1: Cálculo simbólico (SymPy)
│   │   ├── cooling_gradient.py   # Módulo 2.2: Descenso de gradiente en ventiladores
│   │   └── power_lagrange.py     # Módulo 2.3: Multiplicadores de Lagrange multivariable
│   │
│   ├── simulation/               # Motor de simulación estocástica (SimPy)
│   │   ├── engine.py             # Coordinador de eventos discretos y recursos
│   │   ├── traffic_generator.py  # Llegadas Poisson (Teoría de colas)
│   │   └── failure_model.py      # Fallas estocásticas y recuperación de nodos
│   │
│   ├── ui/                       # Dashboard gráfico en tiempo real (Pygame)
│   │   ├── app.py                # Bucle interactivo principal a 60 FPS
│   │   ├── theme.py              # Paleta visual NOC Dark Mode y tipografías
│   │   ├── components/button.py  # Botones interactivos con eventos
│   │   └── renderers/            # Renderizadores de topología, gráficas y HUD
│   │
│   └── reporting/                # Auditoría y persistencia de reportes
│       ├── ai_advisor.py         # Integración con Google Gemini AI (+ fallback offline)
│       └── pdf_generator.py      # Generador de reportes PDF con ReportLab
│
├── tests/                        # Suite de pruebas unitarias automáticas
│   ├── test_math_models.py       # Pruebas de DP, Mochila, SymPy, Gradiente y Lagrange
│   └── test_simulation.py        # Pruebas del motor SimPy y snapshots
│
└── reports/                      # Directorio de reportes PDF generados
```

---

## 3. Fundamentos Matemáticos

### Módulo 1: Programación Dinámica (5 Ptos)

#### 1.1 Determinación de la Ruta Más Corta
Aplica el **Principio de Optimalidad de Bellman** por etapas:
$$V_k(u) = \min_{v \in \text{Vecinos}(u)} \Big[ C(u, v, t) + V_{k-1}(v) \Big]$$
donde el costo dinámico $C(u, v, t)$ incorpora la latencia física base más la penalización por paquetes encolados:
$$C(u, v, t) = \text{base\_latency}_{u,v} + \gamma \cdot \text{longitud\_cola}_v(t)$$
Si un router se congestiona o cae por una falla estocástica, la tabla de DP recalcula la ruta óptima de forma inmediata.

#### 1.2 Modelo de Volumen-Carga (Mochila DP 0/1)
Cada servidor final posee una capacidad computacional finita $W$ (RAM/CPU ponderada). Para un conjunto de tareas $i=1,\dots,N$ con valor $v_i$ y peso $w_i$:
$$DP[i, w] = \max\Big(DP[i-1, w], DP[i-1, w - w_i] + v_i\Big) \quad \text{si } w \ge w_i$$
Se realiza backtracking sobre la matriz para recuperar la combinación exacta de tareas aceptadas que maximizan la prioridad comercial/SLA.

---

### Módulo 2: Optimización No Lineal (5 Ptos)

#### 2.1 Análisis Térmico Simbólico (1 Variable - SymPy)
Función fenomenológica de temperatura en función de la carga $L \in [0, 1]$:
$$T(L) = 22 + 65L - 54L^2 + 30L^3$$
- **Primera Derivada:** $T'(L) = 90L^2 - 108L + 65 > 0$ (Crecimiento estrictamente monotónico).
- **Segunda Derivada:** $T''(L) = 180L - 108 = 0 \implies L_{\text{inflexión}} = 0.60$ (60% de carga).
- **Umbral Crítico de Enfriamiento:** Para $L \ge 60\%$ ($T \ge 48.04^\circ\text{C}$), el régimen térmico entra en aceleración superlineal, activando el protocolo de enfriamiento de emergencia.

#### 2.2 Ajuste Dinámico de Refrigeración (Descenso de Gradiente)
Minimiza el compromiso entre el consumo eléctrico del ventilador $P$ y la penalización por exceso térmico:
$$J(P) = k_{\text{fan}} \cdot P^2 + k_{\text{penalty}} \cdot \max(0, T_{\text{net}}(P) - T_{\text{target}})^2$$
Actualización iterativa en tiempo real:
$$P^{(k+1)} = P^{(k)} - \alpha \cdot \frac{dJ}{dP}(P^{(k)})$$

#### 2.3 Asignación de Presupuesto Eléctrico (Multiplicadores de Lagrange)
Sean $P_1, \dots, P_k$ las potencias asignadas a los clústeres. Maximizamos la función de rendimiento cóncava con rendimientos decrecientes:
$$R(P) = \sum_{i=1}^k \left( \alpha_i P_i - \frac{1}{2}\beta_i P_i^2 \right) \quad \text{sujeto a} \quad g(P) = \sum_{i=1}^k P_i - P_{\text{total}} = 0$$
Lagrangiano:
$$\mathcal{L}(P, \lambda) = R(P) - \lambda \cdot \left(\sum_{i=1}^k P_i - P_{\text{total}}\right)$$
Condiciones necesarias de primer orden resueltas simbólicamente con `sp.solve`:
$$\frac{\partial \mathcal{L}}{\partial P_i} = \alpha_i - \beta_i P_i - \lambda = 0, \qquad \frac{\partial \mathcal{L}}{\partial \lambda} = -\left(\sum P_i - P_{\text{total}}\right) = 0$$

---

## 4. Instalación y Configuración

### 4.1 Clonar el repositorio y preparar entorno
```bash
# Crear entorno virtual (opcional pero recomendado)
python -m venv .venv
# En Windows:
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 4.2 Variables de Entorno (Opcional)
Copiar `.env.example` a `.env`:
```bash
cp .env.example .env
```
Si deseas que los reportes consulten directamente la API de Google Gemini en la nube, coloca tu clave en `GEMINI_API_KEY`.  
*(Nota: Si no se provee clave, el sistema activará automáticamente el motor heurístico local de respaldo, garantizando que el PDF se genere en cualquier computador sin internet).*

---

## 5. Guía de Ejecución para la Defensa Presencial

### 5.1 Ejecutar la Interfaz Gráfica Interactiva (Pygame)
```bash
python src/main.py
```
**Controles del Dashboard:**
- `[ESPACIO]`: Pausar / Reanudar la simulación temporal de SimPy.
- `[F]`: Provocar una falla estocástica manual en un router (demuestra en vivo cómo la ruta de DP se adapta dinámicamente iluminándose en la pantalla).
- `[R]`: Reparar y restablecer todos los nodos caídos.
- `[T]`: Inyectar una ráfaga de tráfico entrante.
- `[P]`: Generar y exportar el informe ejecutivo en PDF con análisis de IA a la carpeta `reports/`.
- `[ESC]`: Salir del simulador.

### 5.2 Demostración Aislada en Consola (Evaluación de 10 Puntos)
Para demostrar la **independencia de cada módulo** requerida por las pautas:
```bash
python src/main.py --cli
```
Imprime en pantalla la resolución paso a paso de cada modelo matemático:
- Enrutamiento DP con y sin congestión.
- Matriz de la Mochila DP y selección óptima de tareas.
- Derivadas analíticas de SymPy y punto de inflexión.
- Convergencia del Descenso de Gradiente.
- Sistema de Lagrange y cálculo del multiplicador óptimo $\lambda^*$.

### 5.3 Ejecutar Suite de Pruebas Unitarias
```bash
pytest -v
```
Ejecuta las pruebas automáticas para certificar 100% de éxito en todos los algoritmos.

### 5.4 Cargar una Configuración Personalizada
```bash
python src/main.py --config config/default_topology.json
```
Valida la integridad de los datos (IDs únicos, conexiones válidas, capacidades estrictamente positivas) antes de iniciar.
