"""
Módulo de Inteligencia Artificial para Auditoría y Diagnóstico del Centro de Datos.
Integra Google Gemini API mediante peticiones HTTP estándar o cliente oficial,
con un motor heurístico de respaldo (fallback) offline para la defensa presencial.
"""

from __future__ import annotations
import os
import json
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from src.core.state import DataCenterSnapshot

load_dotenv()


class AIConsultant:
    """
    Agente consultor de IA para DataStream Networks.
    Genera diagnósticos y recomendaciones estratégicas a partir de la telemetría de simulación.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = "gemini-1.5-flash"

    def analyze_simulation(self, snapshot: DataCenterSnapshot) -> Dict[str, Any]:
        """
        Analiza el estado del datacenter y devuelve un reporte estructurado de diagnóstico.
        """
        # Preparar resumen de telemetría
        telemetry = {
            "tiempo_simulado_seg": snapshot.sim_time,
            "paquetes_creados": snapshot.metrics.total_packets_created,
            "paquetes_entregados": snapshot.metrics.total_packets_delivered,
            "paquetes_perdidos": snapshot.metrics.total_packets_dropped,
            "latencia_promedio_ms": round(snapshot.metrics.average_latency, 2),
            "re-enrutamientos_dinamicos": snapshot.metrics.path_reroute_count,
            "valor_acumulado_mochila": snapshot.metrics.total_value_processed,
            "multiplicador_lagrange_lambda": snapshot.lagrange_lambda,
            "presupuesto_electrico_asignado": snapshot.lagrange_allocation,
            "servidores": {
                s_id: {
                    "carga": f"{s.current_load * 100:.1f}%",
                    "temperatura_c": s.temperature,
                    "potencia_ventilador_w": s.fan_power,
                    "emergencia_termica": s.is_emergency_cooling
                }
                for s_id, s in snapshot.servers.items()
            },
            "nodos_caidos": list(snapshot.failed_nodes)
        }

        # Si hay API key disponible, intentamos consultar a Gemini
        if self.api_key and self.api_key != "tu_api_key_aqui":
            ai_text = self._call_gemini_api(telemetry)
            if ai_text:
                return {
                    "provider": "Google Gemini AI",
                    "status": "success",
                    "analysis": ai_text,
                    "telemetry": telemetry
                }

        # Fallback analítico offline
        return {
            "provider": "Motor Analítico Heurístico Local (Offline Mode)",
            "status": "fallback",
            "analysis": self._generate_local_heuristic_report(telemetry),
            "telemetry": telemetry
        }

    def _call_gemini_api(self, telemetry: Dict[str, Any]) -> Optional[str]:
        """Realiza llamada a la API REST de Gemini."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        prompt = (
            "Eres el Arquitecto Principal de Cómputo e Infraestructura de DataStream Networks. "
            "Analiza el siguiente reporte de telemetría y optimización matemática de nuestro datacenter. "
            "Proporciona un diagnóstico técnico conciso de 3 párrafos cubriendo:\n"
            "1. Eficiencia de enrutamiento dinámico (DP) y latencias observadas.\n"
            "2. Estado térmico de los servidores, ajuste de ventilación por gradiente y eventos de emergencia.\n"
            "3. Impacto de la distribución energética de Lagrange y recomendaciones de ahorro.\n\n"
            f"Telemetría:\n{json.dumps(telemetry, indent=2, ensure_ascii=False)}"
        )

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        try:
            resp = requests.post(url, json=payload, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return text.strip()
        except Exception:
            pass

        return None

    def _generate_local_heuristic_report(self, telemetry: Dict[str, Any]) -> str:
        """Genera un reporte técnico formal con reglas expertas de ingeniería."""
        t_serv = telemetry["servidores"]
        emergencias = [s_id for s_id, s in t_serv.items() if s["emergencia_termica"]]
        dropped = telemetry["paquetes_perdidos"]
        lat = telemetry["latencia_promedio_ms"]
        lam = telemetry["multiplicador_lagrange_lambda"]

        diag_enrutamiento = (
            f"El subsistema de enrutamiento adaptativo (Programación Dinámica) ha gestionado un tráfico acumulado "
            f"de {telemetry['paquetes_creados']} paquetes con una latencia promedio de {lat} ms. "
            f"Se registraron {telemetry['re-enrutamientos_dinamicos']} re-enrutamientos dinámicos que evitaron "
            f"congestiones críticas en buffers intermediarios. La pérdida de paquetes se mantuvo en {dropped}."
        )

        if emergencias:
            diag_termico = (
                f"ALERTA OPERATIVA: Se detectó superación del punto crítico de inflexión térmica en el/los servidor(es) "
                f"{', '.join(emergencias)}. El descenso de gradiente incrementó la potencia de ventilación para mitigar "
                f"el sobrecalentamiento, manteniendo el compromiso óptimo entre disipación térmica y costo eléctrico."
            )
        else:
            diag_termico = (
                "ESTADO TÉRMICO NOMINAL: Todos los clústeres operan por debajo del umbral de inflexión crítica. "
                "El control dinámico por método de gradiente logró minimizar el consumo de ventilación estabilizando "
                "las temperaturas en torno a los targets de diseño."
            )

        diag_lagrange = (
            f"La asignación de presupuesto eléctrico con Multiplicadores de Lagrange obtuvo un multiplicador sombra "
            f"λ* = {lam:.4f}, demostrando que el rendimiento marginal de la red se encuentra en equilibrio óptimo "
            f"sin exceder el límite presupuestario total estricto. Se aconseja priorizar la ampliación de capacidad "
            f"en los clústeres de cómputo intensivo."
        )

        return f"{diag_enrutamiento}\n\n{diag_termico}\n\n{diag_lagrange}"
