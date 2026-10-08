"""
Generador de reportes ejecutivos en formato PDF utilizando ReportLab.
Consolida métricas operativas de SimPy, formulaciones matemáticas de los Módulos 1 y 2,
y el diagnóstico generado por la API de Inteligencia Artificial.
"""

from __future__ import annotations
import os
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable
)

from src.core.state import DataCenterSnapshot
from src.reporting.ai_advisor import AIConsultant


class PDFReportGenerator:
    """
    Generador de informes técnicos y ejecutivos en PDF para DataStream Networks.
    """

    def __init__(self, output_dir: Optional[str | Path] = None):
        if output_dir is None:
            self.output_dir = Path("reports")
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ai_consultant = AIConsultant()

    def generate_report(self, snapshot: DataCenterSnapshot) -> str:
        """
        Genera el archivo PDF con todos los resultados de optimización e IA.
        
        Returns:
            Ruta absoluta del archivo PDF generado.
        """
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reporte_optimizacion_datastream_{timestamp_str}.pdf"
        file_path = self.output_dir / filename

        # 1. Ejecutar análisis de IA
        ai_result = self.ai_consultant.analyze_simulation(snapshot)
        ai_text = ai_result.get("analysis", "No disponible.")
        ai_provider = ai_result.get("provider", "Sistema Local")

        # 2. Configurar documento ReportLab
        doc = SimpleDocTemplate(
            str(file_path),
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        styles = getSampleStyleSheet()
        # Estilos tipográficos personalizados
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=16,
            textColor=colors.HexColor("#0f2b48"),
            spaceAfter=4,
            alignment=1
        )
        subtitle_style = ParagraphStyle(
            'SubtitleStyle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.HexColor("#555555"),
            spaceAfter=12,
            alignment=1
        )
        section_style = ParagraphStyle(
            'SectionStyle',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=12,
            textColor=colors.HexColor("#1b4965"),
            spaceBefore=10,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            'BodyStyle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            textColor=colors.HexColor("#222222"),
            leading=12
        )
        table_text_style = ParagraphStyle(
            'TableText',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=8,
            leading=10
        )
        table_header_style = ParagraphStyle(
            'TableHeader',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=8,
            textColor=colors.whitesmoke,
            leading=10
        )

        elements = []

        # Membrete institucional y empresarial
        elements.append(Paragraph("UNIVERSIDAD JOSÉ ANTONIO PÁEZ — FACULTAD DE INGENIERÍA", subtitle_style))
        elements.append(Paragraph("Escuela de Ingeniería en Computación | Métodos Cuantitativos", subtitle_style))
        elements.append(Paragraph("DATASTREAM NETWORKS: AUDITORÍA DE OPTIMIZACIÓN Y CLOUD COMPUTING", title_style))
        elements.append(Paragraph(f"Fecha de Emisión: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | Tiempo Simulado: {snapshot.sim_time} s", subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f2b48"), spaceAfter=10))

        # Sección 1: Telemetría Operativa (SimPy)
        elements.append(Paragraph("1. Telemetría y Métricas del Centro de Datos (SimPy)", section_style))
        
        telemetry_data = [
            [Paragraph("Métrica Operativa", table_header_style), Paragraph("Valor Obtenido", table_header_style), Paragraph("Evaluación / Estado", table_header_style)],
            [Paragraph("Paquetes Generados (Poisson)", table_text_style), Paragraph(str(snapshot.metrics.total_packets_created), table_text_style), Paragraph("Tráfico Nominal M/M/c", table_text_style)],
            [Paragraph("Paquetes Entregados con Éxito", table_text_style), Paragraph(str(snapshot.metrics.total_packets_delivered), table_text_style), Paragraph(f"Tasa Entrega: {(snapshot.metrics.total_packets_delivered / max(1, snapshot.metrics.total_packets_created))*100:.1f}%", table_text_style)],
            [Paragraph("Paquetes Descartados / Congestión", table_text_style), Paragraph(str(snapshot.metrics.total_packets_dropped), table_text_style), Paragraph("Control de Buffer de Routers", table_text_style)],
            [Paragraph("Latencia Media de Enlace", table_text_style), Paragraph(f"{snapshot.metrics.average_latency:.2f} ms", table_text_style), Paragraph("Optimizado por DP Bellman", table_text_style)],
            [Paragraph("Re-enrutamientos Dinámicos (DP)", table_text_style), Paragraph(str(snapshot.metrics.path_reroute_count), table_text_style), Paragraph("Respuesta adaptativa a fallas", table_text_style)],
            [Paragraph("Valor Acumulado Procesado (SLA)", table_text_style), Paragraph(f"{snapshot.metrics.total_value_processed:.1f} pts", table_text_style), Paragraph("Maximizador Mochila 0/1", table_text_style)]
        ]

        t_telemetry = Table(telemetry_data, colWidths=[200, 130, 200])
        t_telemetry.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1b4965")),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(t_telemetry)
        elements.append(Spacer(1, 10))

        # Sección 2: Módulo 1 - Programación Dinámica
        elements.append(Paragraph("2. Módulo 1: Enrutamiento Dinámico y Modelo Volumen-Carga (DP)", section_style))
        path_str = " -> ".join(snapshot.metrics.active_shortest_path) if snapshot.metrics.active_shortest_path else "En cálculo..."
        elements.append(Paragraph(f"<b>Ruta Óptima Actual (Bellman DP):</b> {path_str}", body_style))
        elements.append(Paragraph("<b>Modelo de la Mochila (0/1):</b> Cada servidor aplica recursión de Bellman maximizando el valor de tareas computacionales admitidas sin exceder la memoria RAM ni núcleos de CPU.", body_style))
        elements.append(Spacer(1, 8))

        # Tabla de Servidores y Carga
        server_rows = [
            [Paragraph("Servidor", table_header_style), Paragraph("Clúster", table_header_style), Paragraph("Carga CPU/RAM", table_header_style), Paragraph("Temperatura", table_header_style), Paragraph("Ventilador (W)", table_header_style), Paragraph("Energía Asignada", table_header_style), Paragraph("Alerta", table_header_style)]
        ]
        for s_id, s in snapshot.servers.items():
            alerta = "EMERGENCIA" if s.is_emergency_cooling else "Nominal"
            server_rows.append([
                Paragraph(s.name, table_text_style),
                Paragraph(s.cluster_id, table_text_style),
                Paragraph(f"{s.current_load*100:.1f}%", table_text_style),
                Paragraph(f"{s.temperature:.1f} °C", table_text_style),
                Paragraph(f"{s.fan_power:.1f} W", table_text_style),
                Paragraph(f"{s.allocated_power:.1f} kW", table_text_style),
                Paragraph(alerta, table_text_style)
            ])

        t_servers = Table(server_rows, colWidths=[105, 75, 75, 70, 75, 75, 55])
        t_servers.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
        ]))
        elements.append(t_servers)
        elements.append(Spacer(1, 10))

        # Sección 3: Módulo 2 - Optimización No Lineal
        elements.append(Paragraph("3. Módulo 2: Optimización Energética No Lineal", section_style))
        elements.append(Paragraph(
            "<b>Cálculo Simbólico (SymPy - 1 Variable):</b> Función térmica <i>T(L) = 22 + 65L - 54L² + 30L³</i>. "
            "Punto de Inflexión analítico resuelto en <b>L = 60.0% (48.04 °C)</b>, marcando el umbral de aceleración térmica crítica.",
            body_style
        ))
        elements.append(Paragraph(
            "<b>Ajuste por Descenso de Gradiente:</b> Potencia continua de ventilación actualizada con <i>P^(k+1) = P^k - α ∇J</i> "
            "minimizando el costo cuadrático de ventilación y disipando el exceso de calor.",
            body_style
        ))
        elements.append(Paragraph(
            f"<b>Asignación de Presupuesto con Multiplicadores de Lagrange:</b> Función de rendimiento cóncava multivariable "
            f"resuelta analíticamente mediante ∇L = 0. Multiplicador de Lagrange obtenido: <b>λ* = {snapshot.lagrange_lambda:.4f}</b>.",
            body_style
        ))
        elements.append(Spacer(1, 10))

        # Sección 4: Inteligencia Artificial
        elements.append(Paragraph(f"4. Diagnóstico Técnico y Dictamen Estratégico ({ai_provider})", section_style))
        # Formatear párrafos del reporte de IA
        for p in ai_text.split("\n\n"):
            if p.strip():
                elements.append(Paragraph(p.strip(), body_style))
                elements.append(Spacer(1, 4))

        elements.append(Spacer(1, 12))
        elements.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cccccc"), spaceAfter=6))
        elements.append(Paragraph("Documento generado automáticamente por el Simulador de Optimización DataStream Networks. Aprobado para evaluación académica.", subtitle_style))

        # Construir PDF
        doc.build(elements)
        return str(file_path.resolve())


if __name__ == "__main__":
    # Prueba independiente del generador de PDF
    print("=== Probando Generación de Reporte PDF ===")
    from config.settings import load_config
    from src.simulation.engine import DataCenterSimulation
    cfg = load_config()
    sim = DataCenterSimulation(cfg)
    sim.advance_time(5.0)
    snap = sim.get_snapshot()

    pdf_gen = PDFReportGenerator()
    pdf_path = pdf_gen.generate_report(snap)
    print(f"Reporte generado exitosamente en: {pdf_path}")
