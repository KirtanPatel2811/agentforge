"""
src/tools/chart_generator.py — Plotly Chart Generation Tool
──────────────────────────────────────────────────────────────
Creates professional charts from structured data, saves to data/outputs/.

Design decisions:
1. WHY PLOTLY OVER MATPLOTLIB? Plotly charts are interactive HTML —
   perfect for Streamlit. They support hover/zoom/pan and look professional.
   Matplotlib is static PNG only (still useful for reports, so we save both).
2. CHART TYPES: bar, line, scatter, pie, histogram, heatmap.
   The Analyst agent picks the right type based on what the data represents.
3. DUAL OUTPUT: Saves both interactive HTML (for dashboard) and static
   PNG (for the report). PNG requires kaleido — we handle it gracefully
   if kaleido isn't installed.
4. CHART REGISTRY: Every chart logged to chart_registry.json in outputs/.
   Writer agent and Streamlit dashboard read this to find all charts.
5. FLEXIBLE INPUT: Accepts JSON string, list-of-dicts, or dict-of-lists.
   Whatever format the Analyst or Coder produces, this tool handles it.

Interview talking point:
    "Charts are registered in a JSON manifest file so the Writer agent
    can query 'all charts for task X' without knowing file names. This
    is the same pattern as a media registry in production systems — loose
    coupling between producers (Analyst) and consumers (Writer/Dashboard)."
"""

import json
import os
import sys
from datetime import datetime
from typing import Optional, Union
from loguru import logger

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from config import settings, OUTPUTS_DIR

try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly not installed. Run: pip install plotly")

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


CHART_REGISTRY_PATH = OUTPUTS_DIR / "chart_registry.json"

COLOUR_PALETTE = [
    "#3B82F6",  # blue
    "#10B981",  # emerald
    "#F59E0B",  # amber
    "#EF4444",  # red
    "#8B5CF6",  # violet
    "#06B6D4",  # cyan
    "#F97316",  # orange
    "#84CC16",  # lime
]


def generate_chart(
    data: Union[str, list, dict],
    chart_type: str,
    x_column: str,
    y_column: str,
    title: str,
    task_id: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    color_column: Optional[str] = None,
    description: Optional[str] = None,
) -> str:
    """
    Generate a Plotly chart and save it to data/outputs/.

    Args:
        data: Data to plot. Can be:
              - JSON string of list-of-dicts: [{"country": "China", "gw": 430}, ...]
              - JSON string of dict-of-lists: {"country": [...], "gw": [...]}
              - Already-parsed list or dict
        chart_type: One of "bar", "line", "scatter", "pie", "histogram", "heatmap"
        x_column: Column for X axis (or labels for pie)
        y_column: Column for Y axis (or values for pie)
        title: Chart title
        task_id: For registry and file naming
        x_label: Custom X axis label
        y_label: Custom Y axis label
        color_column: Optional column for color grouping
        description: Human-readable description of what the chart shows

    Returns:
        JSON string with: success, chart_id, html_path, png_path, message
    """
    if not PLOTLY_AVAILABLE:
        return json.dumps({"success": False, "error": "plotly not installed"})
    if not PANDAS_AVAILABLE:
        return json.dumps({"success": False, "error": "pandas not installed"})

    logger.debug(f"[chart_generator] Creating {chart_type} chart: '{title}'")

    # Parse data into DataFrame
    try:
        df = _parse_to_dataframe(data)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Data parsing failed: {e}"})

    if df.empty:
        return json.dumps(
            {"success": False, "error": "Data is empty — nothing to chart"}
        )

    if x_column not in df.columns:
        return json.dumps(
            {
                "success": False,
                "error": f"Column '{x_column}' not found. Available: {list(df.columns)}",
            }
        )

    if chart_type != "histogram" and y_column not in df.columns:
        return json.dumps(
            {
                "success": False,
                "error": f"Column '{y_column}' not found. Available: {list(df.columns)}",
            }
        )

    # Build the chart
    try:
        fig = _build_chart(
            df=df,
            chart_type=chart_type,
            x_column=x_column,
            y_column=y_column,
            title=title,
            x_label=x_label or x_column,
            y_label=y_label or y_column,
            color_column=color_column,
        )
    except Exception as e:
        logger.error(f"[chart_generator] Build failed: {e}")
        return json.dumps({"success": False, "error": f"Chart creation failed: {e}"})

    # Generate file paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c if c.isalnum() else "_" for c in title.lower())[:40]
    chart_id = f"{task_id}_{safe_title}_{timestamp}"

    html_path = str(OUTPUTS_DIR / f"{chart_id}.html")
    png_path = str(OUTPUTS_DIR / f"{chart_id}.png")

    # Save HTML (always works)
    try:
        fig.write_html(html_path, include_plotlyjs="cdn")
    except Exception as e:
        return json.dumps({"success": False, "error": f"HTML save failed: {e}"})

    # Save PNG (requires kaleido — optional)
    png_saved = False
    try:
        fig.write_image(png_path, width=900, height=500, scale=2)
        png_saved = True
    except Exception as png_err:
        logger.warning(
            f"[chart_generator] PNG save failed: {png_err}\n"
            "Install kaleido for PNG export: pip install kaleido"
        )
        png_path = ""

    # Register the chart
    chart_record = {
        "chart_id": chart_id,
        "task_id": task_id,
        "title": title,
        "chart_type": chart_type,
        "description": description or title,
        "html_path": html_path,
        "png_path": png_path,
        "png_saved": png_saved,
        "created_at": datetime.now().isoformat(),
        "rows": len(df),
        "columns": list(df.columns),
    }
    _register_chart(chart_record)

    logger.info(f"[chart_generator] Saved '{title}' → {os.path.basename(html_path)}")

    return json.dumps(
        {
            "success": True,
            "chart_id": chart_id,
            "html_path": html_path,
            "png_path": png_path,
            "png_saved": png_saved,
            "title": title,
            "rows_plotted": len(df),
            "message": (
                f"Chart '{title}' created. HTML: {os.path.basename(html_path)}"
                + (
                    f", PNG: {os.path.basename(png_path)}"
                    if png_saved
                    else " (PNG skipped — run: pip install kaleido)"
                )
            ),
        }
    )


def _parse_to_dataframe(data: Union[str, list, dict]) -> "pd.DataFrame":
    """Convert JSON string, list-of-dicts, or dict-of-lists to DataFrame."""
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, list):
        return pd.DataFrame(data)
    elif isinstance(data, dict):
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


def _build_chart(
    df, chart_type, x_column, y_column, title, x_label, y_label, color_column
) -> "go.Figure":
    """Build and style a Plotly figure based on chart type."""
    kwargs = dict(
        x=x_column,
        title=title,
        color=color_column,
        color_discrete_sequence=COLOUR_PALETTE,
    )

    if chart_type == "bar":
        fig = px.bar(
            df, y=y_column, labels={x_column: x_label, y_column: y_label}, **kwargs
        )

    elif chart_type == "line":
        fig = px.line(
            df,
            y=y_column,
            markers=True,
            labels={x_column: x_label, y_column: y_label},
            **kwargs,
        )

    elif chart_type == "scatter":
        fig = px.scatter(
            df, y=y_column, labels={x_column: x_label, y_column: y_label}, **kwargs
        )

    elif chart_type == "pie":
        fig = px.pie(
            df,
            names=x_column,
            values=y_column,
            title=title,
            color_discrete_sequence=COLOUR_PALETTE,
        )

    elif chart_type == "histogram":
        fig = px.histogram(
            df,
            x=x_column,
            title=title,
            labels={x_column: x_label},
            color_discrete_sequence=COLOUR_PALETTE,
        )

    elif chart_type == "heatmap":
        numeric_df = df.select_dtypes(include="number")
        fig = px.imshow(
            numeric_df.corr() if len(numeric_df.columns) > 1 else numeric_df,
            title=title,
            color_continuous_scale="Blues",
        )

    else:
        raise ValueError(
            f"Unknown chart type '{chart_type}'. "
            "Valid options: bar, line, scatter, pie, histogram, heatmap"
        )

    # Consistent professional styling
    fig.update_layout(
        title=dict(font=dict(size=18, color="#1F2937"), x=0.05),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, Arial, sans-serif", size=13, color="#374151"),
        margin=dict(l=60, r=40, t=70, b=60),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        xaxis=dict(gridcolor="#F3F4F6", linecolor="#E5E7EB"),
        yaxis=dict(gridcolor="#F3F4F6", linecolor="#E5E7EB"),
    )

    return fig


def _register_chart(record: dict) -> None:
    """Append a chart record to the JSON registry file."""
    registry = []
    if CHART_REGISTRY_PATH.exists():
        try:
            with open(CHART_REGISTRY_PATH, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception:
            registry = []

    registry.append(record)

    with open(CHART_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def get_charts_for_task(task_id: str) -> list[dict]:
    """
    Get all charts generated for a specific task.
    Used by the Writer agent and Streamlit dashboard.
    """
    if not CHART_REGISTRY_PATH.exists():
        return []
    try:
        with open(CHART_REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = json.load(f)
        return [c for c in registry if c.get("task_id") == task_id]
    except Exception:
        return []


if __name__ == "__main__":
    from rich import print as rprint

    rprint("[bold cyan]Testing Chart Generator Tool...[/bold cyan]")

    test_task = "test_charts_001"

    rprint("\n[yellow]Test 1: Bar chart — solar energy by country[/yellow]")
    solar_data = json.dumps(
        [
            {"country": "China", "solar_gw": 430},
            {"country": "USA", "solar_gw": 140},
            {"country": "India", "solar_gw": 80},
            {"country": "Japan", "solar_gw": 84},
            {"country": "Germany", "solar_gw": 67},
            {"country": "Brazil", "solar_gw": 40},
        ]
    )
    result = generate_chart(
        data=solar_data,
        chart_type="bar",
        x_column="country",
        y_column="solar_gw",
        title="Top Countries by Solar Energy Capacity (GW)",
        task_id=test_task,
        x_label="Country",
        y_label="Capacity (GW)",
    )
    r = json.loads(result)
    rprint(f"Success: {r['success']}")
    if r["success"]:
        rprint(f"Message: {r['message']}")

    rprint("\n[yellow]Test 2: Line chart — global growth over time[/yellow]")
    timeline_data = json.dumps(
        [
            {"year": 2018, "gw": 480},
            {"year": 2019, "gw": 627},
            {"year": 2020, "gw": 714},
            {"year": 2021, "gw": 849},
            {"year": 2022, "gw": 1053},
            {"year": 2023, "gw": 1600},
        ]
    )
    result2 = generate_chart(
        data=timeline_data,
        chart_type="line",
        x_column="year",
        y_column="gw",
        title="Global Solar Capacity Growth (GW)",
        task_id=test_task,
    )
    r2 = json.loads(result2)
    rprint(f"Success: {r2['success']}")

    rprint("\n[yellow]Test 3: Pie chart — regional share[/yellow]")
    region_data = json.dumps(
        [
            {"region": "Asia Pacific", "share": 61},
            {"region": "Europe", "share": 18},
            {"region": "Americas", "share": 16},
            {"region": "Rest", "share": 5},
        ]
    )
    result3 = generate_chart(
        data=region_data,
        chart_type="pie",
        x_column="region",
        y_column="share",
        title="Solar Capacity by Region (2023)",
        task_id=test_task,
    )
    r3 = json.loads(result3)
    rprint(f"Success: {r3['success']}")

    rprint("\n[yellow]Test 4: Chart registry[/yellow]")
    charts = get_charts_for_task(test_task)
    rprint(f"Charts registered: {len(charts)}")
    for c in charts:
        rprint(f"  [{c['chart_type']}] {c['title']}")

    rprint("[bold green]✓ Chart generator test complete![/bold green]")
