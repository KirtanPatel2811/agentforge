"""
tests/test_phase2_tools.py — Phase 2 Tool Tests
-------------------------------------------------
Offline tests: python -m pytest tests/test_phase2_tools.py -v -m "not live"
Live tests:    python -m pytest tests/test_phase2_tools.py -v -m live
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Code Executor Tests (fully offline) ──────────────────────────────────────


class TestCodeExecutor:

    def test_simple_code_runs(self):
        from src.tools.code_executor import execute_code

        result = json.loads(execute_code("print('hello agentforge')"))
        assert result["success"] is True
        assert "hello agentforge" in result["stdout"]

    def test_math_computation(self):
        from src.tools.code_executor import execute_code

        result = json.loads(
            execute_code(
                """
import math
result = math.sqrt(144)
print(f"sqrt(144) = {result}")
"""
            )
        )
        assert result["success"] is True
        assert "12.0" in result["stdout"]

    def test_pandas_code(self):
        from src.tools.code_executor import execute_code

        result = json.loads(
            execute_code(
                """
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
print(df.sum().to_dict())
"""
            )
        )
        assert result["success"] is True
        assert result["stdout"].strip() != ""

    def test_syntax_error_caught(self):
        from src.tools.code_executor import execute_code

        result = json.loads(execute_code("def broken(:"))
        assert result["success"] is False
        assert result["error"] != ""

    def test_runtime_error_caught(self):
        from src.tools.code_executor import execute_code

        result = json.loads(execute_code("x = 1 / 0"))
        assert result["success"] is False
        assert (
            "ZeroDivisionError" in result["error"]
            or "division" in result["error"].lower()
        )

    def test_timeout_enforced(self):
        from src.tools.code_executor import execute_code

        result = json.loads(execute_code("while True: pass", timeout=2))
        assert result["success"] is False
        assert (
            "timed out" in result["error"].lower()
            or "timeout" in result["error"].lower()
        )

    def test_stdout_captured(self):
        from src.tools.code_executor import execute_code

        result = json.loads(
            execute_code(
                """
for i in range(3):
    print(f"line {i}")
"""
            )
        )
        assert result["success"] is True
        assert "line 0" in result["stdout"]
        assert "line 2" in result["stdout"]

    def test_numpy_available(self):
        from src.tools.code_executor import execute_code

        result = json.loads(
            execute_code(
                """
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f"mean={arr.mean()}, std={arr.std():.2f}")
"""
            )
        )
        assert result["success"] is True
        assert "mean=3.0" in result["stdout"]

    def test_format_result_success(self):
        from src.tools.code_executor import format_execution_result, execute_code

        raw = execute_code("print('test output')")
        formatted = format_execution_result(raw)
        assert "successfully" in formatted
        assert "test output" in formatted

    def test_format_result_failure(self):
        from src.tools.code_executor import format_execution_result, execute_code

        raw = execute_code("raise ValueError('test error')")
        formatted = format_execution_result(raw)
        assert "failed" in formatted.lower() or "Error" in formatted


# ── Chart Generator Tests (offline) ──────────────────────────────────────────


class TestChartGenerator:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        import src.tools.chart_generator as cg

        monkeypatch.setattr(cg, "OUTPUTS_DIR", tmp_path)
        monkeypatch.setattr(cg, "CHART_REGISTRY_PATH", tmp_path / "chart_registry.json")

    def _solar_data(self):
        return json.dumps(
            [
                {"country": "China", "solar_gw": 430},
                {"country": "USA", "solar_gw": 140},
                {"country": "India", "solar_gw": 80},
                {"country": "Germany", "solar_gw": 67},
            ]
        )

    def test_bar_chart_created(self):
        from src.tools.chart_generator import generate_chart

        result = json.loads(
            generate_chart(
                data=self._solar_data(),
                chart_type="bar",
                x_column="country",
                y_column="solar_gw",
                title="Test Bar Chart",
                task_id="test001",
            )
        )
        assert result["success"] is True
        assert os.path.exists(result["html_path"])

    def test_line_chart_created(self):
        from src.tools.chart_generator import generate_chart

        data = json.dumps([{"year": 2020 + i, "gw": 700 + i * 150} for i in range(4)])
        result = json.loads(
            generate_chart(
                data=data,
                chart_type="line",
                x_column="year",
                y_column="gw",
                title="Line Test",
                task_id="test001",
            )
        )
        assert result["success"] is True

    def test_scatter_chart_created(self):
        from src.tools.chart_generator import generate_chart

        data = json.dumps([{"x": i, "y": i**2} for i in range(5)])
        result = json.loads(
            generate_chart(
                data=data,
                chart_type="scatter",
                x_column="x",
                y_column="y",
                title="Scatter Test",
                task_id="test001",
            )
        )
        assert result["success"] is True

    def test_pie_chart_created(self):
        from src.tools.chart_generator import generate_chart

        data = json.dumps(
            [
                {"region": "Asia", "pct": 60},
                {"region": "EU", "pct": 20},
                {"region": "Americas", "pct": 20},
            ]
        )
        result = json.loads(
            generate_chart(
                data=data,
                chart_type="pie",
                x_column="region",
                y_column="pct",
                title="Pie Test",
                task_id="test001",
            )
        )
        assert result["success"] is True

    def test_invalid_chart_type(self):
        from src.tools.chart_generator import generate_chart

        result = json.loads(
            generate_chart(
                data=self._solar_data(),
                chart_type="foobar",
                x_column="country",
                y_column="solar_gw",
                title="Bad Type",
                task_id="test001",
            )
        )
        assert result["success"] is False

    def test_missing_column_error(self):
        from src.tools.chart_generator import generate_chart

        result = json.loads(
            generate_chart(
                data=self._solar_data(),
                chart_type="bar",
                x_column="nonexistent",
                y_column="solar_gw",
                title="Bad Column",
                task_id="test001",
            )
        )
        assert result["success"] is False
        assert "nonexistent" in result["error"]

    def test_dict_of_lists_input(self):
        from src.tools.chart_generator import generate_chart

        data = json.dumps({"country": ["A", "B", "C"], "gw": [100, 200, 300]})
        result = json.loads(
            generate_chart(
                data=data,
                chart_type="bar",
                x_column="country",
                y_column="gw",
                title="Dict Input Test",
                task_id="test001",
            )
        )
        assert result["success"] is True

    def test_chart_registry(self):
        from src.tools.chart_generator import generate_chart, get_charts_for_task

        generate_chart(
            data=self._solar_data(),
            chart_type="bar",
            x_column="country",
            y_column="solar_gw",
            title="Registry Test",
            task_id="registry_task",
        )
        charts = get_charts_for_task("registry_task")
        assert len(charts) >= 1
        assert charts[0]["title"] == "Registry Test"

    def test_empty_data_error(self):
        from src.tools.chart_generator import generate_chart

        result = json.loads(
            generate_chart(
                data=json.dumps([]),
                chart_type="bar",
                x_column="x",
                y_column="y",
                title="Empty",
                task_id="test001",
            )
        )
        assert result["success"] is False


# ── URL Reader Offline Tests ──────────────────────────────────────────────────


class TestURLReaderOffline:

    def test_content_extraction_from_html(self):
        from src.tools.url_reader import _extract_content
        from bs4 import BeautifulSoup

        html = """
        <html><body>
        <nav>Nav noise</nav>
        <article>
            <h1>Solar Energy</h1>
            <p>China leads with 430 GW installed.</p>
            <p>USA follows with 140 GW of solar installations.</p>
        </article>
        <footer>Footer noise</footer>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        content = _extract_content(soup)
        assert "Solar Energy" in content or "430 GW" in content

    def test_noise_removal(self):
        from src.tools.url_reader import _extract_content
        from bs4 import BeautifulSoup

        html = """
        <html><body>
        <nav>THIS IS NAV NOISE XYZNOISE</nav>
        <article><p>This is the real article content about solar panels.</p></article>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        for tag in soup.find_all("nav"):
            tag.decompose()
        content = _extract_content(soup)
        assert "XYZNOISE" not in content
        assert "solar panels" in content


# ── Live Tests (need internet) ────────────────────────────────────────────────
# Run with: python -m pytest tests/test_phase2_tools.py -v -m live


@pytest.mark.live
class TestWebSearchLive:

    def test_web_search_returns_results(self):
        from src.tools.web_search import web_search

        result = web_search("Python programming language", max_results=3)
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "title" in data[0]
        assert "url" in data[0]

    def test_news_search_returns_results(self):
        from src.tools.web_search import news_search

        result = news_search("artificial intelligence", max_results=3)
        data = json.loads(result)
        assert isinstance(data, list)


@pytest.mark.live
class TestArxivSearchLive:

    def test_arxiv_search_returns_papers(self):
        from src.tools.arxiv_search import arxiv_search

        result = arxiv_search("transformer attention mechanism", max_results=3)
        papers = json.loads(result)
        assert isinstance(papers, list)
        assert len(papers) >= 1
        assert "title" in papers[0]
        assert "pdf_url" in papers[0]

    def test_arxiv_get_known_paper(self):
        from src.tools.arxiv_search import arxiv_get_paper

        result = arxiv_get_paper("1706.03762")
        paper = json.loads(result)
        assert "Attention" in paper["title"]


@pytest.mark.live
class TestURLReaderLive:

    def test_read_wikipedia_page(self):
        from src.tools.url_reader import read_url

        result = read_url(
            "https://en.wikipedia.org/wiki/Python_(programming_language)", max_chars=500
        )
        assert "Python" in result
        assert len(result) > 100

    def test_handles_bad_url(self):
        from src.tools.url_reader import read_url

        result = read_url("https://this-domain-absolutely-does-not-exist-xyz123.com")
        assert "Error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not live"])
