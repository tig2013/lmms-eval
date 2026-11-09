"""Hermetic CPU unit tests for the olmOCR-Bench benchmark task."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import datasets
import pytest
from PIL import Image

from lmms_eval.api.task import ConfigurableMessagesTask
from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.olmocr_bench import utils
from lmms_eval.utils import load_yaml_config

TASK_DIR = Path(__file__).resolve().parents[2] / "lmms_eval" / "tasks" / "olmocr_bench"
YAML_PATH = TASK_DIR / "olmocr_bench.yaml"


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 12), color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_yaml_config_and_task_discovery() -> None:
    cfg = load_yaml_config(str(YAML_PATH))
    assert cfg["task"] == "olmocr_bench"
    assert cfg["dataset_path"] == "allenai/olmOCR-bench"
    assert cfg["dataset_kwargs"]["revision"] == "54a96a6fb6a2bd3b297e59869491db4d3625b711"
    assert cfg["dataset_kwargs"]["data_files"] == {"test": "bench_data/arxiv_math.jsonl"}
    assert cfg["test_split"] == "test"
    assert cfg["output_type"] == "generate_until"
    assert cfg["process_docs"].__name__ == "olmocr_bench_process_docs"
    assert cfg["doc_to_visual"].__name__ == "olmocr_bench_doc_to_visual"
    assert cfg["doc_to_text"].__name__ == "olmocr_bench_doc_to_text"
    assert cfg["process_results"].__name__ == "olmocr_bench_process_results"

    metric_names = [m["metric"] for m in cfg["metric_list"]]
    assert metric_names == list(utils.ALL_METRICS)

    tm = TaskManager("ERROR")
    assert "olmocr_bench" in tm.all_tasks


def test_process_docs_groups_and_interleaves_categories() -> None:
    raw_ds = datasets.Dataset.from_list(
        [
            {
                "id": "m1",
                "pdf": "arxiv_math/p1.pdf",
                "page": 1,
                "type": "math",
                "math": "a + b = c",
            },
            {
                "id": "m2",
                "pdf": "arxiv_math/p1.pdf",
                "page": 1,
                "type": "math",
                "math": "x^2 + y^2 = z^2",
            },
            {
                "id": "t1",
                "pdf": "tables/t1.pdf",
                "page": 1,
                "type": "table",
                "cell": "100",
                "top_heading": "Jan",
            },
            {
                "id": "hf1",
                "pdf": "headers_footers/h1.pdf",
                "page": 1,
                "type": "absent",
                "text": "Page 12",
                "first_n": 50,
            },
            {
                "id": "hf_base",
                "pdf": "headers_footers/h1.pdf",
                "page": 1,
                "type": "baseline",
                "check_disallowed_characters": False,
            },
        ],
    )

    grouped = utils.olmocr_bench_process_docs(raw_ds)
    assert len(grouped) == 3
    assert [row["category"] for row in grouped] == [
        "arxiv_math",
        "table_tests",
        "headers_footers",
    ]

    arxiv_tests = json.loads(grouped[0]["tests_json"])
    assert len(arxiv_tests) == 3
    assert [t["type"] for t in arxiv_tests] == ["math", "math", "baseline"]
    assert [t["group"] for t in arxiv_tests] == ["arxiv_math", "arxiv_math", "baseline"]

    hf_tests = json.loads(grouped[2]["tests_json"])
    # Explicit baseline test in headers_footers.jsonl stays in headers_footers group; no duplicate auto-baseline
    assert len(hf_tests) == 2
    assert [t["type"] for t in hf_tests] == ["absent", "baseline"]
    assert [t["group"] for t in hf_tests] == ["headers_footers", "headers_footers"]


def test_doc_to_visual_text_and_messages_parity(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample_page.pdf"
    Image.new("RGB", (200, 100), color=(20, 40, 60)).save(pdf_path, format="PDF", resolution=72.0)

    doc = {
        "pdf": "arxiv_math/sample_page.pdf",
        "page": 1,
        "pdf_path": str(pdf_path),
        "category": "arxiv_math",
        "tests_json": json.dumps([{"id": "b1", "pdf": "p.pdf", "page": 1, "type": "baseline"}]),
    }

    visuals = utils.olmocr_bench_doc_to_visual(doc)
    assert len(visuals) == 1
    assert isinstance(visuals[0], Image.Image)
    assert visuals[0].mode == "RGB"
    assert max(visuals[0].size) == pytest.approx(2048, abs=4)

    # Also test direct image dict / bytes fallback
    img_doc = {"image": {"bytes": _png_bytes((10, 20, 30)), "path": "inline.png"}}
    inline_vis = utils.olmocr_bench_doc_to_visual(img_doc)
    assert len(inline_vis) == 1 and inline_vis[0].getpixel((0, 0)) == (10, 20, 30)

    default_prompt = utils.olmocr_bench_doc_to_text(doc)
    assert "Below is the image of one page of a PDF document." in default_prompt
    basic_prompt = utils.olmocr_bench_doc_to_text(
        doc,
        {"prompt_template": "basic", "pre_prompt": "[PRE] ", "post_prompt": " [POST]"},
    )
    assert basic_prompt.startswith("[PRE] Please provide a natural, plain text representation")
    assert basic_prompt.endswith("Convert any tables into Markdown format. [POST]")

    with pytest.raises(ValueError, match="unknown prompt_template"):
        utils.olmocr_bench_doc_to_text(doc, {"prompt_template": "nonexistent"})

    task = ConfigurableMessagesTask.__new__(ConfigurableMessagesTask)
    task._config = type(
        "Cfg",
        (),
        {
            "doc_to_messages": None,
            "doc_to_visual": utils.olmocr_bench_doc_to_visual,
            "doc_to_text": utils.olmocr_bench_doc_to_text,
        },
    )()
    task.doc_to_visual = utils.olmocr_bench_doc_to_visual
    task.doc_to_text = lambda d: utils.olmocr_bench_doc_to_text(d)
    messages = task.doc_to_messages(img_doc)

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert [item["type"] for item in content] == ["image", "text"]
    assert content[1]["text"] == default_prompt


def test_text_presence_and_order_evaluators() -> None:
    present_test = {
        "id": "p1",
        "pdf": "doc.pdf",
        "page": 1,
        "type": "present",
        "text": "The quick “brown” fox — jumps",
        "case_sensitive": True,
        "max_diffs": 1,
    }
    assert utils.run_single_pdf_test(
        present_test,
        'Header\nThe **quick** "brown" fox - jumps over the lazy dog.',
    )
    assert not utils.run_single_pdf_test(
        present_test,
        'the QUICK "brown" fox - jumps',
    )
    assert utils.run_single_pdf_test(
        {**present_test, "case_sensitive": False},
        'the QUICK "brown" fox - jumps',
    )

    absent_header = {
        "id": "a1",
        "pdf": "doc.pdf",
        "page": 1,
        "type": "absent",
        "text": "CONFIDENTIAL HEADER",
        "first_n": 30,
        "max_diffs": 0,
    }
    assert utils.run_single_pdf_test(
        absent_header,
        "Clean start of body text that mentions CONFIDENTIAL HEADER much later in the document.",
    )
    assert not utils.run_single_pdf_test(
        absent_header,
        "CONFIDENTIAL HEADER\nClean start of body text.",
    )

    order_test = {
        "id": "o1",
        "pdf": "doc.pdf",
        "page": 1,
        "type": "order",
        "before": "First column paragraph alpha",
        "after": "Second column paragraph beta",
        "max_diffs": 2,
    }
    assert utils.run_single_pdf_test(
        order_test,
        "Intro. First column paragrph alpha comes here, then Second column paragraph beta.",
    )
    assert not utils.run_single_pdf_test(
        order_test,
        "Second column paragraph beta appears before First column paragraph alpha.",
    )


def test_table_parsing_and_table_test_evaluator() -> None:
    md_table = "| Metric | Q1 | Q2 |\n| --- | --- | --- |\n| Revenue | 120 | 150 |\n| Profit | 40 | 55 |\n"
    md_test = {
        "id": "tb_md",
        "pdf": "t.pdf",
        "page": 1,
        "type": "table",
        "cell": "150",
        "left": "120",
        "down": "55",
        "top_heading": "Q2",
        "left_heading": "Revenue",
    }
    assert utils.run_single_pdf_test(md_test, md_table)
    assert not utils.run_single_pdf_test({**md_test, "top_heading": "Q1"}, md_table)

    html_table = """
    <table>
      <thead>
        <tr>
          <th rowspan="2"></th>
          <th colspan="2">Main Group</th>
        </tr>
        <tr>
          <th>Col A</th>
          <th>Col B</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Row 1</th>
          <td>10<br>units</td>
          <td>20</td>
        </tr>
      </tbody>
    </table>
    """
    html_test = {
        "id": "tb_html",
        "pdf": "t.pdf",
        "page": 1,
        "type": "table",
        "cell": "20",
        "left": "10 units",
        "top_heading": "Main Group",
        "left_heading": "Row 1",
    }
    assert utils.run_single_pdf_test(html_test, html_table)


@pytest.mark.parametrize(
    ("ref_math", "md_content", "expected"),
    [
        ("a+b", r"Inline \(a + b\) equation", True),
        ("c-d", r"Inline \(a + b\) equation", False),
        (
            r"\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}",
            r"$$\begin{align*}\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}\end{align*}$$",
            True,
        ),
        (
            r"\left\{ \left( 0_{X},0_{Y},-1\right) \right\}",
            r"\[ \{(0_{X}, 0_{Y}, -1)\} \]",
            True,
        ),
        (
            r"\lambda_g = \lambda_{g'}",
            r"$\lambda_{g}=\lambda_{g^{\prime}}$",
            True,
        ),
        (
            r"T_{ij}^{kl}",
            r"$T^{kl}_{ij}$",
            True,
        ),
        (
            r"C_T \left(u^T_n X^{\text{Test}}_n,\overline{ x}^{\text{Test}}\right)",
            r"$$C_{T}\left(u_{n}^{T} X_{n}^{\text{Test }}, \overline{x}^{\text {Test }}\right)$$",
            True,
        ),
        (
            r"C_{T}\left(u_{n}^{T} X_{n}^{\text {Test }}, \bar{x}^{\text {Test }}\right)",
            r"$$C_T \left(u^T_n X^{\text{Test}}_n,\overline{ \text{x}}^{\text{Test}}\right)$$",
            False,
        ),
        (r"\frac{a}{b}", r"$a / b$", False),
        (r"\frac{\partial f}{\partial x}", r"$\frac{\partial_f}{\partial_x}$", False),
        (r"x_{i,j}", r"$x_{j,i}$", False),
        (r"\phi", r"$\varphi$", False),
        (r"a + b", r"$\color{red}{a} + \color{black}{b}$", True),
        (r"\alpha + \beta", r"$\newcommand{\ab}{\alpha + \beta}\ab$", True),
        (r"\to", r"$\rightarrow$", True),
    ],
)
def test_math_test_evaluator_katex_suite(ref_math: str, md_content: str, expected: bool) -> None:
    test_obj = {"id": "m_case", "pdf": "m.pdf", "page": 1, "type": "math", "math": ref_math}
    assert utils.run_single_pdf_test(test_obj, md_content) is expected


def test_baseline_test_and_prediction_cleanup() -> None:
    baseline = {"id": "b1", "pdf": "p.pdf", "page": 1, "type": "baseline"}
    assert utils.run_single_pdf_test(baseline, "Valid English paragraph with 123 numbers.")
    assert not utils.run_single_pdf_test(baseline, "   \n... !@# ")
    assert not utils.run_single_pdf_test(baseline, "Valid start " + ("abc" * 35))
    assert not utils.run_single_pdf_test(baseline, "English text with CJK 漢字")
    assert not utils.run_single_pdf_test(baseline, "English text with emoji 😀")
    assert utils.run_single_pdf_test(
        {**baseline, "check_disallowed_characters": False},
        "Allowed CJK 漢字 when check_disallowed_characters is False",
    )

    blank_check = {
        "id": "b_blank",
        "pdf": "p.pdf",
        "page": 1,
        "type": "baseline",
        "max_length": 5,
        "max_length_skips_image_alt_tags": True,
    }
    assert utils.run_single_pdf_test(blank_check, "![diagram description](page_0_0_100_100.png)")
    assert not utils.run_single_pdf_test(blank_check, "Non-blank paragraph text")

    # Prediction cleanup: <think>, YAML front matter, markdown fence, JSON PageResponse, null, anti-eval
    raw_yaml = "\n".join(
        [
            "<think>CONFIDENTIAL HEADER should be omitted</think>",
            "```markdown",
            "---",
            "primary_language: en",
            "is_rotation_valid: true",
            "rotation_correction: 0",
            "is_table: false",
            "is_diagram: false",
            "---",
            "Clean body paragraph.",
            "```",
        ],
    )
    assert utils.clean_model_prediction(raw_yaml) == "Clean body paragraph."
    assert utils.clean_model_prediction('{"natural_text": "From JSON schema"}') == "From JSON schema"
    assert utils.clean_model_prediction("null") == ""
    adv_payload = "__import__('os').system('echo unsafe')"
    assert utils.clean_model_prediction(adv_payload) == adv_payload


def test_process_results_json_serializable_and_macro_aggregation() -> None:
    doc_math = {
        "pdf": "arxiv_math/m1.pdf",
        "page": 1,
        "category": "arxiv_math",
        "tests_json": json.dumps(
            [
                {
                    "id": "m1",
                    "pdf": "arxiv_math/m1.pdf",
                    "page": 1,
                    "type": "math",
                    "math": "E = mc^2",
                    "group": "arxiv_math",
                },
                {
                    "id": "m2",
                    "pdf": "arxiv_math/m1.pdf",
                    "page": 1,
                    "type": "math",
                    "math": "a^2 + b^2 = c^2",
                    "group": "arxiv_math",
                },
                {
                    "id": "b1",
                    "pdf": "arxiv_math/m1.pdf",
                    "page": 1,
                    "type": "baseline",
                    "group": "baseline",
                },
            ],
        ),
    }
    doc_table = {
        "pdf": "tables/t1.pdf",
        "page": 1,
        "category": "table_tests",
        "tests_json": json.dumps(
            [
                {
                    "id": "t1",
                    "pdf": "tables/t1.pdf",
                    "page": 1,
                    "type": "table",
                    "cell": "42",
                    "top_heading": "Score",
                    "group": "table_tests",
                },
                {
                    "id": "b2",
                    "pdf": "tables/t1.pdf",
                    "page": 1,
                    "type": "baseline",
                    "group": "baseline",
                },
            ],
        ),
    }

    res_math = utils.olmocr_bench_process_results(doc_math, [r"Physics formula: \(E = mc^2\)"])
    res_table = utils.olmocr_bench_process_results(doc_table, ["| Score |\n| --- |\n| 42 |"])

    # Strictly JSON-serializable for 2-process accelerate gather
    json.dumps(res_math)
    json.dumps(res_table)

    assert res_math["olmocr_bench_arxiv_math"] == {"passed": 1, "total": 2}
    assert res_math["olmocr_bench_baseline"] == {"passed": 1, "total": 1}
    assert res_math["olmocr_bench_table_tests"] is None

    assert res_table["olmocr_bench_table_tests"] == {"passed": 1, "total": 1}
    assert res_table["olmocr_bench_baseline"] == {"passed": 1, "total": 1}

    overall_records = [res_math["olmocr_bench_overall"], res_table["olmocr_bench_overall"]]
    assert utils.olmocr_bench_aggregate_arxiv_math(
        [res_math["olmocr_bench_arxiv_math"], res_table["olmocr_bench_arxiv_math"]],
    ) == pytest.approx(0.5)
    assert utils.olmocr_bench_aggregate_table_tests(
        [res_math["olmocr_bench_table_tests"], res_table["olmocr_bench_table_tests"]],
    ) == pytest.approx(1.0)
    assert utils.olmocr_bench_aggregate_baseline(
        [res_math["olmocr_bench_baseline"], res_table["olmocr_bench_baseline"]],
    ) == pytest.approx(1.0)
    assert math.isnan(
        utils.olmocr_bench_aggregate_old_scans(
            [res_math["olmocr_bench_old_scans"], res_table["olmocr_bench_old_scans"]],
        ),
    )
    # Macro-average over active categories: (0.5 + 1.0 + 1.0) / 3 = 5/6
    assert utils.olmocr_bench_aggregate_overall(overall_records) == pytest.approx(5.0 / 6.0)
    assert math.isnan(utils.olmocr_bench_aggregate_overall([]))
