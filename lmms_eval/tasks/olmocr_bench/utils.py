"""olmOCR-Bench evaluation utilities.

Implements the official prompt, PDF page renderer, unit-test suite
(``present``, ``absent``, ``order``, ``table``, ``math``, ``baseline``), and
macro-averaged category scorer from ``https://github.com/allenai/olmocr``
(``olmocr/bench``) over ``allenai/olmOCR-bench``.
"""

import io
import json
import math
import re
import shutil
import subprocess
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import datasets
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image
from rapidfuzz import fuzz

from lmms_eval.utils import eval_logger

try:
    from fuzzysearch import find_near_matches as _ext_find_near_matches
except ImportError:
    _ext_find_near_matches = None

try:
    from bs4 import BeautifulSoup as _BeautifulSoup
except ImportError:
    _BeautifulSoup = None

DATASET_REPO_ID = "allenai/olmOCR-bench"
DATASET_REVISION = "54a96a6fb6a2bd3b297e59869491db4d3625b711"
TARGET_LONGEST_IMAGE_DIM = 2048

SOURCE_CATEGORIES: tuple[str, ...] = (
    "arxiv_math",
    "old_scans_math",
    "table_tests",
    "old_scans",
    "headers_footers",
    "multi_column",
    "long_tiny_text",
)
ALL_CATEGORIES: tuple[str, ...] = (*SOURCE_CATEGORIES, "baseline")
ALL_METRICS: tuple[str, ...] = (
    "olmocr_bench_overall",
    *(f"olmocr_bench_{category}" for category in ALL_CATEGORIES),
)

PROMPT_FULL_NO_DOCUMENT_ANCHORING = (
    "Below is the image of one page of a PDF document. "
    "Just return the plain text representation of this document as if you were reading it naturally.\n"
    "Turn equations into a LaTeX representation, and tables into markdown format. "
    "Remove the headers and footers, but keep references and footnotes.\n"
    "Read any natural handwriting.\n"
    "This is likely one page out of several in the document, so be sure to preserve any sentences "
    "that come from the previous page, or continue onto the next page, exactly as they are.\n"
    "If there is no text at all that you think you should read, you can output null.\n"
    "Do not hallucinate."
)

PROMPT_BASIC = (
    "Please provide a natural, plain text representation of the document, formatted in Markdown. "
    "Skip any headers and footers. For ALL mathematical expressions, use LaTeX notation with "
    r"\( and \) for inline equations and \[ and \] for display equations. "
    "Convert any tables into Markdown format."
)

PROMPT_YAML = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it naturally. "
    "Convert equations to LateX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following markdown syntax "
    "![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\n"
    "Return your output as markdown, with a front matter section on top specifying values for the "
    "primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."
)

PROMPT_TEMPLATES: dict[str, str] = {
    "full_no_document_anchoring": PROMPT_FULL_NO_DOCUMENT_ANCHORING,
    "basic": PROMPT_BASIC,
    "yaml": PROMPT_YAML,
}

_TEXT_REPLACEMENTS: dict[str, str] = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "＿": "_",
    "–": "-",
    "—": "-",
    "‑": "-",
    "‒": "-",
    "−": "-",
    "\u00b5": "\u03bc",
}

_DISALLOWED_CHARS_RE = re.compile(
    r"["
    r"\u4e00-\u9FFF"
    r"\u3040-\u309F"
    r"\u30A0-\u30FF"
    r"\U0001F600-\U0001F64F"
    r"\U0001F300-\U0001F5FF"
    r"\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF"
    r"]",
    flags=re.UNICODE,
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_OUTER_FENCE_RE = re.compile(
    r"^\s*```(?:markdown|md|html|latex|text)?\s*\n(.*?)\n?\s*```\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)


# Text normalization & prediction cleanup


def normalize_text(md_content: str | None) -> str:
    """Official ``normalize_text`` from ``olmocr/bench/tests.py:47-80``."""
    if md_content is None:
        return ""
    text = str(md_content)
    text = re.sub(r"<br/?>", " ", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"</?b>", "", text)
    text = re.sub(r"</?i>", "", text)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
    text = re.sub(r"\s+", " ", text)
    text = unicodedata.normalize("NFC", text)
    for fancy_char, ascii_char in _TEXT_REPLACEMENTS.items():
        text = text.replace(fancy_char, ascii_char)
    return text


def _strip_yaml_front_matter(markdown_content: str) -> str:
    """Extract body text when output starts with olmOCR YAML front matter."""
    if not markdown_content.startswith("---\n"):
        return markdown_content
    end_index = markdown_content.find("\n---", 4)
    if end_index == -1:
        return markdown_content
    front_matter_str = markdown_content[4:end_index]
    body = markdown_content[end_index + 4 :].lstrip("\r\n")
    if body.startswith("---"):
        body = body[3:].lstrip("\r\n")
    try:
        parsed = yaml.safe_load(front_matter_str)
    except yaml.YAMLError:
        return markdown_content
    if isinstance(parsed, dict):
        if not body.strip() and "natural_text" in parsed:
            nat = parsed.get("natural_text")
            return "" if nat is None else str(nat)
        return body.strip()
    return markdown_content


def clean_model_prediction(raw_prediction: str | None) -> str:
    """Clean raw VLM output into page Markdown before running olmOCR-Bench unit tests."""
    if not raw_prediction:
        return ""
    text = _THINK_BLOCK_RE.sub("", str(raw_prediction))
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = text.strip()

    fence_match = _OUTER_FENCE_RE.match(text)
    if fence_match:
        text = fence_match.group(1).strip()

    if text.startswith("{") and text.endswith("}"):
        try:
            payload = json.loads(text)
            if isinstance(payload, dict) and "natural_text" in payload:
                nat = payload.get("natural_text")
                text = "" if nat is None else str(nat).strip()
        except json.JSONDecodeError:
            pass

    text = _strip_yaml_front_matter(text)
    if text.strip().lower() == "null":
        return ""
    return text


# Near-match search (fuzzysearch parity)


@dataclass(frozen=True)
class NearMatch:
    start: int
    end: int
    dist: int
    matched: str


def _find_exact_matches(subsequence: str, sequence: str) -> list[NearMatch]:
    matches: list[NearMatch] = []
    q_len = len(subsequence)
    if q_len == 0:
        return matches
    pos = sequence.find(subsequence)
    while pos != -1:
        matches.append(NearMatch(start=pos, end=pos + q_len, dist=0, matched=subsequence))
        pos = sequence.find(subsequence, pos + 1)
    return matches


def _consolidate_overlapping_matches(matches: list[NearMatch]) -> list[NearMatch]:
    """Group overlapping matches and retain the minimum-distance match per group."""
    if not matches:
        return []
    ordered = sorted(matches, key=lambda m: (m.start, m.end))
    consolidated: list[NearMatch] = []
    group: list[NearMatch] = [ordered[0]]
    group_end = ordered[0].end
    for match in ordered[1:]:
        if match.start < group_end:
            group.append(match)
            group_end = max(group_end, match.end)
        else:
            consolidated.append(min(group, key=lambda m: (m.dist, -(m.end - m.start), m.start)))
            group = [match]
            group_end = match.end
    consolidated.append(min(group, key=lambda m: (m.dist, -(m.end - m.start), m.start)))
    return consolidated


def _seller_near_matches(subsequence: str, sequence: str, max_l_dist: int) -> list[NearMatch]:
    """Pure-Python Seller's semi-global Levenshtein search with start traceback."""
    m = len(subsequence)
    n = len(sequence)
    if m == 0 or n == 0:
        return []
    prev_dist = list(range(m + 1))
    prev_start = [0] * (m + 1)
    raw_matches: list[NearMatch] = []

    for j in range(1, n + 1):
        seq_char = sequence[j - 1]
        curr_dist = [0] + [0] * m
        curr_start = [j - 1] + [0] * m
        for i in range(1, m + 1):
            cost = 0 if subsequence[i - 1] == seq_char else 1
            sub_d = prev_dist[i - 1] + cost
            ins_d = curr_dist[i - 1] + 1
            del_d = prev_dist[i] + 1
            best_d = min(sub_d, ins_d, del_d)
            curr_dist[i] = best_d
            if best_d == sub_d:
                curr_start[i] = prev_start[i - 1]
            elif best_d == del_d:
                curr_start[i] = prev_start[i]
            else:
                curr_start[i] = curr_start[i - 1]
        if curr_dist[m] <= max_l_dist:
            start_idx = curr_start[m]
            raw_matches.append(
                NearMatch(
                    start=start_idx,
                    end=j,
                    dist=curr_dist[m],
                    matched=sequence[start_idx:j],
                ),
            )
        prev_dist = curr_dist
        prev_start = curr_start

    return _consolidate_overlapping_matches(raw_matches)


def find_near_matches_compat(subsequence: str, sequence: str, max_l_dist: int = 0) -> list[Any]:
    """Use ``fuzzysearch.find_near_matches`` when installed, else pure-Python Seller's DP."""
    if _ext_find_near_matches is not None:
        return list(_ext_find_near_matches(subsequence, sequence, max_l_dist=max_l_dist))
    if max_l_dist <= 0:
        return _find_exact_matches(subsequence, sequence)
    return _seller_near_matches(subsequence, sequence, max_l_dist)


# Table parsing (ported from olmocr/bench/table_parsing.py)


@dataclass(frozen=True)
class TableData:
    """Graph of table cells and neighbor/heading relations from ``table_parsing.py:9-68``."""

    cell_text: dict[tuple[int, int], str]
    heading_cells: set[tuple[int, int]]
    is_rectangular: bool
    up_relations: dict[tuple[int, int], set[tuple[int, int]]]
    down_relations: dict[tuple[int, int], set[tuple[int, int]]]
    left_relations: dict[tuple[int, int], set[tuple[int, int]]]
    right_relations: dict[tuple[int, int], set[tuple[int, int]]]

    def _walk_heading_relations(
        self,
        start: tuple[int, int],
        relation: dict[tuple[int, int], set[tuple[int, int]]],
    ) -> set[tuple[int, int]]:
        resulting_heading_cells: set[tuple[int, int]] = set()
        resulting_end_cells: set[tuple[int, int]] = set()
        visited: set[tuple[int, int]] = set()
        to_visit: set[tuple[int, int]] = {start}

        while to_visit:
            cur = to_visit.pop()
            if cur in visited:
                continue
            visited.add(cur)
            if cur in self.heading_cells:
                resulting_heading_cells.add(cur)
            next_nodes = relation.get(cur)
            if not next_nodes:
                resulting_end_cells.add(cur)
            else:
                to_visit |= next_nodes - visited

        resulting_heading_cells.discard(start)
        resulting_end_cells.discard(start)
        return resulting_heading_cells if resulting_heading_cells else resulting_end_cells

    def top_heading_relations(self, start_row: int, start_col: int) -> set[tuple[int, int]]:
        return self._walk_heading_relations((start_row, start_col), self.up_relations)

    def left_heading_relations(self, start_row: int, start_col: int) -> set[tuple[int, int]]:
        return self._walk_heading_relations((start_row, start_col), self.left_relations)


def _safe_span_int(value: str | int | None, default: int = 1) -> int:
    if value in (None, "", 0):
        return default
    try:
        span = int(value)
    except (TypeError, ValueError):
        return default
    return span if span > 0 else default


def _expand_occupancy_grid(
    row_specs: list[list[dict[str, Any]]],
) -> tuple[
    dict[tuple[int, int], str],
    set[tuple[int, int]],
    dict[tuple[int, int], dict[str, int]],
    list[list[tuple[int, int] | None]],
]:
    cell_text: dict[tuple[int, int], str] = {}
    heading_cells: set[tuple[int, int]] = set()
    cell_meta: dict[tuple[int, int], dict[str, int]] = {}
    occupancy: list[list[tuple[int, int] | None]] = []
    active_rowspans: list[tuple[tuple[int, int], int] | None] = []

    for row_idx, cells in enumerate(row_specs):
        row_entries: list[tuple[int, int] | None] = []
        col_index = 0
        spec_idx = 0
        total_specs = len(cells)

        while spec_idx < total_specs or col_index < len(active_rowspans):
            if col_index < len(active_rowspans) and active_rowspans[col_index] is not None:
                span_entry = active_rowspans[col_index]
                assert span_entry is not None
                cell_id, remaining = span_entry
                row_entries.append(cell_id)
                remaining -= 1
                active_rowspans[col_index] = (cell_id, remaining) if remaining > 0 else None
                col_index += 1
                continue

            if spec_idx >= total_specs:
                if col_index < len(active_rowspans):
                    row_entries.append(None)
                    col_index += 1
                    continue
                break

            spec = cells[spec_idx]
            spec_idx += 1
            text = str(spec.get("text", "") or "")
            rowspan = max(1, _safe_span_int(spec.get("rowspan", 1)))
            colspan = max(1, _safe_span_int(spec.get("colspan", 1)))
            is_heading = bool(spec.get("is_heading", False))

            cell_id = (row_idx, col_index)
            cell_text[cell_id] = text
            if is_heading:
                heading_cells.add(cell_id)
            cell_meta[cell_id] = {
                "row": row_idx,
                "col": col_index,
                "rowspan": rowspan,
                "colspan": colspan,
            }

            required_len = col_index + colspan
            if len(active_rowspans) < required_len:
                active_rowspans.extend([None] * (required_len - len(active_rowspans)))

            for offset in range(colspan):
                current_col = col_index + offset
                row_entries.append(cell_id)
                active_rowspans[current_col] = (cell_id, rowspan - 1) if rowspan > 1 else None

            col_index += colspan

        occupancy.append(row_entries)

    while any(entry is not None for entry in active_rowspans):
        row_entries = []
        for col_index, span_entry in enumerate(active_rowspans):
            if span_entry is None:
                row_entries.append(None)
                continue
            cell_id, remaining = span_entry
            row_entries.append(cell_id)
            remaining -= 1
            active_rowspans[col_index] = (cell_id, remaining) if remaining > 0 else None
        occupancy.append(row_entries)

    return cell_text, heading_cells, cell_meta, occupancy


def _build_table_data_from_specs(row_specs: list[list[dict[str, Any]]]) -> TableData | None:
    """Build ``TableData`` from row specifications (``table_parsing.py:83-260``)."""
    if not row_specs:
        return None
    cell_text, heading_cells, cell_meta, occupancy = _expand_occupancy_grid(row_specs)
    if not cell_text:
        return None

    valid_columns = {idx for row in occupancy for idx, val in enumerate(row) if val is not None}
    if not valid_columns:
        return None
    table_width = max(valid_columns) + 1
    for row in occupancy:
        if len(row) < table_width:
            row.extend([None] * (table_width - len(row)))
        elif len(row) > table_width:
            del row[table_width:]

    table_height = len(occupancy)
    up_rel: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    down_rel: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    left_rel: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    right_rel: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)

    for cell_id, meta in cell_meta.items():
        row_start, col_start = meta["row"], meta["col"]
        row_end = row_start + meta["rowspan"] - 1
        col_end = col_start + meta["colspan"] - 1

        for row in range(row_start, row_end + 1):
            for col in range(col_end + 1, table_width):
                neighbor = occupancy[row][col]
                if neighbor is not None and neighbor != cell_id:
                    right_rel[cell_id].add(neighbor)
                    break
            for col in range(col_start - 1, -1, -1):
                neighbor = occupancy[row][col]
                if neighbor is not None and neighbor != cell_id:
                    left_rel[cell_id].add(neighbor)
                    break

        for col in range(col_start, col_end + 1):
            for row in range(row_end + 1, table_height):
                neighbor = occupancy[row][col]
                if neighbor is not None and neighbor != cell_id:
                    down_rel[cell_id].add(neighbor)
                    break
            for row in range(row_start - 1, -1, -1):
                neighbor = occupancy[row][col]
                if neighbor is not None and neighbor != cell_id:
                    up_rel[cell_id].add(neighbor)
                    break

    return TableData(
        cell_text=cell_text,
        heading_cells=heading_cells,
        up_relations={cid: set(up_rel[cid]) for cid in cell_text},
        down_relations={cid: set(down_rel[cid]) for cid in cell_text},
        left_relations={cid: set(left_rel[cid]) for cid in cell_text},
        right_relations={cid: set(right_rel[cid]) for cid in cell_text},
        is_rectangular=not any(any(x is None for x in row) for row in occupancy),
    )


def _process_table_lines(table_lines: list[str]) -> list[list[str]]:
    table_data: list[list[str]] = []
    separator_row_index: int | None = None
    for i, line in enumerate(table_lines):
        content_without_pipes = line.replace("|", "").strip()
        if content_without_pipes and all(c in "- :" for c in content_without_pipes):
            separator_row_index = i
            break

    for i, line in enumerate(table_lines):
        if i == separator_row_index:
            continue
        if line.strip() and all(c in "- :|" for c in line):
            continue
        cells = [cell.strip() for cell in line.split("|")]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if cells:
            table_data.append(cells)
    return table_data


def _table_data_from_md_lines(table_lines: list[str]) -> TableData | None:
    if len(table_lines) < 2:
        return None
    rows = _process_table_lines(table_lines)
    if not rows:
        return None
    row_specs: list[list[dict[str, Any]]] = [
        [
            {
                "text": cell,
                "rowspan": 1,
                "colspan": 1,
                "is_heading": row_idx == 0 or col_idx == 0,
            }
            for col_idx, cell in enumerate(row)
        ]
        for row_idx, row in enumerate(rows)
    ]
    return _build_table_data_from_specs(row_specs)


def parse_markdown_tables(md_content: str) -> list[TableData]:
    """Extract and parse Markdown pipe tables (``table_parsing.py:263-341``)."""
    lines = md_content.strip().split("\n")
    parsed_tables: list[TableData] = []
    current_table_lines: list[str] = []
    in_table = False

    for line in lines:
        if "|" in line:
            if not in_table:
                in_table = True
                current_table_lines = [line]
            else:
                current_table_lines.append(line)
        elif in_table:
            table = _table_data_from_md_lines(current_table_lines)
            if table is not None:
                parsed_tables.append(table)
            in_table = False

    if in_table:
        table = _table_data_from_md_lines(current_table_lines)
        if table is not None:
            parsed_tables.append(table)

    return parsed_tables


class _StdlibHTMLTableParser(HTMLParser):
    """Fallback HTML table parser matching ``parse_html_tables`` when ``bs4`` is absent."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[tuple[bool, list[dict[str, Any]]]]] = []
        self._table_stack: list[list[tuple[bool, list[dict[str, Any]]]]] = []
        self._thead_depth = 0
        self._row_stack: list[tuple[bool, list[dict[str, Any]]]] = []
        self._cell_stack: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        attr_map = {k.lower(): v for k, v in attrs}
        if t == "table":
            self._table_stack.append([])
        elif t == "thead" and self._table_stack:
            self._thead_depth += 1
        elif t == "tr" and self._table_stack:
            self._row_stack.append((self._thead_depth > 0, []))
        elif t in ("th", "td") and self._row_stack:
            self._cell_stack.append(
                {
                    "tag": t,
                    "rowspan": attr_map.get("rowspan"),
                    "colspan": attr_map.get("colspan"),
                    "chunks": [],
                },
            )
        elif t == "br" and self._cell_stack:
            self._cell_stack[-1]["chunks"].append("\n")

    def handle_data(self, data: str) -> None:
        if self._cell_stack:
            self._cell_stack[-1]["chunks"].append(data)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("th", "td") and self._cell_stack and self._row_stack:
            cell = self._cell_stack.pop()
            in_thead, row_cells = self._row_stack[-1]
            row_cells.append(
                {
                    "text": "".join(cell["chunks"]).strip(),
                    "raw_rowspan": cell["rowspan"],
                    "raw_colspan": cell["colspan"],
                    "is_heading": cell["tag"] == "th" or in_thead,
                },
            )
        elif t == "tr" and self._row_stack and self._table_stack:
            row_entry = self._row_stack.pop()
            self._table_stack[-1].append(row_entry)
        elif t == "thead" and self._thead_depth > 0:
            self._thead_depth -= 1
        elif t == "table" and self._table_stack:
            self.tables.append(self._table_stack.pop())


def parse_html_tables(html_content: str) -> list[TableData]:
    """Extract and parse HTML tables (``table_parsing.py:390-444``)."""
    parsed_tables: list[TableData] = []
    if _BeautifulSoup is not None:
        soup = _BeautifulSoup(html_content, "html.parser")
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if not rows:
                continue
            total_rows = len(rows)
            row_specs: list[list[dict[str, Any]]] = []
            for row_idx, row in enumerate(rows):
                cells = row.find_all(["th", "td"], recursive=False)
                heading_context = row.find_parent("thead") is not None
                row_spec: list[dict[str, Any]] = []
                for cell in cells:
                    for br in cell.find_all("br"):
                        br.replace_with("\n")
                    text = cell.get_text(separator="").strip()
                    raw_rowspan = cell.get("rowspan")
                    raw_colspan = cell.get("colspan")
                    rowspan = _safe_span_int(raw_rowspan, 1)
                    colspan = _safe_span_int(raw_colspan, 1)
                    if isinstance(raw_rowspan, str) and raw_rowspan.strip() == "0":
                        rowspan = max(1, total_rows - row_idx)
                    is_heading = cell.name == "th" or heading_context
                    row_spec.append(
                        {
                            "text": text,
                            "rowspan": rowspan,
                            "colspan": colspan,
                            "is_heading": is_heading,
                        },
                    )
                row_specs.append(row_spec)
            table_data = _build_table_data_from_specs(row_specs)
            if table_data is not None:
                parsed_tables.append(table_data)
        return parsed_tables

    parser = _StdlibHTMLTableParser()
    parser.feed(html_content)
    for raw_rows in parser.tables:
        if not raw_rows:
            continue
        total_rows = len(raw_rows)
        row_specs = []
        for row_idx, (_in_thead, cells) in enumerate(raw_rows):
            row_spec = []
            for cell in cells:
                raw_rowspan = cell["raw_rowspan"]
                raw_colspan = cell["raw_colspan"]
                rowspan = _safe_span_int(raw_rowspan, 1)
                colspan = _safe_span_int(raw_colspan, 1)
                if isinstance(raw_rowspan, str) and raw_rowspan.strip() == "0":
                    rowspan = max(1, total_rows - row_idx)
                row_spec.append(
                    {
                        "text": cell["text"],
                        "rowspan": rowspan,
                        "colspan": colspan,
                        "is_heading": cell["is_heading"],
                    },
                )
            row_specs.append(row_spec)
        table_data = _build_table_data_from_specs(row_specs)
        if table_data is not None:
            parsed_tables.append(table_data)
    return parsed_tables


# Repeat detector & unit test runners


class RepeatDetector:
    """Trailing n-gram repetition detector from ``olmocr/repeatdetect.py:8-46``."""

    def __init__(self, max_ngram_size: int = 5) -> None:
        self.max_ngram_size = max_ngram_size
        self.data = ""

    def add_letters(self, new_str: str) -> None:
        self.data += new_str

    def ngram_repeats(self) -> list[int]:
        result = [0] * self.max_ngram_size
        if not self.data:
            return result
        text = re.sub(r"\s+", " ", self.data)
        for size in range(1, self.max_ngram_size + 1):
            if len(text) < size:
                continue
            target = text[-size:]
            count = 0
            pos = len(text) - size
            while pos >= 0:
                if text[pos : pos + size] == target:
                    count += 1
                    pos -= size
                else:
                    break
            result[size - 1] = count
        return result


def run_text_presence_test(test: dict[str, Any], md_content: str) -> bool:
    """Official ``TextPresenceTest.run`` from ``olmocr/bench/tests.py:150-183``."""
    reference_query = normalize_text(str(test.get("text", "")))
    normalized_md = normalize_text(md_content)
    case_sensitive = bool(test.get("case_sensitive", True))
    if not case_sensitive:
        reference_query = reference_query.lower()
        normalized_md = normalized_md.lower()

    first_n = test.get("first_n")
    last_n = test.get("last_n")
    if first_n and last_n:
        normalized_md = normalized_md[: int(first_n)] + normalized_md[-int(last_n) :]
    elif first_n:
        normalized_md = normalized_md[: int(first_n)]
    elif last_n:
        normalized_md = normalized_md[-int(last_n) :]

    max_diffs = int(test.get("max_diffs") or 0)
    denom = len(reference_query) if len(reference_query) > 0 else 1
    threshold = 1.0 - (max_diffs / denom)
    best_ratio = fuzz.partial_ratio(reference_query, normalized_md) / 100.0
    if test.get("type") == "present":
        return best_ratio >= threshold
    return best_ratio < threshold


def run_text_order_test(test: dict[str, Any], md_content: str) -> bool:
    """Official ``TextOrderTest.run`` from ``olmocr/bench/tests.py:211-226``."""
    before = normalize_text(str(test.get("before", "")))
    after = normalize_text(str(test.get("after", "")))
    normalized_md = normalize_text(md_content)
    max_diffs = int(test.get("max_diffs") or 0)

    before_matches = find_near_matches_compat(before, normalized_md, max_l_dist=max_diffs)
    if not before_matches:
        return False
    after_matches = find_near_matches_compat(after, normalized_md, max_l_dist=max_diffs)
    if not after_matches:
        return False

    return any(b.start < a.start for b in before_matches for a in after_matches)


def run_table_test(test: dict[str, Any], content: str) -> bool:
    """Official ``TableTest.run`` from ``olmocr/bench/tests.py:376-480``."""
    target_cell = normalize_text(str(test.get("cell") or ""))
    up = normalize_text(str(test.get("up") or ""))
    down = normalize_text(str(test.get("down") or ""))
    left = normalize_text(str(test.get("left") or ""))
    right = normalize_text(str(test.get("right") or ""))
    top_heading = normalize_text(str(test.get("top_heading") or ""))
    left_heading = normalize_text(str(test.get("left_heading") or ""))
    ignore_md = bool(test.get("ignore_markdown_tables", False))
    max_diffs = int(test.get("max_diffs") or 0)

    threshold = max(0.5, 1.0 - (max_diffs / (len(target_cell) if len(target_cell) > 0 else 1)))
    tables_to_check: list[TableData] = []
    if not ignore_md:
        tables_to_check.extend(parse_markdown_tables(content))
    tables_to_check.extend(parse_html_tables(content))
    if not tables_to_check:
        return False

    def _relation_ok(
        table_data: TableData,
        comparison_str: str,
        coords: set[tuple[int, int]],
    ) -> bool:
        rel_thresh = max(
            0.5,
            1.0 - (max_diffs / (len(comparison_str) if len(comparison_str) > 0 else 1)),
        )
        for rc in coords:
            cand_text = normalize_text(table_data.cell_text.get(rc, ""))
            if (fuzz.ratio(comparison_str, cand_text) / 100.0) >= rel_thresh:
                return True
        return False

    for table_data in tables_to_check:
        matches: list[tuple[int, int]] = []
        for rc, cell_content in table_data.cell_text.items():
            sim = fuzz.ratio(target_cell, normalize_text(cell_content)) / 100.0
            if sim >= threshold:
                matches.append(rc)
        for rc in matches:
            if up and not _relation_ok(table_data, up, table_data.up_relations.get(rc, set())):
                continue
            if down and not _relation_ok(
                table_data,
                down,
                table_data.down_relations.get(rc, set()),
            ):
                continue
            if left and not _relation_ok(
                table_data,
                left,
                table_data.left_relations.get(rc, set()),
            ):
                continue
            if right and not _relation_ok(
                table_data,
                right,
                table_data.right_relations.get(rc, set()),
            ):
                continue
            if left_heading and not _relation_ok(
                table_data,
                left_heading,
                table_data.left_heading_relations(*rc),
            ):
                continue
            if top_heading and not _relation_ok(
                table_data,
                top_heading,
                table_data.top_heading_relations(*rc),
            ):
                continue
            return True

    return False


def run_baseline_test(test: dict[str, Any], content: str) -> bool:
    """Official ``BaselineTest.run`` from ``olmocr/bench/tests.py:499-545``."""
    base_content_len = len("".join(c for c in content if c.isalnum()).strip())
    max_length = test.get("max_length")
    if max_length is not None:
        check_text = content
        if test.get("max_length_skips_image_alt_tags", False):
            check_text = re.sub(r"!\[.*?\]\(.*?\)", "", content)
            base_content_len = len("".join(c for c in check_text if c.isalnum()).strip())
        return base_content_len <= int(max_length)

    if base_content_len == 0:
        return False

    max_repeats = int(test.get("max_repeats") if test.get("max_repeats") is not None else 30)
    detector = RepeatDetector(max_ngram_size=5)
    detector.add_letters(content)
    for count in detector.ngram_repeats():
        if count > max_repeats:
            return False

    check_disallowed = test.get("check_disallowed_characters")
    if check_disallowed is None or bool(check_disallowed):
        if _DISALLOWED_CHARS_RE.search(content):
            return False

    return True


def _canonicalize_sub_sup(expr: str) -> str:
    """Wrap bare _x / ^x tokens in braces and canonicalize ^{sup}_{sub} -> _{sub}^{sup}."""
    s = re.sub(r"_(\\[A-Za-z]+|[A-Za-z0-9'])", r"_{\1}", expr)
    s = re.sub(r"\^(\\[A-Za-z]+|[A-Za-z0-9'])", r"^{\1}", s)
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\^\{([^{}]*)\}_\{([^{}]*)\}", r"_{\2}^{\1}", s)
    return s


def _normalize_latex_expr(expr: str) -> str:
    """Canonicalize display/spacing/delimiter/sub-superscript variants in LaTeX."""
    s = unicodedata.normalize("NFC", str(expr)).strip()
    s = s.replace("−", "-")

    # Expand zero-arg \newcommand{\cmd}{body} definitions
    macros: list[tuple[str, str]] = []

    def _collect_macro(match: re.Match[str]) -> str:
        macros.append((match.group(1), match.group(2)))
        return ""

    s = re.compile(r"\\newcommand\s*\{\s*\\([A-Za-z]+)\s*\}\s*\{([^{}]*)\}").sub(
        _collect_macro,
        s,
    )
    for cmd_name, cmd_body in macros:
        s = re.sub(rf"\\{cmd_name}(?![A-Za-z])", lambda _m, b=cmd_body: b, s)

    # Unwrap \color{...}{body} and strip standalone \color{...}
    s = re.sub(r"\\color\s*\{[^{}]*\}\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\color\s*\{[^{}]*\}", "", s)

    # Unwrap alignment/equation wrapper environments and remove alignment tabs
    s = re.sub(
        r"\\(?:begin|end)\s*\{\s*(?:align\*?|aligned|gather\*?|gathered|split|equation\*?)\s*\}",
        "",
        s,
    )
    s = s.replace("&", "")

    # Strip sizing, style, and limit placement modifiers
    s = re.sub(
        r"\\(?:displaystyle|textstyle|scriptstyle|scriptscriptstyle|limits|nolimits)(?![A-Za-z])",
        "",
        s,
    )
    s = re.sub(
        r"\\(?:left|right|bigl|bigr|Bigl|Bigr|biggl|biggr|Biggl|Biggr)(?![A-Za-z])\s*",
        "",
        s,
    )

    # Canonical command aliases
    s = re.sub(r"\\(?:dfrac|tfrac)(?![A-Za-z])", r"\\frac", s)
    s = re.sub(r"\\to(?![A-Za-z])", r"\\rightarrow", s)
    s = re.sub(r"\\gets(?![A-Za-z])", r"\\leftarrow", s)
    s = re.sub(r"\\(?:,|;|:|!|(?:quad|qquad|enspace|thinspace)(?![A-Za-z]))", "", s)
    s = re.sub(r"\\(?:mathrm|textrm|text|operatorname)\s*\{\s*([^{}]*?)\s*\}", r"\1", s)

    # Normalize prime superscripts (e.g. g^{\prime} -> g')
    s = re.sub(
        r"\^\{\s*(?:\\prime\s*)+\}",
        lambda m: "'" * m.group(0).count("\\prime"),
        s,
    )
    s = re.sub(r"\\prime(?![A-Za-z])", "'", s)

    s = re.sub(r"[\s\u200a-\u200d\u2060\ufeff]+", "", s)
    s = _canonicalize_sub_sup(s)
    s = re.sub(r"_\{([A-Za-z0-9'])\}", r"_\1", s)
    s = re.sub(r"\^\{([A-Za-z0-9'])\}", r"^\1", s)
    return s


def extract_latex_equations(
    content: str,
    ignore_dollar_delimited: bool = False,
) -> list[str]:
    """Extract equations in exact order from ``olmocr/bench/tests.py:568-590``."""
    patterns = [
        (r"\\\((.+?)\\\)", r"\\\((.+?)\\\)"),
        (r"\\\[(.+?)\\\]", r"\\\[(.+?)\\\]"),
    ]
    if not ignore_dollar_delimited:
        patterns.extend(
            [
                (r"\$\$(.+?)\$\$", r"\$\$(.+?)\$\$"),
                (r"\$(.+?)\$", r"\$(.+?)\$"),
            ],
        )
    equations: list[str] = []
    modified_content = content
    for search_pattern, replace_pattern in patterns:
        matches = re.findall(search_pattern, modified_content, re.DOTALL)
        equations.extend(e.strip() for e in matches)
        modified_content = re.sub(replace_pattern, "", modified_content, flags=re.DOTALL)
    return equations


def run_math_test(test: dict[str, Any], content: str) -> bool:
    """Official ``MathTest.run`` from ``olmocr/bench/tests.py:566-613``."""
    ref_math = str(test.get("math", "")).strip()
    if not ref_math:
        return False
    ignore_dollar = bool(test.get("ignore_dollar_delimited", False))
    equations = extract_latex_equations(content, ignore_dollar_delimited=ignore_dollar)
    if not equations:
        return False

    if any(hyp == ref_math for hyp in equations):
        return True

    norm_ref = _normalize_latex_expr(ref_math)
    if not norm_ref:
        return False
    equations.sort(key=lambda x: -fuzz.ratio(x, ref_math))
    for hyp in equations:
        norm_hyp = _normalize_latex_expr(hyp)
        if norm_ref in norm_hyp:
            return True
    return False


def run_single_pdf_test(test: dict[str, Any], md_content: str) -> bool:
    """Dispatch a single olmOCR-Bench unit test dict against page Markdown."""
    test_type = str(test.get("type", ""))
    if test_type in ("present", "absent"):
        return run_text_presence_test(test, md_content)
    if test_type == "order":
        return run_text_order_test(test, md_content)
    if test_type == "table":
        return run_table_test(test, md_content)
    if test_type == "baseline":
        return run_baseline_test(test, md_content)
    if test_type == "math":
        return run_math_test(test, md_content)
    raise ValueError(f"olmOCR-Bench: unsupported test type {test_type!r}")


# Dataset loading, grouping & PDF page rendering


def _load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


@lru_cache(maxsize=16)
def _fetch_official_category_records(category: str) -> list[dict[str, Any]]:
    jsonl_path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=f"bench_data/{category}.jsonl",
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    return _load_jsonl_records(jsonl_path)


def _group_tests_into_pages(
    records_by_category: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Group unit tests into 1 row per PDF page, injecting ``BaselineTest``s and interleaving."""
    pages_by_cat: dict[str, dict[tuple[str, int], list[dict[str, Any]]]] = {}
    for cat in SOURCE_CATEGORIES:
        pages_by_cat[cat] = defaultdict(list)
    extra_pages: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)

    for category, records in records_by_category.items():
        for raw_test in records:
            pdf = str(raw_test["pdf"])
            page = int(raw_test.get("page", 1))
            test_entry = {k: v for k, v in raw_test.items() if v is not None}
            test_entry["group"] = category
            if category in pages_by_cat:
                pages_by_cat[category][(pdf, page)].append(test_entry)
            else:
                extra_pages[(pdf, page)].append(test_entry)

    category_page_lists: list[list[dict[str, Any]]] = []
    for category in SOURCE_CATEGORIES:
        cat_pages: list[dict[str, Any]] = []
        for (pdf, page), tests in sorted(pages_by_cat[category].items()):
            all_page_tests = list(tests)
            if not any(t.get("type") == "baseline" for t in all_page_tests):
                all_page_tests.append(
                    {
                        "id": f"{pdf}_baseline",
                        "pdf": pdf,
                        "page": page,
                        "type": "baseline",
                        "group": "baseline",
                    },
                )
            cat_pages.append(
                {
                    "doc_id": f"{category}/{pdf}:p{page}",
                    "pdf": pdf,
                    "page": page,
                    "category": category,
                    "num_tests": len(all_page_tests),
                    "tests_json": json.dumps(all_page_tests, ensure_ascii=False),
                },
            )
        if cat_pages:
            category_page_lists.append(cat_pages)

    for (pdf, page), tests in sorted(extra_pages.items()):
        all_page_tests = list(tests)
        cat = str(all_page_tests[0].get("group", "custom"))
        if not any(t.get("type") == "baseline" for t in all_page_tests):
            all_page_tests.append(
                {
                    "id": f"{pdf}_baseline",
                    "pdf": pdf,
                    "page": page,
                    "type": "baseline",
                    "group": "baseline",
                },
            )
        category_page_lists.append(
            [
                {
                    "doc_id": f"{cat}/{pdf}:p{page}",
                    "pdf": pdf,
                    "page": page,
                    "category": cat,
                    "num_tests": len(all_page_tests),
                    "tests_json": json.dumps(all_page_tests, ensure_ascii=False),
                },
            ],
        )

    interleaved: list[dict[str, Any]] = []
    max_len = max((len(lst) for lst in category_page_lists), default=0)
    for idx in range(max_len):
        for lst in category_page_lists:
            if idx < len(lst):
                interleaved.append(lst[idx])
    return interleaved


def _infer_category_from_pdf(pdf_relpath: str, explicit_category: Any = None) -> str:
    if explicit_category:
        return str(explicit_category)
    prefix = pdf_relpath.split("/", 1)[0]
    if prefix == "tables":
        return "table_tests"
    if prefix in SOURCE_CATEGORIES:
        return prefix
    return "arxiv_math"


def olmocr_bench_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Group unit-test rows into 1,403 page-level rows with deterministic category interleaving."""
    if "tests_json" in dataset.column_names:
        return dataset

    # Synthetic / unit-test dataset passed directly to process_docs
    if "_synthetic" in dataset.column_names or len(dataset) != 2927:
        by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in dataset:
            cat = _infer_category_from_pdf(str(row["pdf"]), row.get("category"))
            by_cat[cat].append(dict(row))
        pages = _group_tests_into_pages(by_cat)
        return datasets.Dataset.from_list(pages)

    records_by_category = {cat: _fetch_official_category_records(cat) for cat in SOURCE_CATEGORIES}
    pages = _group_tests_into_pages(records_by_category)
    total_tests = sum(int(p["num_tests"]) for p in pages)
    eval_logger.info(
        f"olmOCR-Bench: grouped {total_tests} unit tests across {len(pages)} PDF pages.",
    )
    return datasets.Dataset.from_list(pages)


def get_pdf_media_box_width_height(local_pdf_path: str, page_num: int) -> tuple[float, float]:
    """Official ``get_pdf_media_box_width_height`` from ``olmocr/data/renderpdf.py:9-36``."""
    command = [
        "pdfinfo",
        "-f",
        str(page_num),
        "-l",
        str(page_num),
        "-box",
        "-enc",
        "UTF-8",
        local_pdf_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ValueError(f"Error running pdfinfo: {result.stderr}")
    for line in result.splitlines() if hasattr(result, "splitlines") else result.stdout.splitlines():
        if "MediaBox" in line:
            media_box = [float(x) for x in line.split(":", 1)[1].strip().split()]
            return abs(media_box[0] - media_box[2]), abs(media_box[3] - media_box[1])
    raise ValueError("MediaBox not found in the PDF info.")


@lru_cache(maxsize=64)
def render_pdf_page_to_png_bytes(
    local_pdf_path: str,
    page_num: int = 1,
    target_longest_image_dim: int = TARGET_LONGEST_IMAGE_DIM,
) -> bytes:
    """Render a PDF page to PNG bytes at ``target_longest_image_dim`` (``renderpdf.py:39-60``)."""
    if shutil.which("pdfinfo") and shutil.which("pdftoppm"):
        width_pt, height_pt = get_pdf_media_box_width_height(local_pdf_path, page_num)
        longest_dim = max(width_pt, height_pt)
        dpi = target_longest_image_dim * 72.0 / longest_dim
        pdftoppm_result = subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-f",
                str(page_num),
                "-l",
                str(page_num),
                "-r",
                str(dpi),
                local_pdf_path,
            ],
            timeout=120,
            capture_output=True,
            check=False,
        )
        if pdftoppm_result.returncode != 0:
            err_msg = pdftoppm_result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"pdftoppm failed on {local_pdf_path} p{page_num}: {err_msg}")
        return pdftoppm_result.stdout

    try:
        import pypdfium2 as pdfium  # type: ignore[import-not-found]

        pdf_doc = pdfium.PdfDocument(local_pdf_path)
        page = pdf_doc[page_num - 1]
        longest_pt = max(float(page.get_width()), float(page.get_height()))
        scale = float(target_longest_image_dim) / longest_pt
        pil_image = page.render(scale=scale).to_pil()
        buf = io.BytesIO()
        pil_image.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()
    except ImportError as exc:
        raise RuntimeError(
            "olmOCR-Bench requires Poppler ('pdfinfo' and 'pdftoppm') or 'pypdfium2' to render PDFs.",
        ) from exc


@lru_cache(maxsize=256)
def _resolve_hf_pdf_path(pdf_relpath: str) -> str:
    return hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=f"bench_data/pdfs/{pdf_relpath}",
        repo_type="dataset",
        revision=DATASET_REVISION,
    )


def _decode_visual_item(item: Any) -> Image.Image:
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    if isinstance(item, (bytes, bytearray)):
        with Image.open(io.BytesIO(item)) as img:
            return img.convert("RGB")
    if isinstance(item, dict):
        if item.get("bytes") is not None:
            with Image.open(io.BytesIO(bytes(item["bytes"]))) as img:
                return img.convert("RGB")
        if item.get("path"):
            with Image.open(item["path"]) as img:
                return img.convert("RGB")
    raise TypeError(f"Unsupported visual representation: {type(item)!r}")


def olmocr_bench_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    """Render or decode the single-page RGB image for an olmOCR-Bench document."""
    for key in ("image", "page_image"):
        if doc.get(key) is not None:
            return [_decode_visual_item(doc[key])]

    page_num = int(doc.get("page", 1))
    pdf_path = doc.get("pdf_path") or _resolve_hf_pdf_path(str(doc["pdf"]))
    png_bytes = render_pdf_page_to_png_bytes(
        str(pdf_path),
        page_num=page_num,
        target_longest_image_dim=TARGET_LONGEST_IMAGE_DIM,
    )
    with Image.open(io.BytesIO(png_bytes)) as img:
        return [img.convert("RGB")]


def olmocr_bench_doc_to_text(
    doc: dict[str, Any],
    lmms_eval_specific_kwargs: dict[str, Any] | None = None,
) -> str:
    """Return the official olmOCR-Bench VLM prompt."""
    del doc
    kwargs = lmms_eval_specific_kwargs or {}
    template_name = str(kwargs.get("prompt_template", "full_no_document_anchoring"))
    if template_name not in PROMPT_TEMPLATES:
        raise ValueError(
            f"olmOCR-Bench: unknown prompt_template={template_name!r}; expected one of {sorted(PROMPT_TEMPLATES)}",
        )
    pre_prompt = str(kwargs.get("pre_prompt", ""))
    post_prompt = str(kwargs.get("post_prompt", ""))
    return f"{pre_prompt}{PROMPT_TEMPLATES[template_name]}{post_prompt}"


# Scoring & rank-0 macro-aggregation


def _extract_doc_tests(doc: dict[str, Any]) -> list[dict[str, Any]]:
    raw = doc.get("tests_json") or doc.get("tests")
    if isinstance(raw, str):
        parsed = json.loads(raw)
    elif isinstance(raw, list):
        parsed = raw
    else:
        raise ValueError(f"olmOCR-Bench: missing 'tests_json' in doc={doc.get('pdf')!r}")
    if not parsed:
        raise ValueError(f"olmOCR-Bench: empty unit-test list for doc={doc.get('pdf')!r}")
    return list(parsed)


def olmocr_bench_process_results(
    doc: dict[str, Any],
    results: list[str],
) -> dict[str, Any]:
    """Run all unit tests for one PDF page and emit per-category pass/total counts."""
    raw_prediction = results[0] if results else ""
    cleaned_md = clean_model_prediction(raw_prediction)
    tests = _extract_doc_tests(doc)

    group_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
    total_passed = 0
    for test in tests:
        group = str(test.get("group") or doc.get("category") or "baseline")
        passed = bool(run_single_pdf_test(test, cleaned_md))
        group_counts[group]["total"] += 1
        if passed:
            group_counts[group]["passed"] += 1
            total_passed += 1

    serializable_counts: dict[str, dict[str, int]] = {}
    for grp, counts in group_counts.items():
        serializable_counts[grp] = {
            "passed": int(counts["passed"]),
            "total": int(counts["total"]),
        }
    overall_record = {
        "pdf": str(doc.get("pdf", "")),
        "category": str(doc.get("category", "")),
        "passed": int(total_passed),
        "total": int(len(tests)),
        "group_counts": serializable_counts,
    }

    out: dict[str, Any] = {"olmocr_bench_overall": overall_record}
    for category in ALL_CATEGORIES:
        out[f"olmocr_bench_{category}"] = serializable_counts.get(category)
    return out


def _aggregate_category_counts(results: list[Any], category: str) -> float:
    passed_sum = 0
    total_sum = 0
    for item in results:
        if item is None:
            continue
        if isinstance(item, dict):
            if "group_counts" in item:
                grp = item["group_counts"].get(category)
                if grp:
                    passed_sum += int(grp["passed"])
                    total_sum += int(grp["total"])
            elif "passed" in item and "total" in item:
                passed_sum += int(item["passed"])
                total_sum += int(item["total"])
    return (passed_sum / total_sum) if total_sum > 0 else float("nan")


def olmocr_bench_aggregate_overall(results: list[dict[str, Any]]) -> float:
    """Unweighted mean (macro-average) of active category pass rates (``benchmark.py:380-389``)."""
    if not results:
        return float("nan")
    category_rates: list[float] = []
    for category in ALL_CATEGORIES:
        rate = _aggregate_category_counts(results, category)
        if not math.isnan(rate):
            category_rates.append(rate)
            eval_logger.info(f"olmOCR-Bench {category}: {rate * 100.0:.2f}%")
    if not category_rates:
        return float("nan")
    overall = sum(category_rates) / len(category_rates)
    eval_logger.info(f"olmOCR-Bench overall ({len(category_rates)} categories): {overall * 100.0:.2f}%")
    return overall


def olmocr_bench_aggregate_arxiv_math(results: list[Any]) -> float:
    return _aggregate_category_counts(results, "arxiv_math")


def olmocr_bench_aggregate_old_scans_math(results: list[Any]) -> float:
    return _aggregate_category_counts(results, "old_scans_math")


def olmocr_bench_aggregate_table_tests(results: list[Any]) -> float:
    return _aggregate_category_counts(results, "table_tests")


def olmocr_bench_aggregate_old_scans(results: list[Any]) -> float:
    return _aggregate_category_counts(results, "old_scans")


def olmocr_bench_aggregate_headers_footers(results: list[Any]) -> float:
    return _aggregate_category_counts(results, "headers_footers")


def olmocr_bench_aggregate_multi_column(results: list[Any]) -> float:
    return _aggregate_category_counts(results, "multi_column")


def olmocr_bench_aggregate_long_tiny_text(results: list[Any]) -> float:
    return _aggregate_category_counts(results, "long_tiny_text")


def olmocr_bench_aggregate_baseline(results: list[Any]) -> float:
    return _aggregate_category_counts(results, "baseline")
