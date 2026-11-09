# olmOCR-Bench

olmOCR-Bench evaluates document OCR and page-to-Markdown conversion across 1,403 single-page PDF
documents and 8,413 executable unit tests spanning scientific math, historical scans, complex
tables, multi-column reading order, header/footer suppression, and dense tiny text.

- Paper: [olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models (arXiv:2502.18443)](https://arxiv.org/abs/2502.18443)
- Dataset: [allenai/olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench)
  (`1,403` PDF pages, `7,019` category unit tests + `1,394` baseline quality checks = `8,413` unit tests)
- Official Code: [allenai/olmocr (`olmocr/bench`)](https://github.com/allenai/olmocr/tree/main/olmocr/bench)

## Task & Metrics

`olmocr_bench_process_docs` groups the 8,413 unit tests by PDF page into 1,403 page-level evaluation
samples (interleaved round-robin across the 7 source categories so `--limit N` smoke runs exercise
every category). Each page is rendered to a PNG image (`target_longest_image_dim=2048`) and transcribed
once by the VLM, after which all unit tests associated with that page are evaluated against the single
Markdown output.

| Metric | Category / Source | Official Test Count | Description |
|---|---|---:|---|
| `olmocr_bench_overall` | Macro-average across 8 categories | 8,413 | Unweighted mean of the 8 category pass rates |
| `olmocr_bench_arxiv_math` | `arxiv_math.jsonl` | 2,927 | LaTeX math equations on arXiv pages (`522` PDFs) |
| `olmocr_bench_old_scans_math` | `old_scans_math.jsonl` | 458 | Mathematical formulas in historical scans (`36` PDFs) |
| `olmocr_bench_table_tests` | `table_tests.jsonl` | 1,022 | Cell adjacency (`up`/`down`/`left`/`right`) and heading relations in Markdown/HTML tables (`188` PDFs) |
| `olmocr_bench_old_scans` | `old_scans.jsonl` | 526 | Text presence, absence, and reading order on historical scans (`98` PDFs) |
| `olmocr_bench_headers_footers` | `headers_footers.jsonl` | 760 | Suppression of running headers/footers (`absent` checks on `first_n`/`last_n` characters, `266` PDFs) |
| `olmocr_bench_multi_column` | `multi_column.jsonl` | 884 | Multi-column reading-order verification (`order` checks, `231` PDFs) |
| `olmocr_bench_long_tiny_text` | `long_tiny_text.jsonl` | 442 | Dense small-font text presence checks (`62` PDFs) |
| `olmocr_bench_baseline` | Auto-injected `BaselineTest` | 1,394 | Non-empty alphanumeric output, no >30 trailing n-gram repeats, and no CJK/emoji hallucinations (`1,394` PDFs) |

## Protocol

- **PDF Rendering**: Uses `pdfinfo` and `pdftoppm` (`target_longest_image_dim=2048` at `2048 * 72 / max(width_pt, height_pt)` DPI) from `olmocr/data/renderpdf.py` (with `pypdfium2` fallback).
- **Prompt**: Defaults to `build_openai_silver_data_prompt_no_document_anchoring` from `olmocr/bench/prompts.py` (`prompt_template: full_no_document_anchoring`), and supports `basic` and `yaml` via `lmms_eval_specific_kwargs`.
- **Unit Test Evaluators**: Implements `TextPresenceTest` (`present`/`absent`), `TextOrderTest` (`order`), `TableTest` (`table`), `MathTest` (`math`), and `BaselineTest` (`baseline`) from `olmocr/bench/tests.py`.

## Usage

```bash
python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
    --tasks olmocr_bench \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/
```
