# Retrieval baseline — 2026-09-09

Run: `uv run python scripts/evaluate_retrieval.py` at 07:55 UTC.

The live Compose backend used Ollama `qwen3-embedding:0.6b` and Qdrant 1.18.2.
Three synthetic PDFs containing nine labeled pages were uploaded, ingested,
and queried through the frontend API proxy. Each question searched all three
papers. Results used 20 candidate slots and selected three passages with MMR
(relevance weight 0.7).

Top-1 accuracy: **9/9 (100%)**. Hit-at-3: **9/9 (100%)**.
Every returned excerpt matched the normalized text of its referenced page.

| Question topic | Expected paper | PDF page | Top-1 correct | Top cosine score |
| --- | --- | ---: | --- | ---: |
| Temperature and capacity loss | Batteries | 1 | Yes | 0.7166332 |
| Charge estimation and error correction | Batteries | 2 | Yes | 0.7750923 |
| Separator thermal shutdown | Batteries | 3 | Yes | 0.7746669 |
| Moisture trigger for watering | Irrigation | 1 | Yes | 0.76946133 |
| Water savings and crop yield | Irrigation | 2 | Yes | 0.7469307 |
| Nitrogen loss measurements | Irrigation | 3 | Yes | 0.60351765 |
| Transit brightness changes | Exoplanets | 1 | Yes | 0.6934047 |
| Doppler shifts and planetary mass | Exoplanets | 2 | Yes | 0.72843826 |
| Sodium atmospheric detection | Exoplanets | 3 | Yes | 0.6977843 |

Additional live checks passed: single-paper and multi-paper filters, exclusion
of an unselected paper even for a question about it, default eight-source cap,
empty results with a deliberately strict experimental threshold, and HTTP 404
for an unknown paper ID. All three temporary uploads and their indexed vectors
were removed afterward.

The backend suite passed 86 tests, including request validation, invalid source
payloads, dependency failures, MMR selection, production route gating, retrieval
timeouts, and concurrent deletion. Ruff and strict MyPy checks passed.

This is a small synthetic smoke baseline, not a representative scientific-paper
benchmark. It does not validate answer generation or establish a safe
answerability threshold. Real papers with longer passages, overlapping chunks,
tables, and less direct questions still need evaluation.

The reproducible inputs are in [retrieval_baseline.json](retrieval_baseline.json).
