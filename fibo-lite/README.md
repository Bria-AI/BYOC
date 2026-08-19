# Fibo Lite

FIBO 1.5 "lite" text-to-image pipeline: a VLM turns the prompt into a structured prompt, then the
Fibo Lite diffuser renders it. AOT-compiled via `bria_external.ml.aot` (driver-versioned S3 paths),
with a JIT fallback.

- `src/fibo_lite/` — pipeline entry (`fibo_lite.py`), `config.py`, and I/O types (`input_output.py`).
- `fibo_lite_diffuser/`, `vlm/` — the diffuser and VLM sub-components.
