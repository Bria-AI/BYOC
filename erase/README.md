# Erase

`erase` is a Bria object-removal / inpainting pipeline. Given an image and a mask, it removes the
masked object and fills the region, running **in-process** on your GPU — no Triton, no inference
server. It is a numerically-matched port of the production `/v2/image/edit/erase` endpoint.

## Two-stage pipeline

1. **LaMa** — a TorchScript `big-lama` model produces a fast coarse fill of the masked region
   (capped at `MAX_IMAGE_SIZE_LAMA`).
2. **SDXL ControlNet ("eraser")** — a diffusers refine pass regenerates the region at generation
   resolution; the result is then alpha-composited back into the full-resolution original.

## Requirements

Unlike `increase-resolution`, this is **not** a TensorRT engine — it runs the standard
`torch` + `diffusers` stack, so the runtime is portable across recent NVIDIA GPUs / CUDA versions:

- **GPU:** an NVIDIA GPU with **≥ 16 GB** VRAM (production runs on **A10G**; any Ampere-or-newer
  card of similar memory works)
- **Python:** **3.10+**
- **Torch:** **2.2.2** (installed from PyPI; default build targets CUDA 12.1)

## Inputs / config

- **Input** (`EraseInput`): image + mask, with `mask_type` (`MANUAL` / `AUTOMATIC`) selecting how the
  mask is interpreted.
- **Config** (`EraseConfig`): the caller supplies all weight paths — the library is not opinionated
  about where they come from. Paths may be passed explicitly or via `ERASE_*` env vars; pydantic
  rejects a config with the required paths missing.

## Model weights

Download the weights from the Hugging Face repo **`briaai/Expansion`** (or any source) and pass the
directories into `EraseConfig`:

- Stage 2: `vae/`, `controlnet_xl_eraser/`, `BRIA-2.3-FAST-LORA-FUSED/` (SDXL base + scheduler)
- Stage 1: `lama/model.pt` (TorchScript `big-lama`)

The paths can also be provided via the `ERASE_*` environment variables (`ERASE_LAMA_MODEL_PATH`,
`ERASE_VAE_MODEL_PATH`, `ERASE_ERASER_CONTROLNET_PATH`, `ERASE_BASE_PIPE_MODEL_PATH`,
`ERASE_SCHEDULER_CONFIG_PATH`).

## Non-obvious decisions

- Weights are **caller-provided** (Hugging Face `briaai/Expansion` or any source) rather than bundled,
  keeping the package light and the source of weights flexible.
- Image helpers live in and are tested by `bria_external.image`; this package only owns the two-stage
  erase logic and its public surface.
