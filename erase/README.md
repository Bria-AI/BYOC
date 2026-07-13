# Erase: BYOC Object-Removal Pipeline

This folder documents the **Python package** path for running Bria **Erase** (object removal /
inpainting) in your own environment: install **`erase`** from the **`bria-erase`** AWS
CodeArtifact repository, let it pull the Bria model weights from Hugging Face, then call the local
pipeline directly from Python. No Triton, no inference server — everything runs in-process on your GPU.

Erase removes a masked object and fills the region in two stages:

1. **LaMa** — a TorchScript `big-lama` model produces a fast coarse fill of the masked region.
2. **SDXL ControlNet ("eraser")** — a diffusers refine pass regenerates the region at generation
   resolution, then the result is alpha-composited back into the full-resolution original.

This is a faithful, numerically-matched port of the production `/v2/image/edit/erase` endpoint.

## Overview

The included **`code_example.ipynb`** notebook demonstrates:

- Obtaining a short-lived CodeArtifact credential from the Bria Engine.
- Installing **`erase`** from **`bria-erase`**.
- Running the local **`Erase`** pipeline on the sample image + mask.
- Saving the erased result and an input / mask / output comparison strip.

## Prerequisites

Unlike Increase Resolution, Erase is **not** a TensorRT engine — it runs the standard
`torch` + `diffusers` stack, so the runtime is portable across recent NVIDIA GPUs and CUDA versions:

- **GPU:** an NVIDIA GPU with **≥ 16 GB** VRAM. Production runs on **A10G**; any Ampere-or-newer
  card of similar memory works (the diffusion stage is the memory driver).
- **Python:** **3.10+**.
- **Torch:** **2.2.2** (installed automatically from PyPI; the default build targets CUDA 12.1).
- Network access to the Bria Engine, AWS CodeArtifact, and Hugging Face.
- `BRIA_API_TOKEN`, used to request a short-lived CodeArtifact credential.

```bash
export BRIA_API_TOKEN="your-api-token-here"
```

## CodeArtifact token

Call the Bria Engine once to obtain a PyPI password for the CodeArtifact repository:

```http
GET https://engine.prod.bria-api.com/v2/auth/access/code_artifact?repository=bria-erase
api_token: <BRIA_API_TOKEN>
```

Use `result.authorization_token` as the password for the CodeArtifact PyPI simple index with
username `aws`.

## Install `erase`

Erase is a single GPU package — there is no CPU/GPU role split. `torch`, `diffusers`, `opencv`,
etc. resolve automatically from PyPI:

```bash
export CODE_ARTIFACT_PASSWORD="<paste authorization_token here>"
# URL-encode the token so characters like +, /, = don't break the index URL
ENCODED_PASSWORD=$(python3 -c "from urllib.parse import quote; print(quote('${CODE_ARTIFACT_PASSWORD}', safe=''))")
BRIA_IDX="https://aws:${ENCODED_PASSWORD}@bria-300465780738.d.codeartifact.us-east-1.amazonaws.com/pypi/bria-erase/simple/"

python3 -m pip install --upgrade "erase" --extra-index-url "$BRIA_IDX"
```

## Model weights

The weights are pulled automatically from the public Hugging Face repo **`briaai/Expansion`** on
first run and cached under `~/.cache/bria/erase`:

- Stage 2: `vae/`, `controlnet_xl_eraser/`, `BRIA-2.3-FAST-LORA-FUSED/` (SDXL base + scheduler).
- Stage 1: `lama/model.pt` (TorchScript `big-lama`).

No manual download is required. To run **fully offline**, download the files once and point the
config at local paths via environment variables (all prefixed `ERASE_`):

```bash
export ERASE_LAMA_MODEL_PATH=/path/to/model.pt
export ERASE_VAE_MODEL_PATH=/path/to/vae
export ERASE_ERASER_CONTROLNET_PATH=/path/to/controlnet_xl_eraser
export ERASE_BASE_PIPE_MODEL_PATH=/path/to/BRIA-2.3-FAST-LORA-FUSED
export ERASE_SCHEDULER_CONFIG_PATH=/path/to/BRIA-2.3-FAST-LORA-FUSED/scheduler/scheduler_config.json
```

## Run the notebook

Open **`code_example.ipynb`** and run the cells top to bottom. It installs the package, runs the
pipeline on the sample inputs, and writes:

- `outputs/erased.png` — the result.
- `outputs/compare.png` — input | masked region | erased, side by side.

## Minimal usage

```python
from erase import Erase, EraseConfig, EraseInput
from PIL import Image

pipe = Erase(config=EraseConfig())   # weights auto-download on first setup()
pipe.setup()

image = Image.open("input.png").convert("RGB")
mask = Image.open("mask.png").convert("L")        # white = region to erase
result = pipe.execute(EraseInput(image=image, mask=mask)).image
result.save("erased.png")

pipe.cleanup()
```
