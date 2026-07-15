# Erase: BYOC Object-Removal Pipeline

This folder documents the **Python package** path for running Bria **Erase** (object removal /
inpainting) in your own environment: install **`erase`** from the **`bria-erase`** AWS
CodeArtifact repository, download the Bria model weights from Hugging Face, then call the local
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
- Downloading the model weights from Hugging Face and passing their local paths to the config.
- Running the local **`Erase`** pipeline on the sample image + mask.
- Saving the erased result and an input / mask / output comparison strip.

## Prerequisites

Unlike Increase Resolution, Erase is **not** a TensorRT engine — it runs the standard
`torch` + `diffusers` stack, so the runtime is portable across recent NVIDIA GPUs and CUDA versions:

- **GPU:** an NVIDIA GPU with **≥ 16 GB** VRAM. Production runs on **A10G**; any Ampere-or-newer
  card of similar memory works (the diffusion stage is the memory driver).
- **Python:** **3.10+**.
- **Torch:** **2.2.2** (installed automatically from PyPI; the default build targets CUDA 12.1).
- **`BRIA_API_TOKEN`** — to request a short-lived CodeArtifact credential.
- **`HF_TOKEN`** — for Hugging Face access to the `briaai/Expansion` model repo.
- Network access to the Bria Engine, AWS CodeArtifact, and Hugging Face.

```bash
export BRIA_API_TOKEN="your-api-token-here"
export HF_TOKEN="your-hf-token-here"
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

Erase is a single GPU package — there is no CPU/GPU role split. A **single index** is enough: the
`bria-erase` repo has `bria-external` and `bria-diffusers` (plus public PyPI) as upstream
repositories, so `erase` and all its deps (`torch`, `diffusers`, `bria-external-ml`,
`bria-diffusers`, …) resolve through this one token/index:

```bash
export CODE_ARTIFACT_PASSWORD="<paste authorization_token here>"
# URL-encode the token so characters like +, /, = don't break the index URL
ENCODED_PASSWORD=$(python3 -c "from urllib.parse import quote; print(quote('${CODE_ARTIFACT_PASSWORD}', safe=''))")
HOST="bria-300465780738.d.codeartifact.us-east-1.amazonaws.com"
BRIA_IDX="https://aws:${ENCODED_PASSWORD}@${HOST}/pypi/bria-erase/simple/"

python3 -m pip install --upgrade "erase" --extra-index-url "$BRIA_IDX"
```

## Model weights

Download the weights from the Hugging Face repo **`briaai/Expansion`** and pass the resulting local
directories into `EraseConfig` (the library is not opinionated about where weights come from — the
repo id lives in your code, not inside the package):

```python
from pathlib import Path
from huggingface_hub import snapshot_download

weights = Path(snapshot_download(
    repo_id="briaai/Expansion",
    token="<HF_TOKEN>",
    allow_patterns=["vae/**", "controlnet_xl_eraser/**", "BRIA-2.3-FAST-LORA-FUSED/**", "lama/**"],
))
```

The snapshot contains:

- Stage 2: `vae/`, `controlnet_xl_eraser/`, `BRIA-2.3-FAST-LORA-FUSED/` (SDXL base + scheduler).
- Stage 1: `lama/model.pt` (TorchScript `big-lama`).

## Run the notebook

Open **`code_example.ipynb`** and run the cells top to bottom. It installs the package, downloads the
weights, runs the pipeline on the sample inputs, and writes:

- `outputs/erased.png` — the result.
- `outputs/compare.png` — input | masked region | erased, side by side.

## Minimal usage

```python
from pathlib import Path
from erase import Erase, EraseConfig, EraseInput
from PIL import Image

config = EraseConfig(
    lama_model_path=str(weights / "lama" / "model.pt"),
    vae_model_path=str(weights / "vae"),
    eraser_controlnet_path=str(weights / "controlnet_xl_eraser"),
    base_pipe_model_path=str(weights / "BRIA-2.3-FAST-LORA-FUSED"),
    scheduler_config_path=str(weights / "BRIA-2.3-FAST-LORA-FUSED" / "scheduler" / "scheduler_config.json"),
)
pipe = Erase(config=config)
pipe.setup()

image = Image.open("input.png").convert("RGB")
mask = Image.open("mask.png").convert("L")        # white = region to erase
result = pipe.execute(EraseInput(image=image, mask=mask)).image
result.save("erased.png")

pipe.cleanup()
```
