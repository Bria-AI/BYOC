# Increase Resolution: BYOC Super-Resolution Pipeline

This folder documents the **Python package** path for running Increase Resolution in your own
environment: install **`increase-resolution`** from the **`bria-increase-res`** AWS
CodeArtifact repository, download the Bria super-resolution **TensorRT engine(s)**, then call the
local pipeline directly from Python.

Increase Resolution upscales an image **2×** or **4×** using a SwinIR super-resolution model. The
image is split into overlapping tiles, each tile is upscaled by the engine in-process, and the tiles
are merged back into the full-resolution result.

## Overview

The included **`code_example.ipynb`** notebook demonstrates:

- Obtaining a short-lived CodeArtifact credential from the Bria Engine.
- Installing **`increase-resolution`** from **`bria-increase-res`**.
- Downloading the super-resolution **`.engine`** files provided by Bria.
- Running the local **`IncreaseResolution`** pipeline on a sample image at 2× and 4×.

## Prerequisites — exact environment (required)

The pipeline runs a **TensorRT engine**, which is compiled for one specific runtime. It will only
load on a matching environment:

- **GPU:** NVIDIA **A10** (Ampere), or another Ampere GPU of the same compute capability.
- **CUDA:** **11.7.1**.
- **TensorRT:** **8.4.1.5** (exact — a different 8.4.x patch will not load the engine).
- **Python:** **3.10**.
- Network access to the Bria Engine and AWS CodeArtifact.
- `BRIA_API_TOKEN`, used to request a short-lived CodeArtifact credential.

> These versions are not flexible. A TensorRT engine is tied to the exact GPU architecture and the
> TensorRT/CUDA versions it was built with; a different GPU or a different TensorRT patch will fail
> to load it.
>
> **Runtime:** the exact CUDA 11.7.1 libraries are not available from pip, so the supported base is
> the NVIDIA image **`nvcr.io/nvidia/pytorch:22.07-py3`** (CUDA 11.7.1 + TensorRT 8.4.1.5); install
> this package into a Python 3.10 environment on top of it.

```bash
export BRIA_API_TOKEN="your-api-token-here"
```

## Runtime setup (required) — a g5 (A10G) instance

The engine is a TensorRT build compiled against the exact CUDA 11.7.1 / TensorRT 8.4.1.5 libraries in
the NVIDIA `22.07` container, so it must run **inside that image on an Ampere GPU**. On AWS that's a
**g5** instance (A10G). This exact sequence was validated end-to-end there:

```bash
# 1. pull the runtime image (once; ~22 GB, from NVIDIA NGC)
docker pull nvcr.io/nvidia/pytorch:22.07-py3

# 2. start it with the GPU + your tokens.
#    To run the notebooks, ALSO add:  -p 8888:8888 -v "$PWD":/work   (see "Run the notebooks")
docker run --gpus all --ipc=host -it \
  -e BRIA_API_TOKEN="$BRIA_API_TOKEN" -e HF_TOKEN="$HF_TOKEN" \
  nvcr.io/nvidia/pytorch:22.07-py3 bash

# 3. inside the container — build a Python 3.10 env (the image ships 3.8):
unset PYTHONPATH LD_LIBRARY_PATH          # keep the image's 3.8 packages / system libtorch out of the venv
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="/root/.local/bin:$PATH"
uv venv --python 3.10 /opt/ir && source /opt/ir/bin/activate
uv pip install pip                        # a uv venv has no pip; the install below (and the notebook) need it
export LD_PRELOAD=/usr/local/cuda-11.7/targets/x86_64-linux/lib/libcublasLt.so.11   # the engine's Myelin graph needs this exact cuBLAS
```

- **`BRIA_API_TOKEN`** — your custom-plan Bria token (for the CodeArtifact credential below).
- **`HF_TOKEN`** — needs **approved access to the gated `briaai/increase-resolution`** repo (request it
  on Hugging Face; Bria approves). The notebook pulls the engines with it.
- `torch` cu117 + `nvidia-tensorrt` come from the `[all]`/`[gpu]` extra in the install step; everything
  else (numpy/opencv/pillow/pydantic/bria-external) resolves automatically.

## CodeArtifact Token

Call the Bria Engine once to obtain a PyPI password for the CodeArtifact repository:

```http
GET https://engine.prod.bria-api.com/v2/auth/access/code_artifact?repository=bria-increase-res
api_token: <BRIA_API_TOKEN>
```

Use `result.authorization_token` as the password for the CodeArtifact PyPI simple index with
username `aws`.

## Install `increase-resolution`

Install the extra for your role:

| Extra | Role | Needs the cu117 / NVIDIA indexes? |
|---|---|---|
| `[all]` | single-machine full pipeline (`execute`) | yes |
| `[gpu]` | worker: tile inference (`TileWorker.infer`) | yes |
| `[cpu]` | coordinator: `split` / `merge` | no (no torch/tensorrt) |

```bash
export CODE_ARTIFACT_PASSWORD="<paste authorization_token here>"
# URL-encode the token so characters like +, /, = don't break the index URL
ENCODED_PASSWORD=$(python3 -c "from urllib.parse import quote; print(quote('${CODE_ARTIFACT_PASSWORD}', safe=''))")
BRIA_IDX="https://aws:${ENCODED_PASSWORD}@bria-300465780738.d.codeartifact.us-east-1.amazonaws.com/pypi/bria-increase-res/simple/"

# full pipeline / worker (needs torch cu117 + nvidia-tensorrt from their indexes):
python3 -m pip install --upgrade "increase-resolution[all]" \
  --extra-index-url https://download.pytorch.org/whl/cu117 \
  --extra-index-url https://pypi.ngc.nvidia.com \
  --extra-index-url "$BRIA_IDX"

# coordinator only (split/merge — no GPU, so no extra indexes needed):
python3 -m pip install --upgrade "increase-resolution[cpu]" --extra-index-url "$BRIA_IDX"
```

You must also have `torch` and `tensorrt==8.4.*` installed matching your CUDA 11.7 environment.

## Engine files

Bria hosts the super-resolution engines on the Hugging Face repo **`briaai/increase-resolution`**
(gated — request access on Hugging Face; once Bria approves, set `HF_TOKEN`). The notebook fetches
them with `huggingface_hub.hf_hub_download`; the two files are:

```text
increase_resolution2.engine   # 2x
increase_resolution4.engine   # 4x
```

> These are TensorRT builds tied to the exact GPU + CUDA/TensorRT runtime documented above — they
> only load on a matching environment.

## Run the notebooks

Both notebooks must run with a **kernel inside the container** (they can't run in a host venv — the
engine only loads in this runtime). Two walkthroughs:

- **`code_example.ipynb`** — simple **image → image** on one machine (`pipeline.execute`).
- **`code_example_distributed.ipynb`** — **tile-level** flow for spreading the GPU work across
  **multiple worker machines**: `split` (coordinator) → `TileWorker.infer` (per worker GPU) →
  `merge` (coordinator).

Start a Jupyter server *in the container* and drive it from your editor. Run this **one command from
this folder** — it starts the container with the notebooks mounted (`-v "$PWD":/work`) and the port
published (`-p 8888:8888`), does the setup, and launches Jupyter. (Both `-v` and `-p` must be on the
`docker run` itself — you can't add them to an already-running container.)

```bash
docker run --gpus all --ipc=host -it -p 8888:8888 -v "$PWD":/work \
  -e BRIA_API_TOKEN="$BRIA_API_TOKEN" -e HF_TOKEN="$HF_TOKEN" \
  nvcr.io/nvidia/pytorch:22.07-py3 bash -lc '
    unset PYTHONPATH LD_LIBRARY_PATH
    curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH=/root/.local/bin:$PATH
    uv venv --python 3.10 /opt/ir && . /opt/ir/bin/activate && uv pip install pip jupyterlab ipykernel
    export LD_PRELOAD=/usr/local/cuda-11.7/targets/x86_64-linux/lib/libcublasLt.so.11
    cd /work && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'
```

Then in VS Code / Cursor: open the notebook → kernel picker → **Existing Jupyter Server** →
`http://localhost:8888` (use the `?token=…` printed in the log) → **run top-to-bottom**. The notebook
installs `increase-resolution`, pulls the engines from `briaai/increase-resolution`, and runs on the
A10G; outputs are saved under `outputs/`. (Headless alternative: `jupyter nbconvert --to notebook --execute code_example.ipynb`.)

## Distributed (tile-level) usage

For high throughput or many GPUs across separate machines, run the tile inference on your own fleet.
The pipeline exposes the three steps so the transport in the middle is yours (queue / RPC / Ray / …):

```python
from increase_resolution import split, merge, TileWorker

res = split(image, scale=4)                        # coordinator (CPU): independent numpy tiles + layout
# --- ship res.tiles to your worker machines; each runs the engine once and returns infer(tile) ---
worker = TileWorker(".../increase_resolution4.engine").setup()
upscaled = [worker.infer(t) for t in res.tiles]    # <-- replace with your cross-machine fan-out
# ----------------------------------------------------------------------------------------------
merge(upscaled, res.layout, res.alpha).save("upscaled.png")   # coordinator (CPU): stitch back
```

`split`/`merge` need no GPU or engine; only the workers do. Tiles are plain numpy arrays so your
transport can serialize them. The output is identical to the single-call `execute`.

## Using the Python package

```python
from increase_resolution import IncreaseResolution, IncreaseResolutionConfig, IncreaseResolutionInput

config = IncreaseResolutionConfig(engine_paths={2: ".../increase_resolution2.engine",
                                                4: ".../increase_resolution4.engine"})
pipeline = IncreaseResolution(config=config)
pipeline.setup()

result = pipeline.execute(IncreaseResolutionInput(image="https://.../photo.jpg", desired_increase=4))
result.image.save("upscaled.png")

pipeline.cleanup()
```

`image` accepts a URL, base64 string, numpy array, or PIL image. `desired_increase` must be a scale
whose engine you loaded. Set `preserve_alpha=True` to keep an RGBA input's alpha in the output.
