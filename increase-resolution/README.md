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

## Runtime setup (required)

The engine is compiled against the exact CUDA 11.7.1 libraries in the NVIDIA `22.07` image, and those
libraries are not available from pip. Start from that base image and prepare a Python 3.10 environment
on top of it. This exact sequence was validated end-to-end on an A10:

```bash
docker run --gpus all -it -v /path/to/engines:/engines nvcr.io/nvidia/pytorch:22.07-py3 bash

# inside the container:
unset PYTHONPATH          # keep the image's Python 3.8 packages out of the venv
unset LD_LIBRARY_PATH     # keep the image's system libtorch from shadowing the venv's torch

python3.10 -m venv /opt/ir && source /opt/ir/bin/activate   # or: uv venv --python 3.10 /opt/ir

# the engine's Myelin graph needs this exact cuBLAS (the image provides it; pip does not)
export LD_PRELOAD=/usr/local/cuda-11.7/targets/x86_64-linux/lib/libcublasLt.so.11
```

The GPU runtime (`torch` cu117 + `nvidia-tensorrt`) is pulled in by the `[gpu]`/`[all]` extra in the
next section (from its package indexes); everything else (numpy/opencv/pillow/pydantic/bria-external)
resolves automatically.

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

Bria provides the super-resolution engines as `.engine` files (one per scale). Download them to a
local folder and pass the paths to the pipeline config:

```text
increase_resolution2.engine   # 2x
increase_resolution4.engine   # 4x
```

## Run the notebooks

Two walkthroughs are included:

- **`code_example.ipynb`** — simple **image → image** on one machine (`pipeline.execute`).
- **`code_example_distributed.ipynb`** — **tile-level** flow for spreading the GPU work across
  **multiple worker machines**: `split` (coordinator) → `TileWorker.infer` (per worker GPU) →
  `merge` (coordinator).

```bash
jupyter notebook code_example.ipynb              # simple
jupyter notebook code_example_distributed.ipynb  # distributed (tile-level)
```

The notebooks save generated files under `outputs/`.

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
