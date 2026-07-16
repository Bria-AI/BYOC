# Increase Resolution: BYOC Super-Resolution Pipeline

Run Bria **Increase Resolution** (2×/4× super-resolution) in your own environment. An image is split into overlapping tiles, each tile is upscaled by the
engine in-process, and the tiles are merged back into the full-resolution result.

Because it's a TensorRT engine, it runs **only** inside a specific NVIDIA container on an Ampere GPU
(see Prerequisites). The fastest way to try it is the example notebook — the **Quickstart** below.

## Prerequisites — exact environment (required)

A TensorRT engine is tied to the exact GPU architecture + CUDA/TensorRT it was built with, so the
runtime is **not** flexible:

- **GPU:** NVIDIA **Ampere** — **A10 / A10G** (on AWS: a **g5** instance).
- **Runtime:** the NVIDIA container **`nvcr.io/nvidia/pytorch:22.07-py3`** (CUDA **11.7.1** + TensorRT
  **8.4.1.5**) — those exact libraries aren't on pip, so everything runs inside this image; **Python 3.10**.
- **`BRIA_API_TOKEN`** — a custom-plan Bria token (used to get a short-lived CodeArtifact credential).
- **`HF_TOKEN`** — with **approved access to the gated `briaai/increase-resolution`** HF repo (request
  access on Hugging Face; Bria approves) — the engines are hosted there.
- Network access to the Bria Engine, AWS CodeArtifact, and Hugging Face.

## Quickstart — run the example notebook

This is the whole thing. You start the NVIDIA container (with Jupyter) and run the notebook, which
does everything for you: fetch the CodeArtifact token → install `increase-resolution` → pull the
engines from HF → upscale. Two walkthroughs are included:

- **`code_example.ipynb`** — simple image → image on one machine.
- **`code_example_distributed.ipynb`** — tile-level flow for spreading GPU work across machines.

**1. Start the container + Jupyter — one command, run from this folder** (`BYOC/increase-resolution/`).
It mounts the notebooks (`-v "$PWD":/work`), publishes the Jupyter port (`-p 8888:8888`), sets up
Python 3.10, and launches Jupyter. (Both `-v` and `-p` must be on the `docker run` — you can't add
them to an already-running container.)

```bash
export BRIA_API_TOKEN="your-custom-plan-token"
export HF_TOKEN="your-hf-token"     # must have approved access to gated briaai/increase-resolution

docker run --gpus all --ipc=host -it -p 8888:8888 -v "$PWD":/work \
  -e BRIA_API_TOKEN="$BRIA_API_TOKEN" -e HF_TOKEN="$HF_TOKEN" \
  nvcr.io/nvidia/pytorch:22.07-py3 bash -lc '
    unset PYTHONPATH LD_LIBRARY_PATH
    curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH=/root/.local/bin:$PATH
    uv venv --python 3.10 /opt/ir && . /opt/ir/bin/activate && uv pip install pip jupyterlab ipykernel huggingface_hub
    export LD_PRELOAD=/usr/local/cuda-11.7/targets/x86_64-linux/lib/libcublasLt.so.11
    cd /work && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'
```

**2. Connect your editor to it.** In VS Code / Cursor: open `code_example.ipynb` → kernel picker
(top-right) → **Select Another Kernel… → Existing Jupyter Server…** → enter
`http://localhost:8888` and the **token** printed in the log above → pick the **Python 3** kernel.

**3. Run the notebook top-to-bottom.** It fetches the CodeArtifact token, `pip install`s the package
(torch-cu117 + tensorrt — a few minutes the first time), pulls the engines from
`briaai/increase-resolution` with your `HF_TOKEN`, and upscales the sample at 2× and 4×. Outputs are
saved under `outputs/`.

> Your `BRIA_API_TOKEN` / `HF_TOKEN` are passed into the container as env vars, so the notebook's
> token cells just work. You do **not** run the manual steps below — the notebook does them for you.
> `code_example_distributed.ipynb` runs the same way (same server + kernel).

---

## Installing the package directly (without the notebook)

Everything below is what the notebook automates. Do it by hand **only** if you're integrating the
`increase-resolution` package into your own code / CLI. Start the same container **without** the
Jupyter/mount parts:

```bash
docker run --gpus all --ipc=host -it \
  -e BRIA_API_TOKEN="$BRIA_API_TOKEN" -e HF_TOKEN="$HF_TOKEN" \
  nvcr.io/nvidia/pytorch:22.07-py3 bash

# inside the container — build the Python 3.10 env (the image ships 3.8):
unset PYTHONPATH LD_LIBRARY_PATH          # keep the image's 3.8 packages / system libtorch out of the venv
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="/root/.local/bin:$PATH"
uv venv --python 3.10 /opt/ir && source /opt/ir/bin/activate
uv pip install pip                        # a uv venv has no pip
export LD_PRELOAD=/usr/local/cuda-11.7/targets/x86_64-linux/lib/libcublasLt.so.11   # the engine's Myelin graph needs this exact cuBLAS
```

### CodeArtifact token

Call the Bria Engine once to obtain a PyPI password for the CodeArtifact repository:

```http
GET https://engine.prod.bria-api.com/v2/auth/access/code_artifact?repository=bria-increase-res
api_token: <BRIA_API_TOKEN>
```

Use `result.authorization_token` as the password for the CodeArtifact PyPI simple index (username `aws`).

### Install `increase-resolution`

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

# full pipeline / worker (needs torch cu117 + nvidia-tensorrt from their indexes).
# huggingface_hub is added explicitly — it's used to download the engines (below) but is not a
# dependency of the increase-resolution package itself:
python3 -m pip install --upgrade "increase-resolution[all]" huggingface_hub \
  --extra-index-url https://download.pytorch.org/whl/cu117 \
  --extra-index-url https://pypi.ngc.nvidia.com \
  --extra-index-url "$BRIA_IDX"

# coordinator only (split/merge — no GPU, so no extra indexes needed):
python3 -m pip install --upgrade "increase-resolution[cpu]" --extra-index-url "$BRIA_IDX"
```

### Engine files

Bria hosts the super-resolution engines on the gated HF repo **`briaai/increase-resolution`** (request
access on Hugging Face; once Bria approves, set `HF_TOKEN`). Fetch them with `huggingface_hub` (added
to the install above; if you used `[gpu]`/`[cpu]`, `pip install huggingface_hub` first):

```python
from huggingface_hub import hf_hub_download
engine_paths = {s: hf_hub_download("briaai/increase-resolution", f"increase_resolution{s}.engine") for s in (2, 4)}
```

> These are TensorRT builds tied to the exact GPU + CUDA/TensorRT runtime above — they only load on a
> matching environment.

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
