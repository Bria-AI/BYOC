# Increase Resolution: BYOC Super-Resolution Pipeline

Run Bria **Increase Resolution** (2×/4× super-resolution) in your own environment. An image is split
into overlapping tiles, each tile is upscaled by Bria's super-resolution model in-process, and the
tiles are merged back into the full-resolution result.

The model is **native PyTorch + `torch.compile`**, so it runs on **any modern CUDA GPU** — no
TensorRT engine, no A10-only build, no NVIDIA container. Two dedicated models are shipped: one for
**2×** and one for **4×**.

## Prerequisites

- **GPU:** any CUDA GPU with enough VRAM (A10/A10G, L4, L40S, A100, H100, RTX, …) and a driver that
  matches your installed PyTorch CUDA build.
- **Python 3.10** and **PyTorch ≥ 2.1** (for `torch.compile`). Install the torch build that matches
  your driver (e.g. CUDA 12.1 → `--index-url https://download.pytorch.org/whl/cu121`).
- **`BRIA_API_TOKEN`** — a custom-plan Bria token (used to get a short-lived CodeArtifact credential
  for the package index).
- **`HF_TOKEN`** — with **approved access to the gated `briaai/increase-resolution`** HF repo (request
  access on Hugging Face; Bria approves) — the model weights are hosted there and downloaded on first
  `setup()`.
- Network access to the Bria Engine, AWS CodeArtifact, and Hugging Face.

> **First `setup()` is slow** — `torch.compile` compiles the model once (tens of seconds to a few
> minutes, per scale) and the pipeline **warms it up** so later calls are fast. This is a one-time
> cost per process.

## Quickstart — run the example notebook

No container needed. On your GPU machine, create a venv, launch Jupyter, and run the notebook — it
fetches the CodeArtifact token → installs `increase-resolution` → downloads the model weights from
HF → upscales. Two walkthroughs are included:

- **`code_example.ipynb`** — simple image → image on one machine.
- **`code_example_distributed.ipynb`** — tile-level flow for spreading GPU work across machines.

**1. Create the environment + launch Jupyter** (from this folder, on your GPU box):

```bash
export BRIA_API_TOKEN="your-custom-plan-token"
export HF_TOKEN="your-hf-token"     # must have approved access to gated briaai/increase-resolution

python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip jupyterlab ipykernel
# install a torch build matching your CUDA driver first, e.g. CUDA 12.1:
pip install "torch>=2.1" --index-url https://download.pytorch.org/whl/cu121
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

**2. Connect your editor to it.** In VS Code / Cursor: open `code_example.ipynb` → kernel picker
(top-right) → **Select Another Kernel… → Existing Jupyter Server…** → enter `http://localhost:8888`
and the **token** printed in the log (e.g. `http://localhost:8888?token=abc`) → pick the **Python 3**
kernel.

**3. Run the notebook top-to-bottom.** It fetches the CodeArtifact token, `pip install`s the package,
downloads the model weights from `briaai/increase-resolution` with your `HF_TOKEN`, and upscales the
sample at 2× and 4×. Outputs are saved under `outputs/`. `code_example_distributed.ipynb` runs the
same way.

---

## Installing the package directly (without the notebook)

Everything below is what the notebook automates. Do it by hand **only** if you're integrating the
`increase-resolution` package into your own code / CLI.

### CodeArtifact token

Call the Bria Engine once to obtain a PyPI password for the CodeArtifact repository:

```http
GET https://engine.prod.bria-api.com/v2/auth/access/code_artifact?repository=bria-increase-res
api_token: <BRIA_API_TOKEN>
```

Use `result.authorization_token` as the password for the CodeArtifact PyPI simple index (username `aws`).

### Install `increase-resolution`

| Extra | Role | GPU / torch? |
|---|---|---|
| `[all]` | single-machine full pipeline (`execute`) | yes (torch) |
| `[gpu]` | worker: tile inference (`TileWorker.infer`) | yes (torch) |
| `[cpu]` | coordinator: `split` / `merge` | no (no torch) |

```bash
export CODE_ARTIFACT_PASSWORD="<paste authorization_token here>"
# URL-encode the token so characters like +, /, = don't break the index URL
ENCODED_PASSWORD=$(python3 -c "from urllib.parse import quote; print(quote('${CODE_ARTIFACT_PASSWORD}', safe=''))")
BRIA_IDX="https://aws:${ENCODED_PASSWORD}@bria-300465780738.d.codeartifact.us-east-1.amazonaws.com/pypi/bria-increase-res/simple/"

# full pipeline (torch is pulled automatically; ensure it matches your CUDA driver — see below):
python3 -m pip install --upgrade "increase-resolution[all]" --extra-index-url "$BRIA_IDX"

# coordinator only (split/merge — no GPU/torch needed):
python3 -m pip install --upgrade "increase-resolution[cpu]" --extra-index-url "$BRIA_IDX"
```

If the default torch build doesn't match your CUDA driver, install the matching one explicitly, e.g.:

```bash
python3 -m pip install "torch>=2.1" --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
```

### Weights

Bria hosts the model weights on the gated HF repo **`briaai/increase-resolution`** (request access on
Hugging Face; once Bria approves, set `HF_TOKEN`). They are **downloaded automatically** on the first
`setup()` into `~/.cache/bria/increase-resolution/` — you don't fetch them by hand.

## Using the Python package

Two dedicated models — construct the one for the scale you want. There is **no** `desired_increase`
argument; the model class is the scale.

```python
from increase_resolution import IncreaseResolution2x, IncreaseResolution4x, IncreaseResolutionInput

model = IncreaseResolution4x()     # or IncreaseResolution2x()
model.setup()                      # downloads model weights -> torch.compile -> warmup (one-time)

result = model.execute(IncreaseResolutionInput(image="https://.../photo.jpg"))
result.image.save("upscaled.png")

model.cleanup()
```

`image` accepts a URL, base64 string, numpy array, or PIL image. Set `preserve_alpha=True` on the
input to keep an RGBA image's alpha in the output.

Configuration flags (via `IncreaseResolutionConfig`, e.g. `IncreaseResolution4x(config=...)`):
`precision` (`"fp32"` default / `"fp16"` for extra speed), `compile_model` (default
`True`), `warmup_iters`, `hf_token`, `cache_folder`, tiling params, and `max_output_dimension`.

## Distributed (tile-level) usage

For high throughput or many GPUs across separate machines, run the tile inference on your own fleet.
The pipeline exposes the three steps so the transport in the middle is yours (queue / RPC / Ray / …).
A `TileWorker` is bound to a scale; the coordinator downloads that scale's weights once and ships the
local path to the workers:

```python
from increase_resolution import split, merge, TileWorker
from increase_resolution.model import download_weights

weights = download_weights(4, hf_repo_id="briaai/increase-resolution", cache_folder="~/.cache/bria/increase-resolution/")

res = split(image, scale=4)                        # coordinator (CPU): independent numpy tiles + layout
# --- ship res.tiles to your worker machines; each builds one TileWorker for this scale ---
worker = TileWorker(scale=4, weights_path=weights).setup()
upscaled = [worker.infer(t) for t in res.tiles]    # <-- replace with your cross-machine fan-out
# ----------------------------------------------------------------------------------------------
merge(upscaled, res.layout, res.alpha).save("upscaled.png")   # coordinator (CPU): stitch back
```

`split`/`merge` need no GPU or weights; only the workers do. Tiles are plain numpy arrays so your
transport can serialize them. The output is identical to the single-call `execute`.
