# Remove Background: BYOC Background-Removal Pipeline

This folder documents the **Python package** path for running Bria **Remove Background** in your own environment: install **`remove_bg`** from the **`bria-remove-bg`** AWS CodeArtifact repository, then call the local pipeline directly from Python.

Remove Background segments the foreground subject out of an input image and returns a cutout with a transparent alpha channel, powered by the **RMBG-2.0** model. It is the same background-removal component reused internally by Packshot and Product Dimensions, exposed here as a standalone pipeline.

## Overview

The included **`code_example.ipynb`** notebook demonstrates:

- Obtaining a short-lived CodeArtifact credential from the Bria Engine.
- Installing **`remove_bg`** from **`bria-remove-bg`**.
- Running the local **`RemoveBg`** pipeline on a sample product image URL.
- Saving the transparent cutout and a checkerboard preview to visualize the alpha channel.

## Prerequisites

To use Remove Background from this distribution, you will need:

- Linux with Python 3.10 or newer.
- Network access to the Bria Engine, AWS CodeArtifact, and Hugging Face.
- `BRIA_API_TOKEN`, used to request a short-lived CodeArtifact credential.
- `HF_TOKEN`, used by the Remove Background runtime to access `briaai/RMBG-2.0`.
- An NVIDIA GPU runtime.
- A CUDA-compatible NVIDIA driver. For CUDA 12.x drivers, this example installs PyTorch/Torchvision from the CUDA 12.6 PyTorch wheel index.

Check the driver-supported CUDA version with:

```bash
nvidia-smi
```

Set your API token and Hugging Face token before running the notebook:

```bash
export BRIA_API_TOKEN="your-api-token-here"
export HF_TOKEN="your-hugging-face-token-here"
```

## CodeArtifact Token

Call the Bria Engine once to obtain a PyPI password for the CodeArtifact repository:

```http
GET https://engine.prod.bria-api.com/v2/auth/access/code_artifact?repository=bria-remove-bg
api_token: <BRIA_API_TOKEN>
User-Agent: BriaPlatform/APIdocs/LLMsAgent
```

Use `result.authorization_token` as the password for the CodeArtifact PyPI simple index with username `aws`.

## Install `remove_bg`

The notebook URL-encodes the token, installs CUDA-compatible PyTorch/Torchvision wheels, then installs Remove Background from CodeArtifact. If you already have a token, the equivalent shell commands are:

```bash
python3 -m pip install --upgrade --force-reinstall torch torchvision \
  --index-url "https://download.pytorch.org/whl/cu126"

export CODE_ARTIFACT_PASSWORD="<paste authorization_token here>"
ENCODED_PASSWORD=$(python3 -c "from urllib.parse import quote; print(quote('${CODE_ARTIFACT_PASSWORD}', safe=''))")
BRIA_IDX="https://aws:${ENCODED_PASSWORD}@bria-300465780738.d.codeartifact.us-east-1.amazonaws.com/pypi/bria-remove-bg/simple/"

python3 -m pip install --upgrade "remove_bg" --extra-index-url "$BRIA_IDX"
```

## Run The Notebook

```bash
jupyter notebook code_example.ipynb
# or
jupyter lab code_example.ipynb
```

The notebook saves generated files under `outputs/`.

## Using The Python Package

You call the library directly:

1. Create `RemoveBg(config=RemoveBgConfig(...))`.
2. Call `remove_bg.setup()` once.
3. For each image, create `RemoveBgInput(image=...)`.
4. Call `result = remove_bg.execute(task)`.
5. Save or display `result.image` — an RGBA image with the background made transparent.
6. Call `remove_bg.cleanup()` when finished.

`RemoveBgConfig(compile_model=False)` disables model compilation for a simpler, faster-starting first run; enable it in a long-lived deployment for better steady-state throughput.

## Note on package/API names

This README follows the same install/config/pipeline shape used by [`erase`](../erase/README.md), [`packshot`](../packshot/README.md), and [`product_dimensions`](../product_dimensions/README.md) (which reuses this component internally via a nested `rmbg_config`). Confirm the exact repository name (`bria-remove-bg`) and class names (`RemoveBg`, `RemoveBgConfig`, `RemoveBgInput`) against the actual published package before distributing — they are inferred from the sibling examples' conventions rather than pulled from the `remove_bg` package's own source.
