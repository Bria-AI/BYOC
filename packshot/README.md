# Packshot: BYOC Product Packshot Pipeline

This folder documents the **Python package** path for running Packshot in your own environment: install **`packshot`** from the **`bria-packshot`** AWS CodeArtifact repository, then call the local pipeline directly from Python.

Packshot creates a centered `2000x2000` product image from an input image. It removes the background, resizes the product with consistent padding, and places it on a white, transparent, or custom hex-color background.
## Overview

The included **`code_example.ipynb`** notebook demonstrates:

- Obtaining a short-lived CodeArtifact credential from the Bria Engine.
- Installing **`packshot`** from **`bria-packshot`**.
- Running the local **`Packshot`** pipeline on a sample product image URL.
- Producing both white-background and transparent-background packshot outputs.

## Prerequisites

To use Packshot from this distribution, you will need:

- Linux with Python 3.10 or newer.
- Network access to the Bria Engine and AWS CodeArtifact.
- `BRIA_API_TOKEN`, used to request a short-lived CodeArtifact credential.
- `HF_TOKEN`, used by the Packshot runtime to access required Hugging Face assets.
- An NVIDIA GPU runtime for the underlying `cutout` package.
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
GET https://engine.prod.bria-api.com/v2/auth/access/code_artifact?repository=bria-packshot
api_token: <BRIA_API_TOKEN>
User-Agent: BriaPlatform/APIdocs/LLMsAgent
```

Use `result.authorization_token` as the password for the CodeArtifact PyPI simple index with username `aws`.

## Install `packshot`

The notebook URL-encodes the token, installs CUDA-compatible PyTorch/Torchvision wheels, then installs Packshot from CodeArtifact. If you already have a token, the equivalent shell commands are:

```bash
python3 -m pip install --upgrade --force-reinstall torch torchvision \
  --index-url "https://download.pytorch.org/whl/cu126"

export CODE_ARTIFACT_PASSWORD="<paste authorization_token here>"
python3 -m pip install --upgrade "numpy>=1.24,<1.25" "packshot" \
  --upgrade-strategy only-if-needed \
  --extra-index-url "https://aws:${CODE_ARTIFACT_PASSWORD}@bria-300465780738.d.codeartifact.us-east-1.amazonaws.com/pypi/bria-packshot/simple/"
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

1. Create `Packshot(config=PackshotConfig(...))`.
2. Call `packshot.setup()` once.
3. For each image, create `PackshotInput(image=..., background_color=..., preserve_alpha=...)`.
4. Call `result = packshot.execute(task)`.
5. Save or display `result.image`.
6. Call `packshot.cleanup()` when finished.

`background_color` accepts `None` for white, a six-character hex color such as `#FFFFFF`, or `transparent`. `preserve_alpha=True` keeps alpha in the output image when using transparent backgrounds.
