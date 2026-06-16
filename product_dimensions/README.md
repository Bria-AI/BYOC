# Product Dimensions: BYOC Dimension Image Pipeline

This folder documents the **Python package** path for running Product Dimensions in your own environment: install **`product_dimensions`** from the **`bria-product-dimensions`** AWS CodeArtifact repository, then call the local pipeline directly from Python.

Product Dimensions renders marketplace-ready dimension images from product photos. It removes the background when the input does not already have transparency, then draws dimension callouts plus optional title, weight, and capacity text.

## Overview

The included **`code_example.ipynb`** notebook demonstrates:

- Obtaining a short-lived CodeArtifact credential from the Bria Engine.
- Installing **`product_dimensions`** and its Bria dependencies from CodeArtifact.
- Running the local **`ProductDimensions`** pipeline on a sample product image URL.
- Producing a dimension callout image with dual-unit labels, weight, and capacity.

## Prerequisites

To use Product Dimensions from this distribution, you will need:

- Linux with Python 3.10 or newer.
- Network access to the Bria Engine, AWS CodeArtifact, and Hugging Face.
- `BRIA_API_TOKEN`, used to request a short-lived CodeArtifact credential.
- `HF_TOKEN`, used by the underlying `remove_bg` runtime to access `briaai/RMBG-2.0`.
- An NVIDIA GPU runtime for background removal.
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
GET https://engine.prod.bria-api.com/v2/auth/access/code_artifact?repository=bria-product-dimensions
api_token: <BRIA_API_TOKEN>
User-Agent: BriaPlatform/APIdocs/LLMsAgent
```

Use `result.authorization_token` as the password for the CodeArtifact PyPI simple index with username `aws`.

## Install `product_dimensions`

The notebook URL-encodes the token, installs CUDA-compatible PyTorch/Torchvision wheels, then installs Product Dimensions and its Bria dependencies from CodeArtifact. If you already have a token, the equivalent shell commands are:

```bash
python3 -m pip install --upgrade --force-reinstall torch torchvision \
  --index-url "https://download.pytorch.org/whl/cu126"

export CODE_ARTIFACT_PASSWORD="<paste authorization_token here>"
CODEARTIFACT_BASE="https://aws:${CODE_ARTIFACT_PASSWORD}@bria-300465780738.d.codeartifact.us-east-1.amazonaws.com/pypi"
python3 -m pip install --upgrade "product_dimensions" \
  --upgrade-strategy only-if-needed \
  --extra-index-url "${CODEARTIFACT_BASE}/bria-product-dimensions/simple/" \
  --extra-index-url "${CODEARTIFACT_BASE}/bria-remove-bg/simple/" \
  --extra-index-url "${CODEARTIFACT_BASE}/bria-external/simple/"
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

1. Create `ProductDimensions(config=ProductDimensionsConfig(...))`.
2. Call `pipeline.setup()` once.
3. For each image, create `ProductDimensionsInput` with an `image`, one or more `DimensionEntry` objects, and optional `title`, `weight`, and `capacity`.
4. Call `result = pipeline.execute(task)`.
5. Save or display `result.image`. When `output_format="dual"`, `result.overlay` contains the transparent overlay image.
6. Call `pipeline.cleanup()` when finished.

`DimensionEntry.name` accepts `height`, `width_bottom`, `width_top`, `length`, or `depth`. `units_display` controls how duplicate entries with the same name and position are merged (`single`, `dual_bullet`, `dual_slash`, or `dual_parens`). `background` accepts `white`, `cream`, `charcoal`, or a hex color such as `#FFFFFF`.
