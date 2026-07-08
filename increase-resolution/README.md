# Increase Resolution: BYOC Super-Resolution Pipeline

This folder documents the **Python package** path for running Increase Resolution in your own
environment: install **`increase-resolution`** from the **`bria-increase-resolution`** AWS
CodeArtifact repository, download the Bria super-resolution **TensorRT engine(s)**, then call the
local pipeline directly from Python.

Increase Resolution upscales an image **2×** or **4×** using a SwinIR super-resolution model. The
image is split into overlapping tiles, each tile is upscaled by the engine in-process, and the tiles
are merged back into the full-resolution result.

## Overview

The included **`code_example.ipynb`** notebook demonstrates:

- Obtaining a short-lived CodeArtifact credential from the Bria Engine.
- Installing **`increase-resolution`** from **`bria-increase-resolution`**.
- Downloading the super-resolution **`.engine`** files provided by Bria.
- Running the local **`IncreaseResolution`** pipeline on a sample image at 2× and 4×.

## Prerequisites — exact environment (required)

The pipeline runs a **TensorRT engine**, which is compiled for one specific runtime. It will only
load on a matching environment:

- **GPU:** NVIDIA **A10** (Ampere), or another Ampere GPU of the same compute capability.
- **CUDA:** **11.7**.
- **TensorRT:** **8.4.x**.
- **Python:** 3.10.
- Network access to the Bria Engine and AWS CodeArtifact.
- `BRIA_API_TOKEN`, used to request a short-lived CodeArtifact credential.

> These versions are not flexible. A TensorRT engine is tied to the GPU architecture and the
> TensorRT/CUDA versions it was built with; a different GPU or newer TensorRT will fail to load it.

```bash
export BRIA_API_TOKEN="your-api-token-here"
```

## CodeArtifact Token

Call the Bria Engine once to obtain a PyPI password for the CodeArtifact repository:

```http
GET https://engine.prod.bria-api.com/v2/auth/access/code_artifact?repository=bria-increase-resolution
api_token: <BRIA_API_TOKEN>
```

Use `result.authorization_token` as the password for the CodeArtifact PyPI simple index with
username `aws`.

## Install `increase-resolution`

```bash
export CODE_ARTIFACT_PASSWORD="<paste authorization_token here>"
python3 -m pip install --upgrade "increase-resolution" \
  --extra-index-url "https://aws:${CODE_ARTIFACT_PASSWORD}@bria-300465780738.d.codeartifact.us-east-1.amazonaws.com/pypi/bria-increase-resolution/simple/"
```

You must also have `torch` and `tensorrt==8.4.*` installed matching your CUDA 11.7 environment.

## Engine files

Bria provides the super-resolution engines as `.engine` files (one per scale). Download them to a
local folder and pass the paths to the pipeline config:

```
increase_resolution2.engine   # 2x
increase_resolution4.engine   # 4x
```

## Run the notebook

```bash
jupyter notebook code_example.ipynb
```

The notebook saves generated files under `outputs/`.

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
