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
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install "nvidia-tensorrt==8.4.1.5" --extra-index-url https://pypi.ngc.nvidia.com
pip install "numpy<2"

# the engine's Myelin graph needs this exact cuBLAS (the image provides it; pip does not)
export LD_PRELOAD=/usr/local/cuda-11.7/targets/x86_64-linux/lib/libcublasLt.so.11
```

Then install `increase-resolution` (next section) into this environment and run the notebook.

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
# URL-encode the token so characters like +, /, = don't break the index URL
ENCODED_PASSWORD=$(python3 -c "from urllib.parse import quote; print(quote('${CODE_ARTIFACT_PASSWORD}', safe=''))")
python3 -m pip install --upgrade "increase-resolution" \
  --extra-index-url "https://aws:${ENCODED_PASSWORD}@bria-300465780738.d.codeartifact.us-east-1.amazonaws.com/pypi/bria-increase-resolution/simple/"
```

You must also have `torch` and `tensorrt==8.4.*` installed matching your CUDA 11.7 environment.

## Engine files

Bria provides the super-resolution engines as `.engine` files (one per scale). Download them to a
local folder and pass the paths to the pipeline config:

```text
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
