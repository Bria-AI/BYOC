# Attribution Agent: BYOC Attribution Pipeline

This folder documents the **Python package** path for running Bria **Attribution** in your own
environment: request a short-lived CodeArtifact credential from the Bria Engine, install
**`bria_attribution`** from the private **`attribution`** AWS CodeArtifact repository, then call the
local `BriaAttributionAgent` directly from Python. No Triton, no inference server — embeddings are
computed in-process and submitted to the Bria API for attribution tracking.

The agent computes embeddings for an image or video and submits them (or the raw asset) to the Bria
API so usage of a given generative model can be attributed back to its source content:

- **Image attribution** — always available. Computes image embeddings with `image-attribution` and
  reports them under an `AttributionModel` (e.g. `IMAGE_RMBG_1`, `IMAGE_FIBO_1`, `IMAGE_ERASER_1`, ...).
- **Video attribution** — requires the optional `video` extra. Computes stills + motion embeddings
  with `video-attribution` and reports them under a video `AttributionModel`
  (e.g. `VIDEO_RMBG_1`, `VIDEO_GEN_1`, `VIDEO_ERASER_1`, ...).

## Overview

The included **`code_example.ipynb`** notebook demonstrates:

- Obtaining a short-lived CodeArtifact credential from the Bria Engine.
- Installing **`bria_attribution`** (and its `video` extra) from the **`attribution`** repository.
- Creating a **`BriaAttributionAgent`** and calling `.setup()`.
- Running image attribution on a sample image URL.
- Running video attribution on a sample video URL.

## Prerequisites

Attribution runs the standard `torch` + `transformers` stack (no TensorRT engine, no diffusion
model), so it's portable across CPU and GPU environments:

- **Python:** **3.10+**.
- **Device:** CPU works out of the box (`device="cpu"`); pass `device="cuda"` for faster embedding
  on an NVIDIA GPU.
- **`BRIA_API_TOKEN`** — to request a short-lived CodeArtifact credential and to submit attribution
  results to the Bria API.
- Network access to the Bria Engine and AWS CodeArtifact.

```bash
export BRIA_API_TOKEN="your-api-token-here"
```

## CodeArtifact token

Call the Bria Engine once to obtain a PyPI password for the `attribution` CodeArtifact repository:

```http
GET https://engine.prod.bria-api.com/v2/auth/access/code_artifact?repository=attribution
api_token: <BRIA_API_TOKEN>
```

Use `result.authorization_token` as the password for the CodeArtifact PyPI simple index with
username `aws`.

## Install `bria_attribution`

```bash
export CODE_ARTIFACT_PASSWORD="<paste authorization_token here>"
# URL-encode the token so characters like +, /, = don't break the index URL
ENCODED_PASSWORD=$(python3 -c "from urllib.parse import quote; print(quote('${CODE_ARTIFACT_PASSWORD}', safe=''))")
HOST="bria-300465780738.d.codeartifact.us-east-1.amazonaws.com"
BRIA_IDX="https://aws:${ENCODED_PASSWORD}@${HOST}/pypi/attribution/simple/"

python3 -m pip install --upgrade "bria_attribution" --extra-index-url "$BRIA_IDX"
# with video support:
python3 -m pip install --upgrade "bria_attribution[video]" --extra-index-url "$BRIA_IDX"
```

## Run the notebook

Open **`code_example.ipynb`** and run the cells top to bottom. It requests the CodeArtifact
credential, installs the package, sets up the agent, then runs image and video attribution on
sample URLs and prints the resulting `request_id` for each.

## Minimal usage

```python
from attribution_agent import BriaAttributionAgent, AttributionModel

agent = BriaAttributionAgent(device="cpu")
agent.setup()

# Image
from PIL import Image
image = Image.open("input.png")
request_id = agent.execute_image(
    image=image,
    model=AttributionModel.IMAGE_RMBG_1,
    api_token="your-api-token",
    agent="your-agent-name",
)

# Video (requires the `video` extra)
request_id = agent.execute_video(
    video="path/or/url/to/video.mp4",
    model=AttributionModel.VIDEO_RMBG_1,
    api_token="your-api-token",
    agent="your-agent-name",
)
```
