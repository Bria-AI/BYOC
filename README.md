# BYOC — Bring Your Own Compute

Run Bria's image AI pipelines directly on your own GPU infrastructure — no calls to the hosted
Bria API, no inference server to stand up. Each pipeline here is a Python package you install from
Bria's private AWS CodeArtifact repository, paired with a runnable notebook that walks through
setup, credentials, and a first inference end to end.

This is the right place to start if you need Bria's models running **in-process, on hardware you
control** — for data residency, latency, air-gapped environments, or cost at scale.

## Pipelines

| Pipeline | What it does | Docs |
|---|---|---|
| **Remove Background** | Segments the foreground subject and returns a transparent cutout, powered by RMBG-2.0. | [`remove_bg/`](remove_bg/README.md) |
| **Packshot** | Produces a centered, marketplace-ready product image on a white, transparent, or custom-color background. | [`packshot/`](packshot/README.md) |
| **Product Dimensions** | Renders marketplace dimension-callout images with optional title, weight, and capacity text. | [`product_dimensions/`](product_dimensions/README.md) |
| **Erase** | Removes a masked object and inpaints the region (LaMa coarse fill + SDXL ControlNet refine). | [`erase/`](erase/README.md) |
| **Increase Resolution** | 2×/4× super-resolution via a tiled TensorRT engine. | [`increase-resolution/`](increase-resolution/README.md) |
| **Fibo-Lite** | Open-source image generation and structured prompt creation. | [`fibo-lite/`](fibo-lite/README.md) |
| **Attribution Agent** | Computes image/video embeddings and reports them to the Bria API for usage attribution. | [`attribution-agent/`](attribution-agent/README.md) |

Each folder's README covers its own prerequisites, environment variables, and exact GPU/CUDA
requirements — these vary per pipeline, so check the one you need before starting.

## How it works, in general

1. **Get credentials.** Every pipeline needs a `BRIA_API_TOKEN`, which the included notebook
   exchanges for a short-lived AWS CodeArtifact credential. Most pipelines also need an `HF_TOKEN`
   for Hugging Face model weights.
2. **Install the package** from Bria's CodeArtifact repository (each pipeline names its own,
   e.g. `bria-remove-bg`, `bria-packshot`, `bria-erase`).
3. **Run the notebook** (`code_example.ipynb` in each folder) to see the pipeline load, run
   inference on a sample image, and produce output — then adapt it to your own images and pipeline.

## Prerequisites (common to all pipelines)

- Linux with an NVIDIA GPU (exact model/VRAM requirements vary by pipeline — see each README).
- Network access to the Bria Engine, AWS CodeArtifact, and Hugging Face.
- A Bria API token (`BRIA_API_TOKEN`) and, for most pipelines, a Hugging Face token (`HF_TOKEN`).

```bash
export BRIA_API_TOKEN="your-api-token-here"
export HF_TOKEN="your-hugging-face-token-here"
```

## Getting help

If you run into issues getting a pipeline running in your environment, reach out via
[bria.ai/contact-us](https://bria.ai/contact-us).
