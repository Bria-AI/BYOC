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
| **Remove Background** | Segments the foreground subject and returns a transparent cutout, powered by RMBG 2.0. Licensed for self-hosting. | [`remove-bg/`](remove-bg/docs/README.md) |
| **FIBO 2** | Generates images from a prompt and edits images from an instruction, with structured captions (VGL) you can reuse. Licensed for self-hosting. | [`fibo-2/`](fibo-2/docs/README.md) |
| **Packshot** | Produces a centered, marketplace-ready product image on a white, transparent, or custom-color background. | [`packshot/`](packshot/README.md) |
| **Product Dimensions** | Renders marketplace dimension-callout images with optional title, weight, and capacity text. | [`product_dimensions/`](product_dimensions/README.md) |
| **Erase** | Removes a masked object and inpaints the region (LaMa coarse fill + SDXL ControlNet refine). | [`erase/`](erase/README.md) |
| **Increase Resolution** | 2×/4× super-resolution via a tiled TensorRT engine. | [`increase-resolution/`](increase-resolution/README.md) |
| **Fibo-Lite** | Open-source image generation and structured prompt creation. | [`fibo-lite/`](fibo-lite/README.md) |
| **Attribution Agent** | Computes image/video embeddings and reports them to the Bria API for usage attribution. | [`attribution-agent/`](attribution-agent/README.md) |

Each folder's README covers its own prerequisites, environment variables, and exact GPU/CUDA
requirements — these vary per pipeline, so check the one you need before starting.

No license yet? Remove Background and FIBO 2 are sold as self-hosting licenses, per model. Each
model's page is `https://platform.bria.ai/licensing/pricing?model=<model>`, with `rmbg-2.0` or
`fibo-2` as the model. The other pipelines are Enterprise deliveries: talk to Bria.

## How it works, in general

1. **Get credentials.** Every pipeline needs a `BRIA_API_TOKEN`: any API key of your Bria
   account, from the platform's API keys page. For the models Bria licenses for self-hosting
   (Remove Background, FIBO 2) that is all: the Bria Engine issues the package credential for the
   models your account holds a license for, and the model weights are downloaded from Bria on the
   first `setup()`. The other pipelines are Enterprise deliveries; their READMEs say what else they
   need, such as an `HF_TOKEN` for Hugging Face model weights.
2. **Install the package** from Bria's CodeArtifact repository (each pipeline names its own,
   e.g. `bria-remove-bg`, `bria-packshot`, `bria-erase`).
3. **Run the notebook** (`code_example.ipynb` in each folder) to see the pipeline load, run
   inference on a sample image, and produce output — then adapt it to your own images and pipeline.

## Prerequisites (common to all pipelines)

- Linux with an NVIDIA GPU (exact model/VRAM requirements vary by pipeline — see each README).
- Network access to the Bria Engine and AWS CodeArtifact. The licensed models (Remove Background,
  FIBO 2) also download their weights from Bria's mirror on CloudFront. Hugging Face access is
  needed only by the pipelines whose README asks for it.
- A Bria API key (`BRIA_API_TOKEN`). A Hugging Face token (`HF_TOKEN`) only for the pipelines whose
  README asks for it.

```bash
export BRIA_API_TOKEN="your-api-key-here"
```

## Getting help

If you run into issues getting a pipeline running in your environment, reach out via
[bria.ai/contact-us](https://bria.ai/contact-us).
