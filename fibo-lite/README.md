# Fibo-Lite: Open Source Image & Prompt Generation

Welcome to **Fibo-Lite** – an open source, client-facing toolkit for advanced image generation and structured prompt creation. This folder documents the **Python package** path: install **`fibo-lite`** from **AWS CodeArtifact** (via the Bria Engine), pull model weights from **Hugging Face**, and run the **`FiboLite`** pipeline in your own process (see **`code_example.ipynb`**).

## Overview

The included **`code_example.ipynb`** notebook demonstrates:

- Obtaining a CodeArtifact credential from the Bria Engine and installing the **`fibo-lite`** package
- Downloading VLM and diffusion checkpoints from Hugging Face and wiring local paths into the pipeline
- Generating images using the Fibo-Lite pipeline with flexible input options
- Reusing a **structured prompt** from a previous generation step, then refining from an **image + edit** prompt

## Prerequisites

To use Fibo-Lite from this distribution, you will need:

- **Linux** with **CUDA** and an **NVIDIA GPU** (the stack is built for **H100 / H200** class hardware)
- **Python 3.12** and **pip** (or another environment that can install wheels from CodeArtifact)
- **Network** access to the Bria Engine, CodeArtifact, and Hugging Face

## Quick Start

### 1. Configure environment variables

Set the following before running the notebook:

| Variable | Purpose |
|----------|---------|
| `BRIA_API_TOKEN` | Used to request a short-lived **CodeArtifact** token for the **`fibo-lite`** PyPI repository |
| `HF_TOKEN` | Hugging Face token for gated **`briaai/*`** model repos |

**Example (Linux/Mac):**

```bash
export BRIA_API_TOKEN="your-api-token-here"
export HF_TOKEN="your-huggingface-token-here"
```

Or add to your shell profile for persistence:

```bash
echo 'export BRIA_API_TOKEN="your-api-token-here"' >> ~/.bashrc
source ~/.bashrc
```

**Optional:**

- Custom diffusion checkpoint: pass a **local directory** into **`FIBOInternalConfig.model_path`** (for example after downloading or syncing weights); see **`code_example.ipynb`**.

### 2. CodeArtifact token (Bria Engine)

Call the **Bria Engine** once to obtain a PyPI password (this is **authentication only**, not image generation):

```http
GET https://engine.bria-api.com/v2/auth/access/code_artifact?repository=fibo-lite
token: <BRIA_API_TOKEN>
```

**Response shape:**

```json
{
  "result": {
    "authorization_token": "<token>",
    "expiration": "2026-05-04T22:42:09+00:00"
  },
  "request_id": "…"
}
```

Use **`result.authorization_token`** as the **password** for the CodeArtifact PyPI **simple** index (username **`aws`**). Encode the password for use inside a URL if needed (see **`code_example.ipynb`**).

### 3. Install `fibo-lite`

Example (password inlined; prefer URL-encoding in scripts):

```bash
export CODE_ARTIFACT_PASSWORD="<paste authorization_token here>"
python3 -m pip install --upgrade "fibo-lite" \
  --extra-index-url "https://aws:${CODE_ARTIFACT_PASSWORD}@bria-300465780738.d.codeartifact.us-east-1.amazonaws.com/pypi/bria-fibo-lite/simple/"
```

### 4. Run the notebook

```bash
jupyter notebook code_example.ipynb
# or
jupyter lab code_example.ipynb
```

Follow the cells to authenticate, install, download weights, run the pipeline, and experiment with **generate → regenerate from structured prompt → image refine**.

---

## Using the Python package

There is **no HTTP server** in this workflow—you call the library directly.

1. Build a **`FiboLiteConfig`** (VLM + diffuser paths, compile flags, etc.) as in the notebook.
2. **`pipeline = FiboLite(config=...)`**, then **`pipeline.setup()`**.
3. For each job, construct **`FiboLiteInput`** with the fields you need, for example:
   - **`prompt`** only for text-to-image (VLM produces structured JSON, then the diffuser runs).
   - **`structured_prompt`** only to skip the VLM and regenerate from JSON (for example the object returned from a previous **`FiboLiteOutput`**).
   - **`prompt`** plus **`image`** (URL or loaded image, per schema) for image-conditioned refinement.
4. **`result = pipeline.execute(task)`** — use **`result.image`** and **`result.structured_prompt`**.
5. When finished, **`pipeline.cleanup()`** to free GPU memory.

Aspect ratio, **`negative_prompt`**, **`num_inference_steps`**, and **`seed`** are supported on **`FiboLiteInput`**; see the package types and **`code_example.ipynb`** for defaults and validation rules.

### VLM settings in `code_example.ipynb`

The notebook builds **`VLMConfig`** with **`max_model_len`** and **`gpu_memory_utilization`**. Both are **tunable** for your GPU and workload:

| Setting | Role |
|--------|------|
| **`max_model_len`** | Upper bound on sequence length the VLM engine allocates for. Higher values support longer user prompts and larger structured JSON outputs but use more VRAM; lower values reduce VLM memory pressure. |
| **`gpu_memory_utilization`** | Fraction of total GPU memory (0–1) that **vLLM** may use for the VLM. Increase if the VLM runs out of memory; decrease to leave more VRAM for the diffusion model when VLM and diffuser share one GPU. |

You can change them in code where **`VLMConfig(...)`** is constructed
---

## License

Fibo-Lite is open source and available under the [MIT License](https://opensource.org/licenses/MIT). If your wheel or deployment ships a different `LICENSE` file, follow that copy.

---

**Empower your creativity with Fibo-Lite – the open source toolkit for next-generation image and prompt generation.**
