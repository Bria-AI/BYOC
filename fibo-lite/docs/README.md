# Fibo Lite

`fibo-lite` generates images from **VGL** — a structured JSON document describing a scene —
running **in-process** on your own GPU. A prompt, a reference image, or a VGL document goes in;
a rendered image and the VGL document it was conditioned on come out.

Two models run behind one call. A **VLM** (`briaai/FIBO-vlm`) turns whatever you sent into a VGL
document, and the **FIBO 1.5 diffuser** (`briaai/Fibo-1.5`) renders that document. You never have
to write VGL by hand — but because the pipeline hands the document back, you can edit it and
re-render, which is the difference between nudging a prompt and changing one attribute of one
object.

## Background

Fibo Lite is the low-latency member of the FIBO 1.5 family. It renders at a **fixed 6 denoising
steps** at **guidance 1.0**, against the 50 steps a full FIBO generation takes — the "lite" is
about latency per image, not a smaller model. The step count is not a knob: the weights are
distilled for it, and guidance above 1.0 would put diffusers on its batch-2 classifier-free path,
which a compiled batch-1 graph cannot serve.

Both halves share a single GPU, so the VLM is deliberately given a small slice of memory
(`gpu_memory_utilization`, default 0.4) and the diffuser takes the rest. To keep the first
request off the compile path, the transformer is compiled ahead of time through
`bria_external.ml.aot` — artifacts resolve per GPU and driver, and a JIT `torch.compile` is the
fallback when no AOT artifact matches.

## Requirements

- **Linux with a CUDA GPU.** H100 / H200 class. The VLM runs on vLLM, which publishes no macOS
  wheel, so the package cannot even be imported off Linux.
- **Python 3.12.** The project is pinned `requires-python = "==3.12.*"`.
- **`BRIA_API_TOKEN`** — exchanged for a short-lived AWS CodeArtifact credential to install the
  package.
- **`HF_TOKEN`** — the `briaai/*` weight repos are gated. Weights are pulled on first `setup()`,
  or downloaded ahead of time and passed as local paths.

Enough GPU memory for both halves at once: the VLM's KV cache must cover one request at
`max_model_len`, or vLLM refuses to start.

## Installation

The package lives in the private `bria-fibo-lite` CodeArtifact repository. Ask the Bria Engine for
a credential, then install with it:

```python
import requests

resp = requests.get(
    "https://engine.prod.bria-api.com/v2/auth/access/code_artifact",
    params={"repository": "bria-fibo-lite"},
    headers={"api_token": BRIA_API_TOKEN},
    timeout=60,
)
resp.raise_for_status()
token = resp.json()["result"]["authorization_token"]
```

```bash
pip install fibo-lite \
  --extra-index-url "https://aws:${TOKEN}@bria-300465780738.d.codeartifact.us-east-1.amazonaws.com/pypi/bria-fibo-lite/simple/"
```

`--extra-index-url`, not `--index-url`: the package's own dependencies come from public PyPI, and
replacing the default index leaves them unresolvable. URL-encode the token — CodeArtifact
credentials contain characters that are not URL-safe.

The credential is short-lived — request a new one when it expires. `BYOC/notebooks/code_example.ipynb`
does all of this end to end, including URL-encoding the token and redacting it from pip's output.

## Usage

```python
from fibo_lite.config import FiboLiteConfig, FiboLiteDiffuserConfig
from fibo_lite.fibo_lite import FiboLite
from fibo_lite.schemas import FiboLiteInput
from fibo_lite.vlm.config import VLMConfig

pipeline = FiboLite(config=FiboLiteConfig(
    vlm_config=VLMConfig(),
    fibo_lite_config=FiboLiteDiffuserConfig(compile_model=False),  # skip compilation on a first run
))
pipeline.setup()  # loads both models and warms the VLM decode grammar

result = pipeline.execute(FiboLiteInput(prompt="a cup of coffee on a wooden table"))
result.image.save("out.png")
print(result.vgl.model_dump_json(indent=2))  # the document the image was rendered from

pipeline.cleanup()
```

`setup()` is expensive and one-time; `execute()` is the per-request call. Call `cleanup()` to
release the GPU.

### What you can send

`FiboLiteInput` accepts `prompt`, `vgl`, and `image` in five meaningful combinations. Which one
you sent decides what the VLM does:

| `prompt` | `vgl` | `image` | What happens |
| --- | --- | --- | --- |
| ✓ | | | **Text to image.** The VLM writes a VGL document from the prompt, the diffuser renders it. |
| | | | Same, from an empty prompt — the VLM invents a scene. |
| ✓ | ✓ | | **Edit a document.** The prompt is an instruction applied to the VGL you passed. |
| | ✓ | | **Re-render.** The document is used verbatim; the VLM is skipped entirely. |
| ✓ | | ✓ | **Edit an image.** The VLM captions your image, applies the instruction, and renders. |
| | | ✓ | **Inspire.** The VLM captions your image into a VGL document and renders from it. |

Anything else — `vgl` and `image` together, for instance — is rejected at validation.

Re-rendering an unchanged document is the cheap path: it skips the VLM, and the same document
with the same seed reproduces the same image byte for byte.

### Input fields

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `prompt` | `str \| None` | `None` | Free text. An editing instruction when `vgl` or `image` is also set. |
| `vgl` | `FiboVGL \| None` | `None` | A VGL document, usually one a previous call returned. |
| `image` | URL / path / PIL / ndarray | `None` | Reference image. |
| `seed` | `int` | `7` | Fixes both the VLM sampling and the diffusion noise. |
| `aspect_ratio` | `"1:1"`, `"2:3"`, `"3:2"`, `"3:4"`, `"4:3"`, `"4:5"`, `"5:4"`, `"9:16"`, `"16:9"` | `1:1` | Resolves to `width`/`height` unless you set those yourself. |
| `width`, `height` | `int \| None` | from `aspect_ratio` | Set both or neither. |
| `num_inference_steps` | `6` | `6` | Fixed by the weights — see Background. |
| `max_tokens` | `int` | `1024` | Cap on the VLM's document. |
| `temperature`, `top_p` | `float` | `0.2`, `0.9` | VLM sampling. |

### Output

`FiboLiteOutput` carries two things:

- **`image`** — a `PIL.Image`, at the requested dimensions.
- **`vgl`** — the `FiboVGL` document the diffuser was conditioned on. Feed it straight back in as
  the `vgl` field to re-render or to edit.

The document always carries `short_description`, `background_setting`, `lighting`, `aesthetics`,
`context`, `objects`, and `artistic_style`: the VLM is constrained to emit that field set even
though VGL itself makes some of them optional.

## Configuration

Three settings classes, all `pydantic-settings` — every field is also an environment variable
under the class's prefix, so a container can be configured without touching code.

`FiboLiteConfig` is the top level and just holds the other two:

| Field | Default | |
| --- | --- | --- |
| `vlm_config` | `VLMConfig()` | The captioning half. |
| `fibo_lite_config` | `FiboLiteDiffuserConfig()` | The rendering half. |

**`VLMConfig`** — environment prefix `VLM_`:

| Field | Default | Effect |
| --- | --- | --- |
| `model_path` | `briaai/FIBO-vlm` | Hugging Face repo, or a local directory. |
| `gpu_memory_utilization` | `0.4` | The VLM's share of the GPU. The diffuser gets what is left, so this is the main dial when the two do not fit. |
| `max_model_len` | `None` | Context length. vLLM will not start unless the remaining memory covers one request at this length. |
| `dtype` | `bfloat16` | |
| `enforce_eager` | `False` | `True` disables CUDA graphs — slower, less memory. |
| `tensor_parallel_size` | `1` | GPUs to shard the VLM across. |
| `attention_backend` | `FLASH_ATTN` | Some GPUs need another; Blackwell RTX 6000 wants `TRITON_ATTN`. |
| `extra_llm_kwargs` | `None` | Passed through to the vLLM constructor. |
| `trust_remote_code` | `True` | |
| `warmup` | `True` | Builds the decode grammar at startup instead of on the first request. |

**`FiboLiteDiffuserConfig`** — environment prefix `FIBO_LITE_`:

| Field | Default | Effect |
| --- | --- | --- |
| `model_path` | `briaai/Fibo-1.5` | Hugging Face repo, or a local directory. |
| `compile_model` | `True` | The master switch. `False` serves the uncompiled pipe — slower per image, but no compile wait. Worth turning off for a first run. |
| `aot_dir` | `None` | Where AOT artifacts live. Set it (with `compile_model`) to take the AOT path instead of JIT. |
| `aot_s3_uri` | `None` | The first container exports and uploads here; later ones download instead of recompiling. |
| `aot_hardware_versioned` | `True` | Resolves artifacts under `<base>/<gpu>/<driver>`, so a different GPU or driver does not load an incompatible export. |
| `aot_enabled` | `True` | `False` forces JIT without unsetting `aot_dir`. |

An AOT failure is never fatal: the pipeline clears the marker, undoes the partial attach, and
falls back to `torch.compile`.

## Examples

- `BYOC/notebooks/code_example.ipynb` — the full walkthrough: credential, install, weight
  download, configuration, and the three main flows.
- `examples/simple.py` — the same flows as a script, plus re-rendering an unchanged document.
