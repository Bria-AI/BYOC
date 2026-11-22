# rmbg-3

`rmbg-3` removes image backgrounds with BRIA RMBG 3, running **in-process** on the GPU (no inference server). It
accepts image inputs such as URLs, base64 strings, numpy arrays and PIL images, and returns each one as a PIL image
with an alpha channel: the original pixels, with the background transparent. It can also return the mask.

RMBG 3 is a fibo-2 edit model distilled to draw an image's alpha matte, with soft edges for hair, fur and
semi-transparent objects. It is larger and slower than RMBG 2.0: it needs a GPU with at least 40 GB of memory.

## Quick start

**Prerequisites:** a Bria API token (`BRIA_API_TOKEN`, from https://platform.bria.ai). Any token works if your account
holds a self-hosting license for RMBG 3. You also need Python 3.11 or 3.12 and an NVIDIA GPU.

Exchange your API token for a short-lived CodeArtifact credential and install `rmbg-3`:

```bash
export BRIA_API_TOKEN=...   # from https://platform.bria.ai

CA=$(curl -s "https://engine.prod.bria-api.com/v2/outpost/access/code_artifact?repository=bria-rmbg-3" \
      -H "api_token: $BRIA_API_TOKEN" \
      | python -c "import sys, json; print(json.load(sys.stdin)['result']['authorization_token'])")

pip install --upgrade "rmbg-3>=0.1.0" \
  --extra-index-url "https://aws:$CA@bria-300465780738.d.codeartifact.us-east-1.amazonaws.com/pypi/bria-rmbg-3/simple/"
```

Then remove a background:

```python
import os

from rmbg_3 import RMBG3, RMBG3Config, RMBG3Input

pipe = RMBG3(config=RMBG3Config(api_token=os.environ["BRIA_API_TOKEN"]))
pipe.setup()  # checks your license, then downloads the weights from Bria into ~/.bria/models on first run

result = pipe.execute([RMBG3Input(image="https://bria-test-images.s3.us-east-1.amazonaws.com/strawberry.jpg")])[0]
result.image.save("cutout.png")  # RGBA cutout with the background removed

pipe.cleanup()
```

The [`notebooks/code_example.ipynb`](../notebooks/code_example.ipynb) bundle walks through the same flow end to end
(auth, install, run).

## Pipeline

`RMBG3` follows the `common.pipeline` lifecycle: `setup()` loads the model, `execute()` runs a list of `RMBG3Input`s,
and `cleanup()` frees the model. Its request and result are `remove_bg`'s, so code written for `RemoveBG` moves over
by changing the class names.

- **Input** (`RMBG3Input`): an image (URL, base64, numpy array, or PIL image, via `bria_external.image`), plus
  `return_mask`, an optional `api_token` (takes precedence over the config's) and an optional `source_request_id`,
  your own id for the request, stored in the attribution record.
- **Output** (`RMBG3Output`): the RGBA cutout at the input's size, and the grayscale `mask` (white = foreground) when
  `return_mask=True`.

## License and weights

`setup()` first checks that your account holds a self-hosting license for RMBG 3 (any API token of the account
identifies it; set it as `api_token`, `RMBG_3_API_TOKEN` or `BRIA_API_TOKEN`). If the license is missing or has ended,
`setup()` raises `common.outpost.LicenseError` with the reason and a link to your licenses page, and nothing is
loaded. The weights are then downloaded from Bria into `~/.bria/models/rmbg-3.0/<version>/` (override the root with
`BRIA_HOME`), verified, and reused on later starts. A connection to Bria is needed when a model starts: the Bria
Engine and Bria's weights CDN (a `*.cloudfront.net` host). Each cutout sends an attribution record, which is how usage
against your plan is counted. No Hugging Face account and no LLM are needed.

## Configuration

Environment variables use the `RMBG_3_` prefix.

| Field | Default | What it does |
|---|---|---|
| `api_token` | `BRIA_API_TOKEN` | The key the license check, the weights download and attribution run under. |
| `compile_model` | `False` | Compiles the model's repeated blocks at setup: a slower start, faster cutouts. |
| `model_path` / `model_revision` | Bria's mirror | A diffusers-format checkpoint (local folder or Hugging Face repo id) to load instead. |
| `transformer_path` / `transformer_revision` | `model_path`'s | The RMBG 3 transformer, when it is kept apart from the rest of the checkpoint. |

## Non-obvious decisions

- **A fixed instruction, no prompt.** RMBG 3 is a fibo-2 edit model: it edits the image by a fixed matte instruction
  and the result becomes the alpha. The instruction is sent as a ready VGL, so no LLM runs.
- **A fixed seed.** The same image always gets the same cutout.
- **4 steps, no guidance.** RMBG 3 is distilled to fibo-2 turbo's schedule; neither is configurable.
- **The image's own alpha is replaced.** A cutout sent in is cut again, as `remove_bg` does.
- Attribution is sent on a background thread per output so it never blocks `execute()`. At exit the interpreter
  waits, for a bounded time, for the ones still in flight, so a script that ends right after its last cutout still
  counts it.
