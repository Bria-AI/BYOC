# fibo-2-turbo

`fibo-2-turbo` edits images with BRIA FIBO 2 Turbo, running **in-process** on the GPU (no inference server). Give it
1 to 4 images and an instruction, and it returns one edited image. A mask limits the edit to one region of a single
image.

**FIBO 2 Turbo edits images only.** It does not generate an image from a prompt alone. Generation is FIBO 2, the
`fibo-2` package, which has its own license.

FIBO 2 Turbo is FIBO 2's edit model, distilled to run in 4 steps without guidance. The instruction is either a
free-form `prompt`, which an LLM of your choice turns into a VGL (the JSON document the model reads), or a ready `vgl`
that carries an `edit_instruction`, which needs no LLM. It needs an NVIDIA GPU with at least 32 GB of memory.

## Quick start

**Prerequisites:** a Bria API token (`BRIA_API_TOKEN`, from https://platform.bria.ai). Any token works if your account
holds a self-hosting license for FIBO 2 Turbo (https://platform.bria.ai/outpost/licenses?model=fibo-2-turbo). You also
need Python 3.11 or 3.12 and an NVIDIA GPU.

Exchange your API token for a short-lived CodeArtifact credential and install `fibo-2-turbo`:

```bash
export BRIA_API_TOKEN=...   # from https://platform.bria.ai

CA=$(curl -s "https://engine.prod.bria-api.com/v2/outpost/access/code_artifact?repository=bria-fibo-2-turbo" \
      -H "api_token: $BRIA_API_TOKEN" \
      | python -c "import sys, json; print(json.load(sys.stdin)['result']['authorization_token'])")

pip install --upgrade "fibo-2-turbo>=0.1.0" \
  --extra-index-url "https://aws:$CA@bria-300465780738.d.codeartifact.us-east-1.amazonaws.com/pypi/bria-fibo-2-turbo/simple/"
```

The same repository also supplies the `fibo-2` package that FIBO 2 Turbo runs on, the build of `diffusers` that
`fibo-2` needs, and Bria's other packages. Everything else comes from PyPI.

Then edit an image:

```python
import os
from io import BytesIO
from urllib.request import urlopen

from PIL import Image

from fibo_2_turbo import Fibo2Turbo, Fibo2TurboConfig, Fibo2TurboInput

pipe = Fibo2Turbo(config=Fibo2TurboConfig(api_token=os.environ["BRIA_API_TOKEN"]))
pipe.setup()  # checks your license, then downloads the weights from Bria into ~/.bria/models on first run

photo = Image.open(BytesIO(urlopen("https://bria-test-images.s3.us-east-1.amazonaws.com/strawberry.jpg").read()))

# A ready VGL that carries the edit instruction: no LLM runs.
vgl = '{"edit_instruction": "Make the strawberry blue"}'
result = pipe.execute(Fibo2TurboInput(images=[photo], vgl=vgl, seed=42))
result.image.save("edited.png")

# The same edit as free-form text needs an LLM (see "The LLM for a free-form prompt" below):
# result = pipe.execute(Fibo2TurboInput(images=[photo], prompt="Make the strawberry blue", seed=42))

pipe.cleanup()
```

The [`notebooks/code_example.ipynb`](../notebooks/code_example.ipynb) bundle walks through the same flow end to end
(auth, install, run).

## Pipeline

`Fibo2Turbo` follows the `common.pipeline` lifecycle: `setup()` checks the license and loads the model, `execute()`
runs one `Fibo2TurboInput` and returns one edited image, and `cleanup()` frees the model.

- **Input** (`Fibo2TurboInput`):
  - `images`: 1 to 4 PIL images to edit. Required. The instruction can refer to them as `<image_1>`, `<image_2>` and
    so on, in this order.
  - The instruction: `prompt`, free-form text that the LLM turns into a VGL, or `vgl`, a ready VGL. A `vgl` sent alone
    is used as it is, and it must carry an `edit_instruction`. A `vgl` sent with a `prompt` is the first image's VGL,
    kept as the baseline of the edit. It must then be a complete VGL, such as the `vgl` that a previous request
    returned.
  - `seed`: from 0 to 2,147,483,647. Without one, a random seed is used.
  - `aspect_ratio`, or `width` and `height`: the shape or the size of the result (see
    [Non-obvious decisions](#non-obvious-decisions)).
  - `mask`: the region of the image to edit, white on black. One image only, at that image's size.
  - `api_token` (optional): the key that the edit's attribution record is sent under. It takes precedence over the
    config's.
  - `source_request_id` (optional): your own id for the request, kept with the edit's usage record.

  Any other field is refused, FIBO 2's generation fields among them (`reference_image`, `style`, `resolution`,
  `negative_prompt`, `composite`).
- **Refused requests.** `Fibo2TurboInput` refuses a request that breaks a rule when you build it, before anything
  runs, with pydantic's `ValidationError` (a `ValueError`). A request without an image gets `IMAGES_REQUIRED`:
  "FIBO 2 Turbo edits images: send 1 to 4 'images'. To generate an image from a prompt, use FIBO 2 (the fibo-2
  package)." A request without a `prompt` and without a `vgl` gets `INSTRUCTION_REQUIRED`: "An edit needs its
  instruction: send a 'prompt', or a 'vgl' that carries an 'edit_instruction'." You can import both messages from
  `fibo_2_turbo`. The other rules (the mask, the size, the `<image_N>` references, a `vgl` sent with a `prompt`) are
  FIBO 2's, with FIBO 2's messages. A `vgl` sent alone without an `edit_instruction` is refused by `execute()`, with
  `ValueError`, before the model runs.
- **Output** (`Fibo2TurboOutput`, which is the `fibo-2` package's `Fibo2Output`): `image`, the edited PIL image;
  `vgl`, the VGL that the model read, which you can send back to reproduce or adjust the edit; `vgl_seconds` and
  `model_seconds`, the time spent on the VGL and in the model. Its other fields belong to FIBO 2's generation and
  composite features, and are always `None` here.

## License and weights

`setup()` first checks that your account holds a self-hosting license for FIBO 2 Turbo, `fibo-2-turbo`. Get one at
https://platform.bria.ai/outpost/licenses?model=fibo-2-turbo. FIBO 2 Turbo has its own license: a FIBO 2 license does
not open it, and you do not need a FIBO 2 license to run it, although the package installs and runs the `fibo-2`
package. Any API token of the account identifies it; set it as `api_token`, `FIBO_2_TURBO_API_TOKEN` or
`BRIA_API_TOKEN`. If the license is missing or has ended, `setup()` raises `common.outpost.LicenseError` with the
reason and a link to your licenses page, and nothing is loaded.

`setup()` then checks the LLM setting ([below](#the-llm-for-a-free-form-prompt)), so a setting that cannot work stops
the start before it downloads the FIBO 2 Turbo weights. The weights are then downloaded from Bria into
`~/.bria/models/fibo-2-turbo/<version>/` (override the root with `BRIA_HOME`), verified, and reused on later starts.
With `FIBO_2_TURBO_MODEL_PATH` set, `setup()` still checks the license, but does not download the FIBO 2 Turbo weights.

A connection to Bria is needed when a model starts: the Bria Engine and Bria's weights CDN (a `*.cloudfront.net`
host). On H100 and H200 GPUs the pipeline uses FlashAttention-3, and a start also needs `huggingface.co`, for its
kernel. Each edit sends an attribution record, which is how usage against your plan is counted. No Hugging Face
account is needed.

## Configuration

Settings are read from the environment with the `FIBO_2_TURBO_` prefix, or passed to `Fibo2TurboConfig`. The `fibo-2`
package's `FIBO_2_*` variables are not read, so a FIBO 2 install on the same server does not change FIBO 2 Turbo.

| Setting | Default | What it does |
|---|---|---|
| `FIBO_2_TURBO_API_TOKEN` | `BRIA_API_TOKEN` | The key the license check, the weights download and attribution run under. |
| `FIBO_2_TURBO_VGL_LLM` | not set | The LLM that writes the VGL of a free-form `prompt`: a litellm model name. Not set: Gemini 3.5 Flash-Lite. See [The LLM for a free-form prompt](#the-llm-for-a-free-form-prompt). |
| `FIBO_2_TURBO_MODEL_PATH` / `FIBO_2_TURBO_MODEL_REVISION` | Bria's mirror | A diffusers-format FIBO 2 Turbo checkpoint (local folder or Hugging Face repo id) to load instead. The revision applies only with a model path. |
| `FIBO_2_TURBO_TRANSFORMER_PATH` / `FIBO_2_TURBO_TRANSFORMER_REVISION` | the model path's | The FIBO 2 Turbo transformer, when it is kept apart from the rest of the checkpoint. A revision needs its path. |
| `FIBO_2_TURBO_COMPILE_MODEL` | `false` | Compiles the model's repeated blocks at setup: a slower start, faster edits. |
| `FIBO_2_TURBO_WARMUP` | `true` | Runs one small pass of the model at setup, so the first edit does not pay for loading the kernels. |
| `FIBO_2_TURBO_DEVICE` | `cuda` | The torch device that the model and the attribution model load on. |
| `FIBO_2_TURBO_ATTRIBUTION_AGENT` | not set | A name sent as the agent of each attribution record. |

### The LLM for a free-form prompt

A free-form `prompt` goes to an LLM with the images, and the LLM writes the VGL of the edit. A `vgl` that carries an
`edit_instruction` needs no LLM. You can use any LLM that [litellm](https://docs.litellm.ai/docs/providers) supports,
with your own credentials: set `FIBO_2_TURBO_VGL_LLM` to the litellm model name, and the credentials of its provider in
the environment. Examples:

| `FIBO_2_TURBO_VGL_LLM` | Credentials |
|---|---|
| not set | `GEMINI_API_KEY` or `GOOGLE_API_KEY` (Gemini 3.5 Flash-Lite on Google AI Studio). Without a key, Gemini 3.5 Flash-Lite runs on Vertex AI, with `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION` and `GOOGLE_APPLICATION_CREDENTIALS` |
| `vertex_ai/gemini-3.5-flash-lite` | `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION` and `GOOGLE_APPLICATION_CREDENTIALS` |
| `openai/gpt-5.4` | `OPENAI_API_KEY` |
| `anthropic/claude-sonnet-4-5` | `ANTHROPIC_API_KEY` |
| `bedrock/us.amazon.nova-pro-v1:0` | AWS credentials: environment variables, an AWS profile or an instance role. The AWS Region in `AWS_REGION_NAME` or `AWS_REGION` |
| `azure/<your deployment>` | `AZURE_API_KEY`, `AZURE_API_BASE` and `AZURE_API_VERSION` |

`setup()` checks the LLM before it downloads the FIBO 2 Turbo weights:

- If you set `FIBO_2_TURBO_VGL_LLM` and litellm knows no provider for it, or the credentials of its provider are not in
  the environment, `setup()` raises `ValueError`. The error gives the names of the missing variables, never their
  values.
- `setup()` checks only the API key of the provider, the three Vertex AI variables, or the AWS credentials for Bedrock.
  It does not check other variables, such as `AZURE_API_BASE` and `AZURE_API_VERSION`. If one of them is missing, the
  first request with a `prompt` fails.
- If you do not set `FIBO_2_TURBO_VGL_LLM` and the environment has no Gemini or Vertex AI credentials, `setup()`
  continues and logs a warning. Free-form prompts are then off: a request with a `prompt` raises `ValueError`. A
  request with a `vgl` alone works.

Limits:

- **Images.** Every request is an edit, so the LLM gets the images and must read images. If litellm does not list the
  LLM as a model that reads images, `setup()` logs a warning. If litellm knows that the LLM reads no images, a request
  with a `prompt` raises `ValueError`, and the pipeline does not call the LLM.
- **Quality.** Bria tuned the VGL prompts on Gemini 3.5 Flash-Lite. Other LLMs write different VGLs, so the edits can
  be different.
- **Models that do not work yet.** GPT-4o does not accept the response format that the VGL library sends, so each
  request with a `prompt` fails. Use GPT-4.1 or a newer OpenAI model. Gemini through a proxy, such as
  `openrouter/google/…`, can fail on edits, because the VGL library sends Gemini-specific JSON settings that the proxy
  ignores. For Gemini, leave `FIBO_2_TURBO_VGL_LLM` unset, or use `gemini/…` or `vertex_ai/…`.
- **Seeds.** The seed also goes to the LLM. The same seed gives the same VGL only on models that accept a seed. Some
  models do not accept a seed, for example Anthropic models and Amazon Nova. On those models, the same seed can give a
  different VGL and a different edit, and the VGL library logs a warning for each call.
- **Cost.** You pay your provider for the LLM calls. The LLM does not change the usage that Bria counts.

## Non-obvious decisions

- **Editing only.** A request needs 1 to 4 images. Without one, it is refused with `IMAGES_REQUIRED`, which points to
  FIBO 2 for generation.
- **4 steps, no guidance.** This is FIBO 2 Turbo's own schedule. Neither is configurable, and there is no negative
  prompt.
- **fibo-2 inside.** The model, the VGL step and the diffusion call are the `fibo-2` package's, in its turbo
  configuration. This package owns the license check (`fibo-2-turbo`), the settings (`FIBO_2_TURBO_*`; `fibo-2` reads
  only the values this package gives it, so no `FIBO_2_*` variable reaches `fibo-2`) and the attribution record, which
  counts each edit as FIBO 2 Turbo (`image_fibo_turbo:2`). A `fibo-2` error message that names `FIBO_2_VGL_LLM` reaches
  you naming `FIBO_2_TURBO_VGL_LLM`.
- **The size of the result.** Without a size, an edit comes back at about one megapixel, in the first image's aspect
  ratio. `width` and `height` (both, up to 4,096 pixels each) give exactly that size: a size near one megapixel is
  made slightly larger and cropped, and a size far from it is made at about one megapixel and resized. `aspect_ratio`
  (for example `"16:9"`) sets the shape of an edit of two or more images, at about one megapixel. A single-image edit
  keeps its image's shape: it ignores `aspect_ratio` and logs a warning. A size and a ratio together are refused.
- **Masks.** A mask (white: the region to edit) needs exactly one image, at that image's size, and takes no `width` or
  `height`. A masked edit comes back at the image's own size, with the image's own pixels outside the mask.
- **Image references.** An instruction refers to the images as `<image_1>`, `<image_2>` and so on, in the order sent.
  A reference to an image that was not sent is refused, instead of editing the wrong image.
- **Any image mode.** Each image is converted to RGB before the edit, so an RGBA PNG, a palette GIF or a CMYK JPEG is
  edited. An alpha channel is not used.
- Attribution is sent on a background thread per edit so it never blocks `execute()`. At exit the interpreter waits,
  for a bounded time, for the ones still in flight, so a script that ends right after its last edit still counts it.
