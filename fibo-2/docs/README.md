# fibo-2

fibo-2 is Bria's image model for both text-to-image generation and instruction editing. Give it a
prompt and it generates an image; give it a prompt and up to four images and it edits them, for
example placing an object from one image into another. A mask restricts an edit to one region of a
single image. A reference image generates a new image (inspire mode): on its own, a close variation of the image;
with a prompt, the prompt's scene, using the image as a visual reference. Before the model sees a free-form prompt,
Bria's VGL library and an LLM expand it into a structured caption, a VGL. The default LLM is Gemini, and you can use
a different LLM. The VGL is returned with every image as `vgl`, and can be sent back to reproduce or adjust the image.

## Requirements

- An NVIDIA GPU with at least 40 GB of memory. On H100 and H200 GPUs the pipeline uses
  FlashAttention-3.
- Python 3.11 or 3.12.
- A Bria API token (`BRIA_API_TOKEN`, from https://platform.bria.ai). Any token of your account works
  if the account has a self-hosting license for fibo-2.
- Approximately 26 GB of disk space for the weights.
- For free-form prompts, an LLM: set `FIBO_2_VGL_LLM` and the credentials of its provider. See
  [The LLM for free-form prompts](#the-llm-for-free-form-prompts). A request that sends a `vgl` alone, without a
  prompt, needs no LLM.

## Usage

```python
import os

from fibo_2 import Fibo2Config, Fibo2Input, Fibo2Pipeline

pipeline = Fibo2Pipeline(config=Fibo2Config(api_token=os.environ["BRIA_API_TOKEN"]))
pipeline.setup()  # checks your license, then downloads the weights from Bria into ~/.bria/models on the first run

# Generate
result = pipeline.execute(Fibo2Input(prompt="A red bicycle leaning against a white brick wall", seed=42))
result.image.save("bicycle.png")

# Edit
edited = pipeline.execute(Fibo2Input(images=[result.image], prompt="Make the bicycle blue", seed=42))

# Inspire: a new scene, with a PIL image as visual reference
inspired = pipeline.execute(
    Fibo2Input(reference_image=result.image, prompt="A red scooter on a cobbled street", seed=42)
)
```

## License and weights

`setup()` first checks that your account has a self-hosting license for fibo-2. Any API token of your
account identifies the account. Set the token as `api_token`, `FIBO_2_API_TOKEN` or `BRIA_API_TOKEN`.

If your account has no license for fibo-2, or the license has ended, `setup()` raises
`common.outpost.LicenseError`. The error gives the reason and a link to your licenses page. The
pipeline then loads nothing.

Then `setup()` downloads the weights from Bria into `~/.bria/models/fibo-2/<version>/`. To use a
different root folder, set `BRIA_HOME`. The pipeline checks each file. On the next start, it uses the
same files again and does not download them again.

Each start needs a connection to Bria: to the Bria Engine (`engine.prod.bria-api.com`) and to the
Bria weights CDN (a `*.cloudfront.net` host). On H100 and H200 GPUs, a start also needs `huggingface.co`, for the
FlashAttention-3 kernel. Each generation sends an attribution record. Bria uses these records to
count your usage against your plan. You do not need a Hugging Face account.

To use a checkpoint of your own, set `model_path` (`FIBO_2_MODEL_PATH`). `setup()` still checks the
license, but it does not download weights. The turbo model (`FIBO_2_TURBO`) needs this setting,
because Bria supplies only the full model.

## Configuration

Settings are read from the environment with the `FIBO_2_` prefix, or passed to `Fibo2Config`:

| Setting | Default | Meaning |
|---|---|---|
| `FIBO_2_API_TOKEN` | `BRIA_API_TOKEN` | Your Bria API token. `setup()` uses it for the license check and to download the weights |
| `FIBO_2_MODEL_PATH` | not set | A checkpoint of your own in diffusers format: a local folder or a Hugging Face repo id. If you do not set it, `setup()` downloads the weights from Bria |
| `FIBO_2_TURBO` | `false` | Whether the checkpoint is the distilled turbo model. The full model runs 30 steps with guidance; turbo runs 4 steps without. Turbo needs `FIBO_2_MODEL_PATH` |
| `FIBO_2_TEXT_IMAGES` | `true` | Show the model an image of each text the caption asks for, as fibo-2 was trained. Captions without text are unchanged |
| `FIBO_2_WARMUP` | `true` | Run one small generation during setup |
| `FIBO_2_VGL_LLM` | not set | The LLM that writes the VGL of a free-form prompt: a litellm model name. If you do not set it, Gemini 3.5 Flash-Lite writes the VGL. See [The LLM for free-form prompts](#the-llm-for-free-form-prompts) |

## The LLM for free-form prompts

The VGL library sends a free-form prompt to an LLM, and the LLM writes the VGL. You can use each LLM that
[litellm](https://docs.litellm.ai/docs/providers) supports. Set `FIBO_2_VGL_LLM` to the litellm model name. Set the
credentials of the provider in the environment. Examples:

| `FIBO_2_VGL_LLM` | Credentials |
|------------------|-------------|
| not set | `GEMINI_API_KEY` or `GOOGLE_API_KEY` (Gemini 3.5 Flash-Lite on Google AI Studio). If you do not set a key, Gemini 3.5 Flash-Lite runs on Vertex AI, with `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION` and `GOOGLE_APPLICATION_CREDENTIALS` |
| `vertex_ai/gemini-3.5-flash-lite` | `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION` and `GOOGLE_APPLICATION_CREDENTIALS` |
| `openai/gpt-5.4` | `OPENAI_API_KEY` |
| `anthropic/claude-sonnet-4-5` | `ANTHROPIC_API_KEY` |
| `bedrock/us.amazon.nova-pro-v1:0` | AWS credentials: environment variables, an AWS profile or an instance role. The AWS Region in `AWS_REGION_NAME` or `AWS_REGION` |
| `azure/<your deployment>` | `AZURE_API_KEY`, `AZURE_API_BASE` and `AZURE_API_VERSION` |

`setup()` checks the LLM before it downloads the weights:

- If you set `FIBO_2_VGL_LLM` and the credentials of its provider are not in the environment, `setup()` raises
  `ValueError`. The error gives the names of the missing variables. If litellm does not know the provider,
  `setup()` also raises `ValueError`.
- `setup()` checks only the API key of the provider, the three Vertex AI variables, or the AWS credentials for
  Bedrock. It does not check other variables, such as `AZURE_API_BASE` and `AZURE_API_VERSION`. If one of them is
  missing, the first request with a prompt fails.
- If you do not set `FIBO_2_VGL_LLM` and the environment has no Gemini or Vertex AI credentials, `setup()`
  continues and logs a warning. Free-form prompts are then off: a request with a prompt raises `ValueError`. A
  request that sends a `vgl` alone, without a prompt, works.

Limits:

- **Images.** An edit and inspire send the images to the LLM, so they need an LLM that reads images. If litellm
  does not list the LLM as a model that reads images, `setup()` logs a warning. If litellm knows that the LLM reads
  no images, a request with images raises `ValueError`, and the pipeline does not call the LLM. An LLM that reads
  only text can write the VGL of a prompt without images.
- **Quality.** Bria tuned the VGL prompts on Gemini 3.5 Flash-Lite. Other LLMs write different VGLs, so the quality
  of the images can be different.
- **Models that do not work yet.** GPT-4o does not accept the response format that the VGL library sends, so each
  request with a prompt fails. Use GPT-4.1 or a newer OpenAI model. Gemini through a proxy, such as
  `openrouter/google/…`, can fail on edits, because the VGL library sends Gemini-specific JSON settings that the proxy
  ignores. For Gemini, leave `FIBO_2_VGL_LLM` unset, or use `gemini/…` or `vertex_ai/…`.
- **Seeds.** The same seed gives the same VGL only on models that accept a seed. Some models do not accept a seed,
  for example Anthropic models and Amazon Nova. On those models, the same seed can give a different VGL and a
  different image, and the VGL library logs a warning for each call.
- **Cost.** You pay your provider for the LLM calls. The LLM does not change the usage that Bria counts.

## Non-obvious decisions

- **Any output size.** fibo-2 generates images whose sides are multiples of 16 pixels, at about one
  megapixel in any aspect ratio. Any requested width and height is accepted: a size near a megapixel
  is generated slightly larger and cropped to the exact size, and a size far from one is generated
  at a megapixel and resized. An aspect ratio gets the model's own size for it, for example
  1424x800 for 16:9. Twenty-one ratios are on offer: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9,
  16:10, 10:16, 2:1, 1:2, 21:9, 9:21, 3:1, 1:3, 4:1, 1:4, 5:1 and 1:5, and each comes back in exactly
  that ratio (16:9 as 1408x792, trimmed from the 1424x800 it is generated at).
- **Edits.** Images are used in the order given; prompts can refer to them as `<image_1>`,
  `<image_2>` and so on, and a prompt that names an image which wasn't sent is refused.
  Without a size, an edit takes the shape of the first image. A masked edit returns at the original
  image's size, with its pixels unchanged outside the mask, so it takes no `width` or `height`.
- **Inspire.** `reference_image` alone generates a close variation of the image: the VGL library describes the
  image. With `prompt` it generates the prompt's scene, using the image only as a visual reference. fibo-2 itself
  never sees the image. Inspire takes no `images`, `mask` or `vgl`.
- **Changing a VGL.** `vgl` with `prompt` changes the VGL instead of using it as-is. Without `images`, the VGL
  library refines it by the prompt. With `images`, send the VGL of the first image, for example the `vgl` returned
  with it: the edit starts from it. This needs an LLM, and the `vgl` must be a full VGL, such as one that a previous
  request returned.
- **Styles.** `style` isn't available for BYOC yet; leave it unset.
- **Seeds** run from 0 to 2,147,483,647 (a 32-bit integer).
- **Steps and guidance** are fixed by the checkpoint and cannot be set per request.
- **Negative prompts** work only on the full model. Turbo runs without guidance, so it ignores `negative_prompt` and the default negative caption.
