# remove_bg

`remove_bg` is a package for removing image backgrounds with BRIA RMBG, running **in-process** on
the GPU (no inference server). It accepts image inputs such as URLs, base64 strings, numpy arrays,
and PIL images, and returns a PIL image with an alpha channel. It can also return the generated
mask.

## Quick start

**Prerequisites:** a Bria API token (`BRIA_API_TOKEN`), a Hugging Face token (`HF_TOKEN`) with
access to [`briaai/RMBG-2.0`](https://huggingface.co/briaai/RMBG-2.0), Python 3.10+, and an NVIDIA
GPU.

Exchange your API token for a short-lived CodeArtifact credential and install `remove_bg`:

```bash
export BRIA_API_TOKEN=...   # from https://platform.bria.ai
export HF_TOKEN=...          # a token whose account has access to briaai/RMBG-2.0

CA=$(curl -s "https://engine.prod.bria-api.com/v2/auth/access/code_artifact?repository=bria-remove-bg" \
      -H "api_token: $BRIA_API_TOKEN" -H "token: $BRIA_API_TOKEN" \
      | python -c "import sys, json; print(json.load(sys.stdin)['result']['authorization_token'])")

pip install --upgrade remove_bg \
  --extra-index-url "https://aws:$CA@bria-300465780738.d.codeartifact.us-east-1.amazonaws.com/pypi/bria-remove-bg/simple/"
```

Then remove a background:

```python
from remove_bg import RemoveBG, RemoveBGConfig, RemoveBGInput

pipe = RemoveBG(config=RemoveBGConfig(compile_model=False))
pipe.setup()  # loads RMBG-2.0 (downloads the weights from Hugging Face on first run)

result = pipe.execute(
    [RemoveBGInput(image="https://bria-test-images.s3.us-east-1.amazonaws.com/strawberry.jpg")]
)[0]
result.image.save("cutout.png")  # RGBA cutout with the background removed

pipe.cleanup()
```

The [`notebooks/code_example.ipynb`](../notebooks/code_example.ipynb) bundle walks through the same
flow end to end (auth, install, run, and a transparency preview).

## Pipeline

`RemoveBG` follows the `common.pipeline` lifecycle: `setup()` loads the RMBG model (optionally
`torch.compile`-d), `execute()` runs a batch of `RemoveBGInput`s, and `cleanup()` frees the model
and empties the CUDA cache.

- **Input** (`RemoveBGInput`): an image (URL, base64, numpy array, or PIL image, via
  `bria_external.image`), plus `return_mask` and alpha-preservation flags. The image is resolved
  eagerly in `model_post_init` rather than lazily.
- **Output** (`RemoveBGOutput`): the RGBA cutout, and the raw `mask` when `return_mask=True`.

## Non-obvious decisions

- **`legacy_mode`** reproduces the older "spring" output by skipping the newer inference
  optimizations that slightly change pixels — kept for numerical back-compat, off by default.
- **`batch_size`** and **`compile_model`** trade warm-up cost for steady-state throughput; compile
  is on by default and paid once in `setup()`.
- Attribution is fired off on a background daemon thread per output so it never blocks `execute()`.
