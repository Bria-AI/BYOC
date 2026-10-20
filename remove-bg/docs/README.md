# remove_bg

`remove_bg` is a package for removing image backgrounds with BRIA RMBG, running **in-process** on
the GPU (no inference server). It accepts image inputs such as URLs, base64 strings, numpy arrays,
and PIL images, and returns a PIL image with an alpha channel. It can also return the generated
mask.

## Quick start

**Prerequisites:** a Bria API token (`BRIA_API_TOKEN`, from https://platform.bria.ai). Any token works if your account
holds a self-hosting license for RMBG 2.0. You also need Python 3.10+ and an NVIDIA GPU.

Exchange your API token for a short-lived CodeArtifact credential and install `remove_bg`:

```bash
export BRIA_API_TOKEN=...   # from https://platform.bria.ai

CA=$(curl -s "https://engine.prod.bria-api.com/v2/outpost/access/code_artifact?repository=bria-remove-bg" \
      -H "api_token: $BRIA_API_TOKEN" -H "token: $BRIA_API_TOKEN" \
      | python -c "import sys, json; print(json.load(sys.stdin)['result']['authorization_token'])")

pip install --upgrade "remove_bg>=0.5.0" \
  --extra-index-url "https://aws:$CA@bria-300465780738.d.codeartifact.us-east-1.amazonaws.com/pypi/bria-remove-bg/simple/"
```

Then remove a background:

```python
import os

from remove_bg import RemoveBG, RemoveBGConfig, RemoveBGInput

pipe = RemoveBG(config=RemoveBGConfig(compile_model=False, api_token=os.environ["BRIA_API_TOKEN"]))
pipe.setup()  # checks your license, then downloads the weights from Bria into ~/.bria/models on first run

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

## License and weights

`setup()` first checks that your account holds a self-hosting license for RMBG 2.0 (any API token of the account
identifies it; set it as `api_token`, `RMBG_API_TOKEN` or `BRIA_API_TOKEN`). If the license is missing or has ended,
`setup()` raises `common.outpost.LicenseError` with the reason and a link to your licenses page, and nothing is
loaded. The weights are then downloaded from Bria into `~/.bria/models/rmbg-2.0/<version>/` (override the root with
`BRIA_HOME`), verified, and reused on later starts. A connection to Bria is needed when a model starts: the Bria
Engine and Bria's weights CDN (a `*.cloudfront.net` host). Each generation sends an attribution record, which is how
usage against your plan is counted. No Hugging Face account is needed.

**Upgrading from 0.4 or earlier:** the pipeline now needs an API token (`api_token`, `RMBG_API_TOKEN` or
`BRIA_API_TOKEN`): `setup()` uses it for the license check and the weights download, so code that built
`RemoveBGConfig()` with none of them set has to set one now. `HF_TOKEN` is no longer used.

## Troubleshooting

- **`CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` from `setup()` on an AWS Deep Learning AMI:** the AMI's
  `LD_LIBRARY_PATH` puts its own CUDA toolkit ahead of the CUDA libraries installed with `torch`. Run
  `unset LD_LIBRARY_PATH` in the shell before you start Python or Jupyter.

## Non-obvious decisions

- **`legacy_mode`** reproduces the older "spring" output by skipping the newer inference
  optimizations that slightly change pixels — kept for numerical back-compat, off by default.
- **`batch_size`** and **`compile_model`** trade warm-up cost for steady-state throughput; compile
  is on by default and paid once in `setup()`.
- Attribution is sent on a background thread per output so it never blocks `execute()`. At exit the interpreter
  waits, for a bounded time, for the ones still in flight, so a script that ends right after its last cutout still counts it.
