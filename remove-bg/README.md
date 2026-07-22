# remove_bg

`remove_bg` is a package for removing image backgrounds with BRIA RMBG, running **in-process** on
the GPU (no inference server). It accepts image inputs such as URLs, base64 strings, numpy arrays,
and PIL images, and returns a PIL image with an alpha channel. It can also return the generated
mask.

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
