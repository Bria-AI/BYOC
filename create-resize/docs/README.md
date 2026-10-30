# resize

Recompose an image to a fixed ad canvas so the subject lands clear of the copy.

You have one photo and need it as a 9:16 story, a 1:1 square and a 728×90 banner — and each of
those has a headline, a CTA and a logo sitting on top of it. A normal crop knows the shape it has
to hit but nothing about the text, so it centers the subject and the headline lands on the
subject's face. `resize` takes one extra input — rectangles saying where the copy will sit — and
picks a crop that puts the subject in the space the copy leaves empty.

`resize` is a multi-capability package. Today it provides **smart crop**; each future capability
gets its own sub-package.

## Requirements

- Python 3.10+
- CPU only. The search is numpy over ~25 points; there is nothing here a GPU accelerates.
- No model weights to download.
- Input limits, enforced with an error rather than a silent downscale: canvas at most 16384 px per
  side, source image at most 80 MP, and a URL image must download in under 40 MB. The MP and
  download caps are configurable (`RESIZE_MAX_SOURCE_MEGAPIXELS`, `RESIZE_MAX_DOWNLOAD_MB`).
- A saliency credential **only** if you want the package to predict saliency for you. Saliency is
  predicted by a vision model called through OpenRouter, so this is an **OpenRouter API key, not
  your `BRIA_API_TOKEN`** — set `RESIZE_SALIENCY_API_KEY`, or pass `api_key` on the request. Your
  image is sent to that third-party endpoint on this path (`openai/gpt-5.6-luna` by default;
  override with `RESIZE_SALIENCY_MODEL`). Supply `points` yourself and no image is sent to the
  model; a URL source is still fetched over https.
- **Install access**: distributed through Bria's AWS CodeArtifact repository `bria-create-resize`,
  so installation needs a CodeArtifact credential (see [Installation](#installation)).

## Installation

`create_resize` (imported as `resize`) publishes to Bria's AWS CodeArtifact PyPI repository
`bria-create-resize`, not to public PyPI. Installing is two steps.

1. Request a short-lived CodeArtifact credential from the Bria Engine with your `BRIA_API_TOKEN`:

   ```bash
   curl -s "https://engine.prod.bria-api.com/v2/auth/access/code_artifact?repository=bria-create-resize" \
     -H "api_token: $BRIA_API_TOKEN" \
     -H "token: $BRIA_API_TOKEN"
   ```

   The response contains `result.authorization_token`. Use it as the CodeArtifact password with
   the username `aws`. The token is domain-scoped, so the same credential authenticates every
   Bria CodeArtifact repository below.

2. Install `create_resize`, pointing pip at the repository that hosts it and the ones hosting its
   Bria dependencies. Replace `<TOKEN>` with the authorization token from step 1:

   ```bash
   BASE="https://aws:<TOKEN>@bria-300465780738.d.codeartifact.us-east-1.amazonaws.com/pypi"

   pip install "create_resize>=0.1.0" \
     --extra-index-url "$BASE/bria-create-resize/simple/" \
     --extra-index-url "$BASE/everywhere-common/simple/" \
     --extra-index-url "$BASE/bria-external/simple/"
   ```

`../notebooks/code_example.ipynb` automates both steps in Python. There are no model weights to fetch
and no GPU build step. The dependency set is numpy, OpenCV, SciPy, requests, pydantic and
pydantic-settings, plus Bria's `byoc-common` and `bria-external-logs` — which is why the two extra
index URLs above are needed.

> Do not `pip install resize` from public PyPI: that name belongs to an unrelated project. The
> import name here is `resize`, but the distribution is `create_resize`.

## Usage

```python
from resize import KeepoutBox, ResizeConfig, SaliencyPoint, Resize, SmartCropInput

pipeline = Resize(ResizeConfig())
pipeline.setup()

result = pipeline.smart_crop(
    SmartCropInput(
        image="https://example.com/photo.jpg",  # URL or base64
        canvas=(1080, 1920),  # the ad size you need, in pixels
        keepout=[
            KeepoutBox(box=(0.08, 0.06, 0.84, 0.16), color="#ffffff"),  # headline
            KeepoutBox(box=(0.24, 0.85, 0.52, 0.07), color="#ffffff"),  # CTA
        ],
        points=[SaliencyPoint(x=0.42, y=0.22, weight=1.0, label="face")],
    )
)

print(result.crop)  # where to cut, in ORIGINAL source pixels
print(result.metrics)  # how much of the subject survived, and how much is under the copy

pipeline.cleanup()
```

To embed the decision in another pipeline without the lifecycle, call the function directly on a
BGR ndarray. With `points` supplied, as below, it performs no I/O at all — omit them and it makes
the saliency model call, taking the credential from `api_key=` or the `OPENROUTER_API_KEY`
environment variable. `RESIZE_SALIENCY_API_KEY` is read by `ResizeConfig`, so it applies only on
the pipeline path above:

```python
from resize.smart_crop import decide_crop

# plain lists/dicts here, not the pydantic models above
decision = decide_crop(
    bgr,  # a BGR ndarray
    (1080, 1920),
    [{"box": [0.08, 0.06, 0.84, 0.16], "color": "#ffffff"}],
    points=[{"x": 0.42, "y": 0.22, "weight": 1.0}],
)
```

## The geometry

The crop rectangle is **allowed to leave the image**. Its top-left may be negative or past the
edge. One rule splits it:

```
rect ∩ source  →  real pixels (a crop)
rect \ source  →  pixels to generate (outpainting)
```

A rect inside the image is a pure crop; a rect containing the image is a pure outpaint. That is
why `crop.y` often comes back negative: the window starts above the photo, and that band is
exactly what needs generating. `metrics.outpaint_fraction` quantifies it.

Because generated background under text is free while dropping the subject is expensive, the
optimizer will happily invent 40% of a canvas rather than crop a face away. That is the objective
working, not a failure.

## Inputs

| field | type | default | notes |
|---|---|---|---|
| `image` | str | — | http(s) URL or base64 |
| `canvas` | `[width, height]` | — | target canvas in pixels |
| `keepout` | list | `[]` | where the copy sits. **Normalized 0–1 to the canvas**, top-left origin |
| `points` | list \| None | predicted | saliency to use instead of predicting. **Normalized 0–1 to the source image**, top-left origin — a different basis from `keepout`, which is normalized to the canvas |
| `algorithm` | `"grid"` \| `"field"` \| null | `null` | which search runs. Omit to use the deployment default (`RESIZE_ALGORITHM`, ships as `grid`) &mdash; see below |
| `outpaint_bias` | float | `0.25` | grid only. 0 = tightest crop that keeps the subject, 1 = widest |
| `avoid_strength` | float | `1.5` | grid only. How hard to push the subject out from under the copy |
| `contrast_strength` | float | `1.0` | grid only. Weight on background/ink contrast |
| `render` | bool | `false` | also return a composed canvas |
| `fill` | `[r, g, b]` | `[0, 0, 0]` | RGB for the un-generated region when `render: true` |
| `max_image_dim` | int | `1600` | cap on the returned image's long edge. Does not affect the crop rect |
| `field_params` | dict | `{}` | field only. Objective overrides, e.g. `{"w_spread": 6.0}` |
| `api_key` | str \| None | — | per-request saliency credential, overriding the configured one |

Each keep-out box is `{"box": [x, y, w, h], "color": "#rrggbb", "role": "headline"}`. `color` and
`role` are both optional: `role` is a free-text label for your own bookkeeping — it is accepted and ignored, not
returned — and `color` is the
element's ink, which drives a contrast term so the background chosen under white text is flat and
dark enough to read against. Without a color the decision is purely geometric.

> **Boxes are normalized, not pixels.** Passing canvas pixels is the one integration mistake worth
> guarding against: the box simply never matches a saliency point, the keep-out term contributes
> nothing, and you get a layout-blind crop with healthy-looking metrics. `KeepoutBox` rejects
> out-of-range values — each component outside 0–1, and boxes whose right or bottom edge runs past
> the canvas — so this fails at the boundary instead. `decide_crop` takes plain lists and does not
> validate them, so on that path the check is yours.

## Outputs

These are the fields of `SmartCropOutput`, returned by the pipeline. `decide_crop` returns a plain
dict with a different shape — `saliency["predicted"]` rather than `saliency_predicted`, and a BGR
ndarray under `_image` rather than a base64 `image_base64`.

| field | meaning |
|---|---|
| `crop` | where to cut, in **original source pixels**. May be negative or extend past the source |
| `crop_normalized` | the same rect divided by the source dimensions |
| `metrics.retained` | fraction of saliency weight surviving the crop |
| `metrics.occluded_by_layout` | retained weight landing under a keep-out box, each box widened by a 1% margin first |
| `metrics.clear` | `retained − occluded_by_layout`: in frame **and** visible |
| `metrics.outpaint_fraction` | fraction of the canvas not covered by real pixels |
| `metrics.min_element_contrast` | worst element's contrast against its background, 0–1. `null` when no box carried a `color`, since the contrast term never ran |
| `source` | source image size in pixels, as `{width, height}` |
| `canvas` | the canvas size you asked for, as `{width, height}` |
| `algorithm` | which arm ran: `"grid"` or `"field"` |
| `points` | the saliency used — cache and pass back for other canvas sizes |
| `saliency_predicted` | `true` if a model call was made, `false` if you supplied `points` |
| `image_base64` | `data:image/jpeg;base64,...` — present only when `render: true` |
| `image_size` | size of `image_base64` after `max_image_dim` |

`retained`, `occluded_by_layout`, `clear` and `outpaint_fraction` are recomputed independently of
the objective that chose the window, so they are not the search grading its own homework.
`min_element_contrast` is the exception: on the `grid` arm it is the winning window's own contrast
term, which was weighted into the cost it minimized.

**The default output is coordinates, not an image**, and deliberately: the rect applies to your
full-resolution original, so you crop at full quality in your own pipeline and choose your own
model for the region that needs generating. Set `render: true` if you want a composed canvas —
but note it *composes*, placing real pixels and leaving the rest flat. It does not generate.

**Applying the rect yourself takes care.** `crop.x` and `crop.y` may be negative, and a plain
`bgr[y : y + h, x : x + w]` slice does not fail on that — numpy reads a negative start as an
offset from the *end* of the array, so you silently get a different region at a different size.
For a rect of `x=205, y=-326, w=810, h=1439` against a 1152×896 source, that slice returns 326
rows of the bottom of the photo instead of the 1439 rows starting above it.

`apply_crop`, exported from `resize.smart_crop`, does the clamped placement for you:

```python
from resize.smart_crop import apply_crop

canvas = apply_crop(bgr, result.crop.model_dump(), (1080, 1920), fill=(0, 0, 0))
```

`fill` here is **BGR**, unlike the request's `fill` in the Inputs table above, which is RGB — it
goes straight into the output array — and covers the region the rect leaves outside the source,
which is what you would hand to an expansion model.

### Cache the points

Saliency depends only on the image — not the canvas, not the layout. It is the slowest stage by a
wide margin. Predict once, then pass `points` back for every other ad size of that photo.

## Configuration

`Resize` is configured with a `ResizeConfig`. Every field can also be set from an environment
variable using the `RESIZE_` prefix shown below (`saliency_timeout_s` becomes
`RESIZE_SALIENCY_TIMEOUT_S`).

**`ResizeConfig`** (env prefix `RESIZE_`):

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `algorithm` | `"grid"` \| `"field"` | `"grid"` | Which search to run when a request does not name one. `grid` is the default because it is the arm with a human evaluation behind it; see [The two algorithms](#the-two-algorithms). A request's `algorithm` overrides this. |
| `saliency_model` | `str` | `"openai/gpt-5.6-luna"` | OpenRouter model id used to predict saliency. Ignored when the request supplies `points`. |
| `saliency_api_key` | `str \| None` | `None` | OpenRouter credential. Required only when a request omits `points`; a request's `api_key` overrides it. |
| `saliency_endpoint` | `str` | `"https://openrouter.ai/api/v1/chat/completions"` | Chat-completions endpoint for the saliency call. Point this at a compatible gateway or a proxy in your own network. |
| `saliency_timeout_s` | `float` | `90.0` | Per-request timeout on the saliency call, in seconds. |
| `search_canvas` | `int` | `1400` | Long edge, in pixels, of the reduced canvas the search runs on. Lower is faster and coarser; the objective is scale-invariant, so the chosen rect maps back to full resolution either way. |
| `work_max_dim` | `int` | `2048` | Source images larger than this on their long edge are downscaled before the search. Does not affect the returned rect, which is always in original source pixels. |
| `max_source_megapixels` | `float` | `80.0` | Requests whose decoded source exceeds this are rejected. A guard against decoding a decompression bomb. |
| `max_download_mb` | `float` | `40.0` | Cap on a URL source's download size, enforced while streaming as well as against `Content-Length`. |
| `device` | `"cpu"` \| `"cuda"` | `"cpu"` | Present for interface parity with other pipelines. The search is numpy on the CPU and there is no model to place, so this is not read today. |

## The two algorithms

Both pick a window and are scored by the same code, so neither grades its own homework.

**`grid`** (default) searches 18 scales × 11×11 positions and scores placement, point-in-box
overlap, contrast, size and upscale.

**`field`** (opt-in) maximizes the soft-min signed clearance of the weighted points from the
keep-out borders, so points settle deep in the open area rather than just outside the boxes.

Measured on 888 pairs — 12 photos × 74 layouts — with identical cached saliency.

| metric | `grid` | `field` |
|---|---|---|
| clear saliency | 0.931 | **0.981** |
| retained | 0.940 | **0.999** |
| occluded by the copy | **0.008** | 0.019 |
| min element contrast | **0.377** | 0.323 |
| real pixels (% of canvas) | **62.2%** | 46.3% |
| search time | 174 ms | **87 ms** |

`field` keeps the subject clear of the copy better and runs about twice as fast. It pays in
generated area and contrast, because filling a canvas from a smaller real region means expanding.

`grid` remains the default: those numbers are all geometric, and none of them says whether 46%
real pixels *looks* right once the fill is generated. The human A/B has not been run on `field`.

## Limitations

- **Occlusion is scored per point, not per area.** Points have no size, so a hat whose center
  clears a box — by more than the 1% margin — scores as clear while visually sitting under it.
- **Neither arm is bit-reproducible across platforms once a box carries a `color`.** `grid`'s
  discrete argmin flips a near-tie to an adjacent step under small float differences in the
  contrast pipeline. The `field` *objective* reads no pixels, but the arm then re-ranks its
  near-optima on pixel contrast (`w_contrast`, default 3.0), so it inherits the same sensitivity.
  With no colored box, `field` builds no contrast field at all and is identical everywhere.
- **`field`'s objective does not model contrast.** The search itself is purely geometric; contrast
  enters afterwards, as a re-rank over the solver's near-optima. Contrast is then scored for both
  arms with the same code, so a regression shows rather than hides.
- **Face detection is dated.** Haar cascades, which is why `opencv-python-headless` is pinned
  below 5 — OpenCV 5 ships no cascade data and faces would silently stop being must-keeps.
- **`render` does not generate.** It composes real pixels and leaves the rest flat. Pair the
  returned rect with an expansion model for a finished background.
- **Weights are hand-tuned.** The objective's λ values were set on a few images.
