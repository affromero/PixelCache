# Changelog

## \[0.1.0\] - 2026-05-14

This is a breaking refactor relative to `0.0.x`. `pixelcache` was a pre-1.0
library; the `0.0.x` API was not stabilized. The version bump signals the
breaks explicitly.

### Hot-path perf (correctness-preserving)

- `HashableImage.__init__` no longer writes a temp PNG to disk for in-memory
  inputs (numpy, tensor, PIL, bytes). The source path stays `None` until the
  first `get_filename()` call, which materializes a temp on demand. Eliminates
  per-construction disk I/O in tight loops (video frames, mask batches).
- `read_image` is now single-decode. The previous double-open EXIF-check
  was replaced with `getexif()` on the same open handle; if no rotation is
  needed, the file is decoded via torchvision's fast `decode_jpeg` /
  `decode_png` path. EXIF-rotated images go through `ImageOps.exif_transpose`
  on the open handle.
- `HashableImage.__hash__` caches an `xxhash.xxh3_64` content fingerprint
  on the instance. Cached lookups are ~400 ns (was ~17 µs with `@jaxtyped`
  - full-buffer materialization). `xxhash` is a hard runtime dep.
- `HashableImage.__eq__` short-circuits on identity → hash → content.
- `HashableImage.__repr__` no longer calls `mean / std / min / max / get_filename` (those made `repr()` O(image) and disk-touching). For the
  expensive form, call `.summary()` explicitly.
- `HashableImage.numpy()` returns a fresh copy by default; `numpy_view()`
  is the explicit zero-copy escape hatch (writeable=False).
- `HashableImage.tensor()` and `.pil()` likewise return safe copies;
  `tensor_view()` / `pil_view()` are the opt-in zero-copy variants.
- The 17 `@lru_cache(maxsize=MAX_IMG_CACHE)` instance-method decorators
  are gone. `MAX_IMG_CACHE` was 5; with a full-bytes hash as the cache
  key, hit rate was near zero. Callers wanting caching wrap their own
  methods with an `N` tuned to their workload.

### Breaking API removals

- `pixelcache/main.py` is **deleted**. The canonical import path was always
  `from pixelcache import ...`; the `pixelcache.main` submodule had zero
  Hax-CV callers and only two internal consumers (README + an example),
  both updated.
- `MAX_IMG_CACHE` is no longer exported.
- 22 `HashableImage` methods with zero callers in Hax-CV are removed:
  `is_empty`, `flip_lr`, `to_space_color`, `compress_image` (method form),
  `merge_rgb`, `group_regions_binary`, `draw_points`, `draw_bbox`,
  `get_canny_edge`, `differential_mask`, `draw_lines`, `draw_text`,
  `draw_polygon`, `mask2polygon`, `mask2points`, `maskidx2bbox`,
  `split_masks`, `convert_binary_to_value`, `extract_binary_from_value`,
  `logical_and`, `logical_or`, `logical_and_reduce`.
- The `tools/image.py` orphan utilities `compress_image` and
  `convert_to_space_color` are removed (their only callers were the
  deleted methods above).
- `HashableDict` and `HashableList` are now **immutable**. They inherit
  from `collections.abc.Mapping` / `Sequence` (read-only protocols) — no
  `__setitem__`, `__delitem__`, `insert`, or `append`. To "modify", construct
  a new instance from the merged data. Constructor deep-copies mutable
  leaf values (ndarrays, tensors, PIL images) so external mutation of the
  source cannot invalidate the cached structural hash.
- `HashableList.__hash__` switched from `frozenset(items)` to `tuple(items)`.
  Order and multiplicity now matter — `[1, 2]`, `[2, 1]`, `[1, 1, 2]` are
  distinct.

### Bug fixes

- `ImageCrop.__call__`: the normalized branch was passing the `HashableImage`
  object (not the PIL image) to `torchvision.transforms.functional.crop`,
  and both branches were passing `bottom` / `right` instead of `height` /
  `width` for the last two args. Result: wrong dimensions on every crop.
  Fixed.
- `Points.xy` / `Points.xyn`: both scaled `(x, y)` point pairs by
  `(height, width)` instead of `(width, height)`. For non-square images
  this swapped both axes. Fixed.
- `HashableImage.is_rgb()`: crashed on PIL inputs with
  `AttributeError: no .shape`. Now branches explicitly on
  `Image.mode == "RGB"`. The same fix unblocks `hash()` and `__eq__` on
  PIL-mode images.
- `HashableImage.__setitem__` (masked assign) was mutating the source
  tensor in place for torch-mode images. `tensor()` now returns a clone,
  so masked assignment cannot leak back.
- `HashableDict.__eq__` with ndarray values no longer raises
  "truth value of an array is ambiguous". Tensor / ndarray / PIL values
  are compared by content; PIL specifically by `mode + size + bytes`
  (PIL's own `__eq__` is object-identity).
- `tools/image.py:compress_image` switched from `NamedTemporaryFile.name`
  (file got GC'd, leaving a dangling path) to `tempfile.mkstemp` + close.
- `tools/mask.py:morphologyEx` had a `"elipse"` typo and a dead
  `np.asarray(...)` whose result was never assigned. Fixed.
- `HashableImage.shape` raised on grayscale (`L`-mode) images. Fixed —
  `L` returns `(h, w)` like binary.

### Structural

- `pixelcache/main.py` (4935 lines) decomposed into `pixelcache/core.py`
  (HashableImage), `pixelcache/_collections.py` (HashableDict +
  HashableList), `pixelcache/_types.py` (BoundingBox + ImageCrop + Points),
  and `pixelcache/data/palette.py` (the 768-line color table).
- Hax-CV's forbidden-pattern set applies inside pixelcache during this
  refactor (see `pixelcache/CLAUDE.md`). Suppression count dropped: 11
  fewer `# type: ignore` markers.
- `tools/utils.py` shrank from 875 → 69 lines after the palette extract.
  `tools/image.py` from 1428 → 945 after deleting orphaned utilities.
  `tools/mask.py` from 1297 → 962.

### Dependencies

- Added: `xxhash >= 3.5.0` (hard runtime dep for `__hash__`).
- Removed: `requests` (`urllib.request.urlopen` is enough).
- Added dev: `pytest >= 8.0`, `pytest-benchmark >= 4.0`.

### Known limitations (intentional, deferred)

- `HashableDict.__hash__` and `HashableList.__hash__` combine
  per-element hashes via Python's `hash((...))` / `hash(frozenset(...))`,
  which uses the process-randomized hash seed (`PYTHONHASHSEED`).
  Hashes are stable **within a process** but not across processes.
  This matches `HashableImage.__hash__`'s outer combine (the inner
  xxhash content fingerprint is stable; the outer `hash((mode, dtype, shape, content))` is not). Acceptable for in-process cache keys;
  not safe for persisted hashes. Switching to a fully-stable digest
  is a separate consideration with downstream implications and is
  out of scope for v0.1.0.

### Tests

- New `tests/` directory with 76 tests covering smoke, hash/eq, copy
  semantics, transforms, kept methods, perf budgets, `make_image_grid`
  immutable-input regression, and one regression test per bug surfaced
  during 10 rounds of adversarial review.
- New `.github/workflows/tests.yml` runs the suite on push to `main`
  and on PRs.

### Documentation

- README reorganized: badges (PyPI / Downloads / Python 3.10+ / Tests /
  Publish / License / uv / Ruff / mypy / jaxtyping / Socket / PRs), an
  ASCII flow diagram of `HashableImage` I/O, a "What can you build with
  it?" use-cases section, two embedded visual examples (transformations
  grid + mask workflow), full Basic Usage and Usage Example sections,
  cross-promo "You might also like" table, and the v0.1.0 perf
  highlights moved to the bottom.
- `pixelcache/examples/visuals.py` is the reproducible generator for the
  two README PNGs. All README asset URLs are pinned to the `v0.1.0`
  release tag so PyPI's bundled long description is immune to later
  asset churn on `main`.
- Install snippet uses `uv add pixelcache` as the canonical path
  (Poetry is no longer first-class for this project).
- `uv.lock` removed from the repo — `pyproject.toml` is the source of
  truth for downstream consumers.
- Dropped upper-bound version pins on `tyro` and `pre-commit` (libraries
  should only set lower bounds — upper bounds cause downstream resolver
  conflicts).
