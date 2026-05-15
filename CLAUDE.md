# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with pixelcache.

## What this is

`pixelcache` is the image I/O wrapper used by Hax-CV. It is also published standalone on PyPI. Its job is to give callers a single `HashableImage` type that wraps a `numpy.ndarray`, `torch.Tensor`, or `PIL.Image.Image` with caching-friendly equality semantics, lazy I/O, and convenience operations (resize, crop, blend, color-space conversion, etc.).

Pixelcache is the layer that wraps `cv2.imread` / `cv2.imwrite` / `PIL.Image.open` — those calls are appropriate **inside** pixelcache. The rule "use `HashableImage` instead of direct cv2/PIL" applies to *consumers* (e.g., Hax-CV), not to pixelcache itself.

## Conventions

Hax-CV's `CLAUDE.md` and forbidden-pattern rules apply here, with these unavoidable carveouts:

| Hax-CV rule                                                   | Pixelcache stance                                                                                                                                                                                                                                                     |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cv2.imread` / `cv2.imwrite` / `Image.open` forbidden         | Allowed — pixelcache IS the wrapper                                                                                                                                                                                                                                   |
| `subprocess.run([..., "-m", "hax_3dllm..."])` forbidden       | N/A — pixelcache doesn't run Hax-CV                                                                                                                                                                                                                                   |
| `from klog.path import path_*` required          | Required. `klog` is a runtime dep of pixelcache; the `path_*` helpers are reachable and used. The `Path \| str` type annotation may still be imported (`from pathlib import Path`) but operations (`Path.exists`, `Path.open`, etc.) must go through `path_*`. |
| `from klog import get_logger` instead of `print()`     | Required.                                                                                                                                                                                                                                                             |
| No `Any` / `dict[str, Any]` / `list[Any]`                     | Required. Use TypeVar / ParamSpec / Protocol / explicit unions.                                                                                                                                                                                                       |
| No `# noqa`, `# type: ignore`, `# fmt: off`, `# mypy: ignore` | Required — fix the underlying issue.                                                                                                                                                                                                                                  |
| No `# TODO` / `# FIXME` / `# HACK` / `# XXX`                  | Required. Implement, remove, or ticket.                                                                                                                                                                                                                               |
| Inline `Literal[...]` (non-binary) forbidden                  | Required — define `class Foo(str, Enum)`.                                                                                                                                                                                                                             |
| Files ≤1000 lines                                             | Required.                                                                                                                                                                                                                                                             |
| `@dataclass` (stdlib) forbidden                               | Use `pydantic.dataclasses.dataclass` (already in use) or `pydantic.BaseModel`.                                                                                                                                                                                        |
| `__all__` only in `__init__.py`                               | Required.                                                                                                                                                                                                                                                             |
| Imports inside functions forbidden                            | Required (only documented circular-import exception allowed).                                                                                                                                                                                                         |
| Test sync mandatory                                           | Required — when modifying a source file with a test file, update the test in the same change.                                                                                                                                                                         |

## Build & test

```bash
uv sync

.venv/bin/pre-commit run --all-files
.venv/bin/pytest tests/
.venv/bin/mypy pixelcache/
```

The consumer venv (Hax-CV's `.venv`) has pixelcache editable-installed at `dependencies/pixelcache/`. Worktrees of pixelcache import from their own path — keep a separate venv per worktree if running tests locally.

## Public API surface

Single canonical import path:

```python
from pixelcache import HashableImage, HashableDict, HashableList, BoundingBox, ImageCrop, Points, ImageSize
```

Internal modules live under `pixelcache/_hashable_image/`, `pixelcache/types/`, `pixelcache/data/`, `pixelcache/tools/`. External code MUST NOT import from `pixelcache.main` (that path is removed) or from private submodules — only from `pixelcache` directly.

## Performance contracts

These are enforced via `tests/test_*_perf.py` with `pytest-benchmark`:

| Operation                                  | Budget                          |
| ------------------------------------------ | ------------------------------- |
| `HashableImage(numpy_array)` for 4MP image | \< 100 µs (must NOT touch disk) |
| `HashableImage(path).numpy()` for 4MP JPEG | \< 50 ms (single-decode)        |
| `hash(img)` for an already-hashed instance | \< 1 µs (cached)                |

The benchmark suite is the regression gate. Don't merge changes that move these budgets without a corresponding update.
