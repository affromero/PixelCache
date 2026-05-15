import os
import random

import numpy as np
import torch
from klog import get_logger

logger = get_logger()

max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min


def seed_everything(
    seed: int | None = None,
    *,
    workers: bool = False,
    verbose: bool = True,
    cuda_deterministic: bool = True,
) -> int:
    """Seed Python, NumPy, and PyTorch RNGs deterministically.

    Sets `PL_GLOBAL_SEED` and `PL_SEED_WORKERS` env vars so spawned
    subprocesses (Lightning ddp_spawn etc.) inherit the seed.

    Args:
        seed: Seed value. If `None`, read from `PL_GLOBAL_SEED` env;
            falls back to `0` if unset or invalid.
        workers: Set `PL_SEED_WORKERS=1` so dataloader workers seed too.
        verbose: Log the chosen seed.
        cuda_deterministic: Force `cudnn.deterministic=True` and
            `cudnn.benchmark=False` when CUDA is available.

    Returns:
        The seed actually applied (after env-var resolution / clamping).

    """
    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = 0
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = 0
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        seed = 0

    if verbose:
        logger.info(f"Seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    # Seed the legacy global numpy RNG (NPY002): downstream code that
    # calls `np.random.*` directly relies on the global state. Modern
    # call sites should construct their own `np.random.default_rng()`.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if cuda_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed
