import os
import random

import numpy as np
import torch

max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min


def seed_everything(seed: int | None = None, *, workers: bool = False) -> int:
    """Set the seed for pseudo-random number generators in torch, numpy, and.

        Python's random module.

    This function also sets the following environment variables:
    - ``PL_GLOBAL_SEED``: Passed to spawned subprocesses (e.g., ddp_spawn
        backend).
    - ``PL_SEED_WORKERS``: Set to 1 if ``workers=True``.

    Arguments:
        seed (int | None): The seed for the global random state in
            Lightning. If ``None``,
            the function will read the seed from the ``PL_GLOBAL_SEED``
            environment variable.
            If both are ``None`` and the ``PL_GLOBAL_SEED`` environment
            variable is not set,
            then the seed defaults to 0.
        workers (bool): If set to ``True``, configures all dataloaders
            passed to the
            Trainer with a ``worker_init_fn``. If the user already provides
            such a function
            for their dataloaders, setting this argument will have no
            influence. See also:
    :func:`~lightning_fabric.utilities.seed.pl_worker_init_function`.
            Defaults to False.

    Returns:
        None
    Example:
        >>> set_seed(42, workers=True)

    Note:
        The function does not return any value. It modifies the global state
            of several modules and environment variables.

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

    print(f"Seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed
