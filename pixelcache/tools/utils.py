import os
import random

import numpy as np
import torch
from jaxtyping import UInt8

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed


def color_palette() -> UInt8[np.ndarray, "256 3"]:
    """Generate a color palette array with RGB values.

    This function does not take any arguments. It generates a color palette
        with RGB values and returns it as a NumPy array.

    Returns:
        np.array: A NumPy array of shape (256, 3), where each row represents
            an RGB color. The values in the array range from 0 to 255,
            representing the intensity of red, green, and blue respectively.

    Example:
        >>> color_palette()

    Note:
        The generated color palette can be used for color mapping in data
            visualizations.

    """
    _palette = [
        0,
        0,
        0,
        0,
        0,
        139,
        255,
        255,
        84,
        0,
        255,
        0,
        139,
        0,
        139,
        0,
        128,
        128,
        128,
        128,
        128,
        139,
        0,
        0,
        218,
        165,
        32,
        144,
        238,
        144,
        160,
        82,
        45,
        148,
        0,
        211,
        255,
        0,
        255,
        30,
        144,
        255,
        255,
        218,
        185,
        85,
        107,
        47,
        255,
        140,
        0,
        50,
        205,
        50,
        123,
        104,
        238,
        240,
        230,
        140,
        72,
        61,
        139,
        128,
        128,
        0,
        0,
        0,
        205,
        221,
        160,
        221,
        143,
        188,
        143,
        127,
        255,
        212,
        176,
        224,
        230,
        244,
        164,
        96,
        250,
        128,
        114,
        70,
        130,
        180,
        0,
        128,
        0,
        173,
        255,
        47,
        255,
        105,
        180,
        238,
        130,
        238,
        154,
        205,
        50,
        220,
        20,
        60,
        176,
        48,
        96,
        0,
        206,
        209,
        0,
        191,
        255,
        40,
        40,
        40,
        41,
        41,
        41,
        42,
        42,
        42,
        43,
        43,
        43,
        44,
        44,
        44,
        45,
        45,
        45,
        46,
        46,
        46,
        47,
        47,
        47,
        48,
        48,
        48,
        49,
        49,
        49,
        50,
        50,
        50,
        51,
        51,
        51,
        52,
        52,
        52,
        53,
        53,
        53,
        54,
        54,
        54,
        55,
        55,
        55,
        56,
        56,
        56,
        57,
        57,
        57,
        58,
        58,
        58,
        59,
        59,
        59,
        60,
        60,
        60,
        61,
        61,
        61,
        62,
        62,
        62,
        63,
        63,
        63,
        64,
        64,
        64,
        65,
        65,
        65,
        66,
        66,
        66,
        67,
        67,
        67,
        68,
        68,
        68,
        69,
        69,
        69,
        70,
        70,
        70,
        71,
        71,
        71,
        72,
        72,
        72,
        73,
        73,
        73,
        74,
        74,
        74,
        75,
        75,
        75,
        76,
        76,
        76,
        77,
        77,
        77,
        78,
        78,
        78,
        79,
        79,
        79,
        80,
        80,
        80,
        81,
        81,
        81,
        82,
        82,
        82,
        83,
        83,
        83,
        84,
        84,
        84,
        85,
        85,
        85,
        86,
        86,
        86,
        87,
        87,
        87,
        88,
        88,
        88,
        89,
        89,
        89,
        90,
        90,
        90,
        91,
        91,
        91,
        92,
        92,
        92,
        93,
        93,
        93,
        94,
        94,
        94,
        95,
        95,
        95,
        96,
        96,
        96,
        97,
        97,
        97,
        98,
        98,
        98,
        99,
        99,
        99,
        100,
        100,
        100,
        101,
        101,
        101,
        102,
        102,
        102,
        103,
        103,
        103,
        104,
        104,
        104,
        105,
        105,
        105,
        106,
        106,
        106,
        107,
        107,
        107,
        108,
        108,
        108,
        109,
        109,
        109,
        110,
        110,
        110,
        111,
        111,
        111,
        112,
        112,
        112,
        113,
        113,
        113,
        114,
        114,
        114,
        115,
        115,
        115,
        116,
        116,
        116,
        117,
        117,
        117,
        118,
        118,
        118,
        119,
        119,
        119,
        120,
        120,
        120,
        121,
        121,
        121,
        122,
        122,
        122,
        123,
        123,
        123,
        124,
        124,
        124,
        125,
        125,
        125,
        126,
        126,
        126,
        127,
        127,
        127,
        128,
        128,
        128,
        129,
        129,
        129,
        130,
        130,
        130,
        131,
        131,
        131,
        132,
        132,
        132,
        133,
        133,
        133,
        134,
        134,
        134,
        135,
        135,
        135,
        136,
        136,
        136,
        137,
        137,
        137,
        138,
        138,
        138,
        139,
        139,
        139,
        140,
        140,
        140,
        141,
        141,
        141,
        142,
        142,
        142,
        143,
        143,
        143,
        144,
        144,
        144,
        145,
        145,
        145,
        146,
        146,
        146,
        147,
        147,
        147,
        148,
        148,
        148,
        149,
        149,
        149,
        150,
        150,
        150,
        151,
        151,
        151,
        152,
        152,
        152,
        153,
        153,
        153,
        154,
        154,
        154,
        155,
        155,
        155,
        156,
        156,
        156,
        157,
        157,
        157,
        158,
        158,
        158,
        159,
        159,
        159,
        160,
        160,
        160,
        161,
        161,
        161,
        162,
        162,
        162,
        163,
        163,
        163,
        164,
        164,
        164,
        165,
        165,
        165,
        166,
        166,
        166,
        167,
        167,
        167,
        168,
        168,
        168,
        169,
        169,
        169,
        170,
        170,
        170,
        171,
        171,
        171,
        172,
        172,
        172,
        173,
        173,
        173,
        174,
        174,
        174,
        175,
        175,
        175,
        176,
        176,
        176,
        177,
        177,
        177,
        178,
        178,
        178,
        179,
        179,
        179,
        180,
        180,
        180,
        181,
        181,
        181,
        182,
        182,
        182,
        183,
        183,
        183,
        184,
        184,
        184,
        185,
        185,
        185,
        186,
        186,
        186,
        187,
        187,
        187,
        188,
        188,
        188,
        189,
        189,
        189,
        190,
        190,
        190,
        191,
        191,
        191,
        192,
        192,
        192,
        193,
        193,
        193,
        194,
        194,
        194,
        195,
        195,
        195,
        196,
        196,
        196,
        197,
        197,
        197,
        198,
        198,
        198,
        199,
        199,
        199,
        200,
        200,
        200,
        201,
        201,
        201,
        202,
        202,
        202,
        203,
        203,
        203,
        204,
        204,
        204,
        205,
        205,
        205,
        206,
        206,
        206,
        207,
        207,
        207,
        208,
        208,
        208,
        209,
        209,
        209,
        210,
        210,
        210,
        211,
        211,
        211,
        212,
        212,
        212,
        213,
        213,
        213,
        214,
        214,
        214,
        215,
        215,
        215,
        216,
        216,
        216,
        217,
        217,
        217,
        218,
        218,
        218,
        219,
        219,
        219,
        220,
        220,
        220,
        221,
        221,
        221,
        222,
        222,
        222,
        223,
        223,
        223,
        224,
        224,
        224,
        225,
        225,
        225,
        226,
        226,
        226,
        227,
        227,
        227,
        228,
        228,
        228,
        229,
        229,
        229,
        230,
        230,
        230,
        231,
        231,
        231,
        232,
        232,
        232,
        233,
        233,
        233,
        234,
        234,
        234,
        235,
        235,
        235,
        236,
        236,
        236,
        237,
        237,
        237,
        238,
        238,
        238,
        239,
        239,
        239,
        240,
        240,
        240,
        241,
        241,
        241,
        242,
        242,
        242,
        243,
        243,
        243,
        244,
        244,
        244,
        245,
        245,
        245,
        246,
        246,
        246,
        247,
        247,
        247,
        248,
        248,
        248,
        249,
        249,
        249,
        250,
        250,
        250,
        251,
        251,
        251,
        252,
        252,
        252,
        253,
        253,
        253,
        254,
        254,
        254,
        255,
        255,
        255,
        0,
        0,
        0,
    ]
    return np.asarray(_palette).reshape(-1, 3)
