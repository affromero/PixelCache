import functools
import os
import random
import string
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

from beartype import beartype

_T = TypeVar("_T")

DISABLE_LRU_CACHE = os.getenv("DISABLE_LRU_CACHE", "False") == "True"


def lru_cache(maxsize: int | None = 128) -> Callable[[_T], _T]:
    """Decorate a function with functools.lru_cache, unless in unit test mode.

    This decorator wraps a function with functools.lru_cache, providing
        caching
    functionality for the function's results. However, if the code is
        running in
    unit test mode, the decorator is a no-op and does not provide caching.

    Arguments:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function. If in unit test mode, returns the
            original function
        without caching. Otherwise, returns the function wrapped with
            lru_cache, providing
        caching of its results.

    Example:
        >>> @unit_test_lru_cache_decorator
        >>> def example_function(arg1, arg2):
        >>>     return arg1 + arg2
    Note:
        This decorator is useful for preventing caching during unit tests,
            where
        repeated function calls with the same arguments are often needed.

    """

    def decorator(func: Callable[[_T], _T]) -> Callable[[_T], _T]:
        """Apply an LRU cache to the input function or return the function.

            itself.

        This decorator function applies an LRU cache to the given function
            if the code is not running in unittest mode. If it is in
            unittest mode, the function is returned as is.

        Arguments:
            func (Callable[..., Any]): The function to be decorated.

        Returns:
            Callable[..., Any]: The decorated function with an applied LRU
                cache if not in unittest mode, or the original function if
                in unittest mode.

        Example:
            >>> @apply_cache
            >>> def test_func(x, y):
            >>>     return x + y
        Note:
            This decorator is useful for optimizing functions that are
                called repeatedly with the same arguments, but should not be
                used in unittest mode to avoid cached results.

        """
        # Use cache when not in unit test mode
        if not DISABLE_LRU_CACHE:
            # Apply functools lru_cache
            return functools.lru_cache(maxsize=maxsize, typed=True)(func)

        # No-operation decorator
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Act as a wrapper for another function, preserving its.

                metadata and allowing flexible argument passing.

            This function can be used to wrap another function,
                maintaining its metadata. It accepts any number of
                positional and keyword arguments, which are passed
                directly to the wrapped function.

            Arguments:
                *args (Any): Represents any number of positional
                    arguments that can be passed to the wrapped
                    function.
                **kwargs (Any): Represents any number of keyword
                    arguments that can be passed to the wrapped
                    function.

            Returns:
                Any: Returns the result of calling the wrapped function
                    with the provided arguments and keyword arguments.

            Example:
                >>> wrapped_function =
                    wrapper_function(original_function, arg1, arg2,
                    keyword_arg1=value1)

            Note:
                The wrapped function and its arguments are not specified
                    until the wrapper function is called.

            """
            return func(*args, **kwargs)

        return wrapper

    return cast(Callable[[_T], _T], decorator)


def get_cache_dir() -> Path:
    """Locate a platform-appropriate cache directory for flit to use.

    This function identifies the appropriate cache directory for the
        specified platform and app. It does not ensure
    that the cache directory exists.

    Arguments:
        platform (str): The platform for which the cache directory is to be
            located.
        app (str): The application for which the cache directory is to be
            located.
        flit (bool | None): A flag indicating if flit is to be used.
            Defaults to None.

    Returns:
        str: The path of the located cache directory.

    Example:
        >>> locate_cache_dir("Windows", "flit", flit=True)

    Note:
        The function does not create the cache directory, it only locates
            the appropriate directory for the given platform and app.

    """
    # Linux, Unix, AIX, etc.
    if os.name == "posix" and sys.platform != "darwin":
        # use ~/.cache if empty OR not set
        xdg = os.environ.get("XDG_CACHE_HOME", None) or Path.expanduser(
            Path("~/.cache"),
        )
        cache_dir = Path(xdg)

    # Mac OS
    elif sys.platform == "darwin":
        cache_dir = Path(Path.expanduser(Path("~")), "Library/Caches")

    # Windows
    else:
        local = os.environ.get("LOCALAPPDATA", None) or Path.expanduser(
            Path("~\\AppData\\Local"),
        )
        cache_dir = Path(local)

    return cache_dir


@beartype
def sha256sum(filename: str) -> str:
    """Calculate the SHA-256 hash of a file and return the first 8 characters.

    This function takes a filename as an argument, calculates the SHA-256
        hash of the file, and returns the first 8 characters of the hash.

    Arguments:
        filename (str): A string representing the name of the file for which
            the SHA-256 hash needs to be calculated.

    Returns:
        str: A string containing the first 8 characters of the SHA-256 hash
            of the file.

    Example:
        >>> calculate_file_hash("example.txt")

    Note:
        The file must exist in the current working directory or a full path
            must be provided.

    """
    from sha256sum import sha256sum as _sha256sum

    return _sha256sum(filename)[:8]


@beartype
def pseudo_hash(idx: int, length: int = 6) -> str:
    """Generate a pseudo-random hash based on the given index and length.

    Arguments:
        idx (int): The index used to seed the random number generator.
        length (int, optional): The length of the hash to be generated.
            Defaults to 6.

    Returns:
        str: A string representing the pseudo-random hash generated based on
            the given index and length.

    Example:
        >>> generate_hash(10, 6)

    Note:
        The hash generated is pseudo-random, meaning it will generate the
            same result if the same index and length are provided.

    """
    random.seed(idx)
    return "".join(random.choice(string.ascii_letters) for _ in range(length))  # noqa: S311
