"""HashableDict and HashableList — immutable, content-addressable analogues
of `dict` and `list` for use as cache keys and hashable parameters.

Both classes are **immutable after construction**. There is no
`__setitem__`, `__delitem__`, `insert`, or `append` — the constructor
recursively wraps nested `dict` / `list` values into hashable
equivalents, and deep-copies mutable leaf values (tensors, ndarrays,
PIL images) so external mutation of the source can't invalidate the
cached structural hash.

Equality is content-equality (with array-aware comparison for tensor /
ndarray / PIL values); both `__eq__` and `__hash__` are short-circuit /
cached for hot-path cache-key usage.
"""

from collections.abc import (
    ItemsView,
    Iterator,
    KeysView,
    Mapping,
    Sequence,
    ValuesView,
)
from typing import SupportsIndex, TypeVar, cast, overload

import numpy as np
import torch
from PIL import Image

_T = TypeVar("_T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def _wrap_value(v: object) -> object:
    """Wrap nested containers and deep-copy mutable leaf values.

    Maps:
    - `dict` → `HashableDict` (recursive wrap of values).
    - `list` → `HashableList` (recursive wrap of items).
    - `HashableDict` / `HashableList` → fresh instance built from
      `to_dict()` / `to_list()` so the new wrapper doesn't share state.
    - `np.ndarray` → independent `.copy()` (prevents constructor
      aliasing from invalidating the cached hash later).
    - `torch.Tensor` → `.detach().cpu().clone()` (same reason).
    - `PIL.Image` → `.copy()`.
    - Everything else: returned as-is (assumed already immutable).
    """
    if isinstance(v, dict):
        return HashableDict(v)
    if isinstance(v, list):
        return HashableList(v)
    if isinstance(v, HashableDict):
        return HashableDict(v.to_dict())
    if isinstance(v, HashableList):
        return HashableList(v.to_list())
    if isinstance(v, np.ndarray):
        return v.copy()
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().clone()
    if isinstance(v, Image.Image):
        return v.copy()
    return v


def _compare_values(a: object, b: object) -> bool:
    """Compare two HashableDict/HashableList values for content equality.

    Tensor → `torch.equal`. ndarray → `np.array_equal`. PIL → mode +
    size + bytes (PIL's own `__eq__` is object-identity, not content).
    Anything else falls back to `==`. Returns `False` on type mismatch
    so dict eq cannot raise on heterogeneous values.
    """
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return bool(torch.equal(a, b))
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return bool(np.array_equal(a, b))
    if isinstance(a, Image.Image) and isinstance(b, Image.Image):
        return (
            a.mode == b.mode
            and a.size == b.size
            and a.tobytes() == b.tobytes()
        )
    if type(a) is not type(b):
        return False
    return bool(a == b)


class HashableDict(Mapping[_KT, _VT]):
    """Immutable, content-addressable dictionary.

    Inherits from `collections.abc.Mapping` (read-only protocol).
    Construction deep-copies/wraps every value so mutation of the
    source can't invalidate the cached hash. To "modify", construct
    a new instance with the desired overrides.
    """

    def __init__(self, data: dict[_KT, _VT]) -> None:
        """Initialize a HashableDict, wrapping nested containers.

        Raw `dict` values become `HashableDict`; raw `list` values
        become `HashableList`. Existing HashableDict / HashableList
        values are deep-copied to make this instance independent of
        the caller's structure. Other value types pass through.
        """
        self.__data: dict[_KT, _VT] = {
            k: cast("_VT", _wrap_value(v)) for k, v in data.items()
        }
        # Lazy hash cache; every mutation path resets this to None.
        self._cached_hash: int | None = None

    def __hash__(self) -> int:
        """Return a cached structural hash for this dict.

        First call walks the items, hashing raw `torch.Tensor` /
        `np.ndarray` / `PIL.Image` values via their byte buffers and
        delegating to `hash()` for everything else (HashableImage,
        HashableDict, HashableList — which now cache their own hashes —
        and ordinary hashable types). Subsequent calls return the
        cached int.

        Returns:
            Hash int derived from a `frozenset` of `(key, value-hash)`
            tuples.

        """
        if self._cached_hash is not None:
            return self._cached_hash
        items: dict[_KT, int] = {}
        for k, v in self.__data.items():
            if isinstance(v, torch.Tensor):
                items[k] = hash(v.detach().cpu().numpy().tobytes())
            elif isinstance(v, np.ndarray | Image.Image):
                items[k] = hash(v.tobytes())
            else:
                items[k] = hash(v)
        self._cached_hash = hash(frozenset(items.items()))
        return self._cached_hash

    def __eq__(self, other: object) -> bool:
        """Compare key-by-key with content-aware value comparison.

        Short-circuits on identity, type, hash mismatch, and key-set
        mismatch. Per-value comparison uses `_compare_values` so
        tensors / ndarrays / PIL images are compared by content (and
        PIL specifically by mode + size + bytes — PIL's own `__eq__`
        is object-identity).
        """
        if self is other:
            return True
        if not isinstance(other, HashableDict):
            return NotImplemented
        if self.__hash__() != other.__hash__():
            return False
        if self.__data.keys() != other.__data.keys():
            return False
        return all(
            _compare_values(a, other.__data[k]) for k, a in self.__data.items()
        )

    def to_dict(
        self,
    ) -> dict[_KT, _VT]:
        """Convert the HashableDict object into a dictionary.

        This method recursively converts any nested HashableDict or
            HashableList objects into standard Python dictionaries or lists,
            respectively.

        Returns:
            dict: A dictionary containing the key-value pairs of the
                HashableDict object. Any nested HashableDict or HashableList
                objects are converted into dictionaries or lists,
                respectively.

        """
        to_dict: dict[_KT, _VT] = {}
        for k, v in self.__data.items():
            if isinstance(v, HashableDict):
                to_dict[k] = cast("_VT", v.to_dict())
            elif isinstance(v, HashableList):
                to_dict[k] = cast("_VT", v.to_list())
            else:
                to_dict[k] = v
        return to_dict

    def copy(self) -> "HashableDict[_KT, _VT]":
        """Create a copy of the HashableDict object.

        This method generates an exact replica of the current HashableDict
            object,
        preserving all key-value pairs in the new instance.

        Args:
            self (HashableDict): The HashableDict object to be duplicated.

        Returns:
            HashableDict: A new HashableDict object that mirrors the
                original.

        """
        return HashableDict(self.__data.copy())

    def values(self) -> ValuesView[_VT]:
        """Return a view over the underlying dict's values."""
        return self.__data.values()

    def keys(self) -> KeysView[_KT]:
        """Return a view over the underlying dict's keys."""
        return self.__data.keys()

    def items(self) -> ItemsView[_KT, _VT]:
        """Return a view over the underlying dict's (key, value) pairs."""
        return self.__data.items()

    def __repr__(self) -> str:
        """Return a string representation of the HashableDict object.

        This method generates a string that provides a readable
            representation of the HashableDict object. It can be used for
            debugging and logging purposes.

        Args:
            self (HashableDict): The instance of HashableDict object to be
                represented.

        Returns:
            str: A string representation of the HashableDict object.

        """
        return f"HashableDict: {self.__data}"

    def __getitem__(self, __name: _KT, /) -> _VT:
        """Retrieve the value associated with a specific key in a HashableDict.

            object.

        Args:
            __name (_KT): The key for which the associated value needs to be
                retrieved.

        Returns:
            _VT: The value associated with the specified key in the
                HashableDict object.

        """
        if __name not in self.__data:
            msg = f"Key {__name} not found in HashableDict"
            raise KeyError(msg)
        return self.__data[__name]

    def __iter__(self) -> Iterator[_KT]:
        """Make instances of the HashableDict class iterable.

        This method makes instances of the HashableDict class iterable by
            returning an iterator over the keys of the dictionary.

        Args:
            self (HashableDict): The instance of the HashableDict class.

        Returns:
            Iterator: An iterator object that can traverse through all the
                keys of the dictionary stored in the HashableDict instance.

        """
        return iter(self.__data)

    def __len__(self) -> int:
        """Return the length of the HashableDict object.

        This method computes the length of the HashableDict object by
            returning the length of the data stored within it.

        Args:
            None
        Returns:
            int: An integer representing the length of the data stored
                within the HashableDict object.

        """
        return len(self.__data)


class HashableList(Sequence[_T]):
    """Immutable, content-addressable ordered list.

    Inherits from `collections.abc.Sequence` (read-only protocol).
    Construction deep-copies/wraps every item so mutation of the
    source can't invalidate the cached hash. To "modify", construct
    a new instance from the desired iterable.
    """

    def __init__(self, data: list[_T]) -> None:
        """Initializes an instance of the HashableList class.

        This method converts any dictionaries or lists within the input list
            to their hashable equivalents
        (HashableDict or HashableList) and stores the modified list in the
            instance.

        Args:
            self (HashableList): The instance of the HashableList class.
            data (List[_T]): A list of elements of any type (_T). If the
                elements are dictionaries or lists,
                              they are converted to HashableDict or
                HashableList respectively.

        Returns:
            None

        """
        self.__data: list[_T] = [
            cast("_T", _wrap_value(item)) for item in data
        ]
        self._cached_hash: int | None = None

    def __hash__(self) -> int:
        """Return a cached order-sensitive structural hash for this list.

        First call walks the items in-order, hashing raw `torch.Tensor`
        / `np.ndarray` / `PIL.Image` values via their byte buffers and
        delegating to `hash()` for everything else. The per-item hashes
        are combined as a `tuple` (not `frozenset`) so order and
        multiplicity matter — `[1, 2]`, `[2, 1]`, and `[1, 1, 2]` are
        distinct.
        """
        if self._cached_hash is not None:
            return self._cached_hash
        items: list[int] = []
        for item in self.__data:
            if isinstance(item, torch.Tensor):
                items.append(hash(item.detach().cpu().numpy().tobytes()))
            elif isinstance(item, np.ndarray | Image.Image):
                items.append(hash(item.tobytes()))
            else:
                items.append(hash(item))
        self._cached_hash = hash(tuple(items))
        return self._cached_hash

    def __eq__(self, other: object) -> bool:
        """Compare element-wise in order, with content-aware comparison.

        Short-circuits on identity, type, hash, and length mismatch.
        Each element is compared via `_compare_values` (tensors,
        ndarrays, and PIL images by content; PIL specifically by mode
        + size + bytes).
        """
        if self is other:
            return True
        if not isinstance(other, HashableList):
            return NotImplemented
        if self.__hash__() != other.__hash__():
            return False
        if len(self.__data) != len(other.__data):
            return False
        return all(
            _compare_values(a, b)
            for a, b in zip(self.__data, other.__data, strict=True)
        )

    def to_list(self) -> list[_T]:
        """Convert the HashableList object into a regular Python list.

        This method recursively converts any nested HashableDict or
            HashableList objects into their respective list representations.

        Returns:
            List: A list containing the elements of the HashableList object,
                with any nested HashableDict or HashableList objects
                converted into regular Python lists.

        """
        to_list = []
        for idx in range(len(self.__data)):
            if isinstance(self.__data[idx], HashableDict):
                to_list.append(
                    cast(
                        "_T",
                        cast(
                            "HashableDict[_KT, _VT]",  # type: ignore[valid-type]
                            self.__data[idx],
                        ).to_dict(),
                    ),
                )
            elif isinstance(self.__data[idx], HashableList):
                to_list.append(
                    cast(
                        "_T",
                        cast("HashableList[_T]", self.__data[idx]).to_list(),
                    ),
                )
            else:
                to_list.append(self.__data[idx])
        return to_list

    def __repr__(self) -> str:
        """Return a string representation of the HashableList object.

        This method transforms the HashableList object into a string format.
            The string contains the class name 'HashableList' followed by
            the data stored in the object.

        Args:
            self (HashableList): The HashableList object itself.

        Returns:
            str: A string representation of the HashableList object. The
                string includes the class name 'HashableList' and the data
                stored in the object.

        """
        return f"HashableList: {self.__data}"

    def copy(self) -> "HashableList[_T]":
        """Create a copy of the HashableList object.

        This method generates a new HashableList object by duplicating the
            data stored within the original list.

        Args:
            self (HashableList): The HashableList object to be copied.

        Returns:
            HashableList: A new HashableList object containing the same data
                as the original list.

        """
        return HashableList(self.__data.copy())

    def __iter__(self) -> Iterator[_T]:
        """Enable iteration over instances of the HashableList class.

        This method makes instances of the HashableList class iterable,
            allowing
        them to be used in a for loop or any other iteration context.

        Args:
            self (HashableList): The instance of the HashableList class.

        Returns:
            Iterator: An iterator object that enables iteration over the
                data
            stored in the HashableList instance.

        """
        return iter(self.__data)

    @overload
    def __getitem__(self, __index: SupportsIndex, /) -> _T: ...

    @overload
    def __getitem__(self, __index: slice, /) -> "HashableList[_T]": ...

    def __getitem__(
        self,
        __index: SupportsIndex | slice,
        /,
    ) -> _T | "HashableList[_T]":
        """Retrieve an element or a slice of elements from the HashableList.

            object.

        This method allows for retrieving an element or a slice of elements
            from the HashableList object.

        Args:
            __index (int | slice): The index or slice to be retrieved from
                the HashableList object.

        Returns:
            Any: The element or slice of elements from the HashableList
                object.

        """
        if isinstance(__index, slice):
            return HashableList(self.__data[__index])
        return self.__data[__index]

    def __len__(self) -> int:
        """Calculate the length of the HashableList object.

        This method determines the length of the HashableList object by
            returning the length of the data stored within the object.

        Args:
            self (HashableList): The HashableList object for which the
                length needs to be determined.

        Returns:
            int: An integer representing the length of the HashableList
                object.

        """
        return len(self.__data)

    def __mul__(self, other: int) -> "HashableList[_T]":
        """Multiply all elements in the HashableList by a specified integer.

        This method iterates over each element in the HashableList,
            multiplies it by the given integer value,
        and returns a new HashableList with the resulting values.

        Args:
            self (HashableList): The current HashableList instance.
            other (int): The integer value to multiply the elements by.

        Returns:
            HashableList: A new HashableList object containing the elements
                of the original HashableList
            multiplied by the specified integer value.

        """
        return HashableList(self.__data * other)
