"""HashableDict and HashableList — content-addressable dict/list analogues.

Both classes wrap nested `dict`/`list` values in their hashable equivalents
at construction time and cache a content hash that is invalidated on
mutation (`__setitem__`, `__delitem__`, `insert`). They allow tensors,
ndarrays, and PIL images as values and hash them via byte buffers.
"""

from collections.abc import (
    Iterable,
    Iterator,
    MutableMapping,
    MutableSequence,
)
from typing import SupportsIndex, TypeVar, cast, overload

import numpy as np
import torch
from PIL import Image

_T = TypeVar("_T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


class HashableDict(MutableMapping[_KT, _VT]):
    """Hashable dictionary class."""

    def __init__(self, data: dict[_KT, _VT]) -> None:
        """Initialize an instance of the HashableDict class.

        This method converts nested dictionaries and lists within the input
            dictionary into HashableDict and HashableList objects,
            respectively, to initialize an instance of the HashableDict
            class.

        Arguments:
            self (HashableDict): The instance of the HashableDict class.
            data (dict): A dictionary containing key-value pairs where the
                values can be dictionaries or lists.

        Returns:
            None

        """
        new_data: dict[_KT, _VT] = {}
        for k, v in data.items():
            if isinstance(v, dict):
                new_data[k] = cast("_VT", HashableDict(v))
            elif isinstance(v, list):
                new_data[k] = cast("_VT", HashableList(v))
            elif isinstance(v, HashableDict):
                new_data[k] = cast("_VT", HashableDict(v.to_dict()))
            elif isinstance(v, HashableList):
                new_data[k] = cast("_VT", HashableList(v.to_list()))
            else:
                new_data[k] = v
        self.__data = new_data
        # Lazy hash cache; safe because __data is replaced by __setitem__
        # / __delitem__ etc. which must invalidate it.
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
        """Compare two HashableDict instances for equality.

        This method checks if the data in the calling HashableDict instance
            is equal to the data in another HashableDict instance.

        Arguments:
            self ('HashableDict'): The instance of HashableDict calling the
                method.
            other ('HashableDict'): The other instance of HashableDict to
                compare with.

        Returns:
            bool: Returns True if the two HashableDict instances have the
                same data, otherwise returns False.

        """
        if not isinstance(other, HashableDict):
            return NotImplemented
        return self.__data == other.__data

    def to_dict(
        self,
    ) -> dict[_KT, _VT]:
        """Convert the HashableDict object into a dictionary.

        This method recursively converts any nested HashableDict or
            HashableList objects into standard Python dictionaries or lists,
            respectively.
        Arguments: None
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

        Arguments:
            self (HashableDict): The HashableDict object to be duplicated.

        Returns:
            HashableDict: A new HashableDict object that mirrors the
                original.

        """
        return HashableDict(self.__data.copy())

    def values(self) -> Iterable[_VT]:  # type: ignore[override]
        """Retrieve all values from a HashableDict.

        This method iterates over the HashableDict and returns a list
            containing all the values.

        Returns:
            List[Any]: A list containing all the values in the HashableDict.

        """
        return self.__data.values()

    def keys(self) -> Iterable[_KT]:  # type: ignore[override]
        """Retrieve all keys from a HashableDict.

        This method iterates over the HashableDict and returns a list of all
            keys present in the dictionary.

        Returns:
            List[Hashable]: A list containing all keys in the HashableDict.

        """
        return self.__data.keys()

    def items(self) -> Iterable[tuple[_KT, _VT]]:  # type: ignore[override]
        """Retrieve all key-value pairs from the HashableDict.

        This method returns an iterator over the (key, value) pairs in the
            HashableDict.

        Returns:
            Iterator[Tuple[Hashable, Any]]: An iterator over the (key,
                value) pairs in the HashableDict.

        """
        return self.__data.items()

    def __repr__(self) -> str:
        """Return a string representation of the HashableDict object.

        This method generates a string that provides a readable
            representation of the HashableDict object. It can be used for
            debugging and logging purposes.

        Arguments:
            self (HashableDict): The instance of HashableDict object to be
                represented.

        Returns:
            str: A string representation of the HashableDict object.

        """
        return f"HashableDict: {self.__data}"

    def __getitem__(self, __name: _KT, /) -> _VT:
        """Retrieve the value associated with a specific key in a HashableDict.

            object.

        Arguments:
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

    def __setitem__(self, __name: _KT, __value: _VT, /) -> None:
        """Assign a value to a key; invalidates the hash cache."""
        self.__data[__name] = __value
        self._cached_hash = None

    def __delitem__(self, __name: _KT, /) -> None:
        """Remove a key from the dict; invalidates the hash cache."""
        del self.__data[__name]
        self._cached_hash = None

    def __iter__(self) -> Iterator[_KT]:
        """Make instances of the HashableDict class iterable.

        This method makes instances of the HashableDict class iterable by
            returning an iterator over the keys of the dictionary.

        Arguments:
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

        Arguments:
            None
        Returns:
            int: An integer representing the length of the data stored
                within the HashableDict object.

        """
        return len(self.__data)


class HashableList(MutableSequence[_T]):
    """Hashable list class."""

    def __init__(self, data: list[_T]) -> None:
        """Initializes an instance of the HashableList class.

        This method converts any dictionaries or lists within the input list
            to their hashable equivalents
        (HashableDict or HashableList) and stores the modified list in the
            instance.

        Arguments:
            self (HashableList): The instance of the HashableList class.
            data (List[_T]): A list of elements of any type (_T). If the
                elements are dictionaries or lists,
                              they are converted to HashableDict or
                HashableList respectively.

        Returns:
            None

        """
        new_data: list[_T] = []
        for item in data:
            if isinstance(item, dict):
                new_data.append(cast("_T", HashableDict(item)))
            elif isinstance(item, list):
                new_data.append(cast("_T", HashableList(item)))
            elif isinstance(item, HashableDict):
                new_data.append(cast("_T", HashableDict(item.to_dict())))
            elif isinstance(item, HashableList):
                new_data.append(cast("_T", HashableList(item.to_list())))
            else:
                new_data.append(item)
        self.__data = new_data
        self._cached_hash: int | None = None

    def __hash__(self) -> int:
        """Return a cached structural hash for this list.

        First call walks the items, hashing raw `torch.Tensor` /
        `np.ndarray` / `PIL.Image` values via their byte buffers and
        delegating to `hash()` for everything else. Subsequent calls
        return the cached int.
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
        self._cached_hash = hash(frozenset(items))
        return self._cached_hash

    def __eq__(self, other: object) -> bool:
        """Compare the hash values of two HashableList objects.

        This method compares the hash value of the HashableList object
            calling the method (self)
        with the hash value of another HashableList object (other).

        Arguments:
            self ('HashableList'): The HashableList object calling the
                method.
            other ('HashableList'): The HashableList object to compare with.

        Returns:
            bool: Returns True if the hash values of both HashableList
                objects are equal, False otherwise.
                  If the 'other' object is not an instance of HashableList,
                it returns NotImplemented.

        """
        if not isinstance(other, HashableList):
            return NotImplemented
        return self.__hash__() == other.__hash__()

    def to_list(self) -> list[_T]:
        """Convert the HashableList object into a regular Python list.

        This method recursively converts any nested HashableDict or
            HashableList objects into their respective list representations.
        Arguments: None
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

        Arguments:
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

        Arguments:
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

        Arguments:
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

        Arguments:
            __index (int | slice): The index or slice to be retrieved from
                the HashableList object.

        Returns:
            Any: The element or slice of elements from the HashableList
                object.

        """
        if isinstance(__index, slice):
            return HashableList(self.__data[__index])
        return self.__data[__index]

    @overload
    def __setitem__(self, key: SupportsIndex, value: _T, /) -> None: ...

    @overload
    def __setitem__(
        self,
        key: SupportsIndex,
        value: Iterable[_T],
        /,
    ) -> None: ...

    @overload
    def __setitem__(self, key: slice, value: Iterable[_T], /) -> None: ...

    def __setitem__(
        self,
        key: SupportsIndex | slice,
        value: _T | Iterable[_T],
    ) -> None:
        """Assign an item or slice in place; invalidates the hash cache."""
        data_list = self.to_list()
        if isinstance(value, HashableList):
            value = cast("_T | Iterable[_T]", value.to_list())
        if isinstance(key, slice):
            data_list[key] = cast("Iterable[_T]", value)
        else:
            data_list[key] = cast("_T", value)
        self.__data = HashableList(data_list).__data
        self._cached_hash = None

    def __len__(self) -> int:
        """Calculate the length of the HashableList object.

        This method determines the length of the HashableList object by
            returning the length of the data stored within the object.

        Arguments:
            self (HashableList): The HashableList object for which the
                length needs to be determined.

        Returns:
            int: An integer representing the length of the HashableList
                object.

        """
        return len(self.__data)

    @overload
    def __delitem__(self, __index: int, /) -> None: ...

    @overload
    def __delitem__(self, __index: slice, /) -> None: ...

    def __delitem__(self, __index: int | slice, /) -> None:
        """Delete an item or slice; invalidates the hash cache."""
        del self.__data[__index]
        self._cached_hash = None

    def insert(self, __index: int, __value: _T, /) -> None:
        """Insert a value at the given index; invalidates the hash cache."""
        self.__data.insert(__index, __value)
        self._cached_hash = None

    def __mul__(self, other: int) -> "HashableList[_T]":
        """Multiply all elements in the HashableList by a specified integer.

        This method iterates over each element in the HashableList,
            multiplies it by the given integer value,
        and returns a new HashableList with the resulting values.

        Arguments:
            self (HashableList): The current HashableList instance.
            other (int): The integer value to multiply the elements by.

        Returns:
            HashableList: A new HashableList object containing the elements
                of the original HashableList
            multiplied by the specified integer value.

        """
        return HashableList(self.__data * other)
