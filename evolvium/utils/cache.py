import time
from collections import OrderedDict
from typing import Any, Hashable, Optional


class Cache(object):
    def __init__(self, size: int = 100) -> None:
        """
        Initialize a Cache object with a least-recently-used (LRU) eviction policy.

        Args:
            size (int): Maximum number of elements allowed in the cache at any time.

        Returns:
            None
        """
        self.storage = OrderedDict()
        self.limit = size

    def add(self, key: Hashable, value: Any) -> None:
        """
        Add an element to the cache. If the cache exceeds the maximum size, the least-recently-used item is evicted.

        Args:
            key (Hashable): The key used to store and retrieve the value.
            value (Any): The data to be stored in the cache.

        Returns:
            None
        """
        current_time = time.time()
        self.storage[key] = {"value": value, "timestamp": current_time}
        self.storage.move_to_end(key)
        self._enforce_size_constraint()

    def get(self, key: Hashable) -> Optional[Any]:
        """
        Retrieve the value associated with the given key from the cache.

        Args:
            key (Hashable): The key to retrieve the corresponding value.

        Returns:
            Any or None: The value associated with the key if it exists, or None if the key is not found.
        """
        current_time = time.time()
        if key not in self.storage:
            return None
        self.storage[key]["timestamp"] = current_time
        self.storage.move_to_end(key)
        return self.storage[key]["value"]

    def _enforce_size_constraint(self) -> None:
        """
        Ensure the cache does not exceed its size limit by evicting the least-recently-used item if necessary.

        Args:
            None

        Returns:
            None
        """
        if len(self.storage) > self.limit:
            self.storage.popitem(last=False)

    def __repr__(self) -> str:
        """
        Return a string representation of the Cache object for debugging purposes.

        Args:
            None

        Returns:
            str: A dictionary-like string representation of the cache's contents,
                 showing keys, values, and their corresponding timestamps.
        """
        return str(
            {
                key: {"value": meta["value"], "timestamp": meta["timestamp"]}
                for key, meta in self.storage.items()
            }
        )

    def __getitem__(self, key: Hashable) -> Any:
        """Allow dictionary-style access: cache[key]"""
        return self.get(key)

    def __setitem__(self, key: Hashable, value: Any) -> None:
        """Allow dictionary-style assignment: cache[key] = value"""
        self.add(key, value)

    def __contains__(self, key: Hashable) -> bool:
        """Allow 'in' operator: key in cache"""
        return key in self.storage
