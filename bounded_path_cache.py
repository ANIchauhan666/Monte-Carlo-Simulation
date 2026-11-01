"""Small byte-bounded cache for reusable Monte Carlo path arrays."""

from collections import OrderedDict


class BoundedPathCache:
    """LRU cache whose retained NumPy path arrays stay under a byte budget."""

    def __init__(self, max_bytes=96 * 1024 * 1024):
        if not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer")
        self.max_bytes = max_bytes
        self._bytes = 0
        self._items = OrderedDict()

    def get_or_create(self, key, factory, cache=True):
        if cache and key in self._items:
            self._items.move_to_end(key)
            return self._items[key]

        value = factory()
        size = int(value.nbytes)
        if not cache or size > self.max_bytes:
            return value

        while self._items and self._bytes + size > self.max_bytes:
            _, old_value = self._items.popitem(last=False)
            self._bytes -= int(old_value.nbytes)

        self._items[key] = value
        self._bytes += size
        return value

    @property
    def cached_bytes(self):
        return self._bytes
