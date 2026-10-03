"""Binary search over a sorted list."""

class BinarySearch:
    """Stores usernames in sorted order and uses binary search."""

    def __init__(self, items: list[str] = []) -> None:
        """Create an empty sorted list or build from `items`.

        Args:
            items: Usernames to sort and store.
        """
        self.build(items)

    def build(self, items: list[str]) -> None:
        """Sort and store `items`.

        Args:
            items: Usernames to sort and store.
        """
        self._items = sorted(items)
        self.size = len(self._items)

    def contains(self, item: str) -> bool:
        """Return whether `item` occurs using a binary search.

        Args:
            item: Username to search for.

        Returns:
            True when the username is found, otherwise False.
        """
        low = 0
        high = len(self._items) - 1
        while low <= high:
            middle = (low + high) // 2
            current = self._items[middle]
            if current == item:
                return True
            if current < item:
                low = middle + 1
            else:
                high = middle - 1
        return False
