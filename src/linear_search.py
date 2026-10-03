"""Linear search in a list."""

class LinearSearch:
    """Stores usernames in a list and searches linearly."""

    def __init__(self, items: list[str] = []) -> None:
        """Create an empty list or build from `items`.

        Args:
            items: Usernames list.
        """
        self.build(items)

    def build(self, items: list[str]) -> None:
        """Replace stored values with `items` in the input order.

        Args:
            items: Usernames to store.
        """
        self._items = list(items)
        self.size = len(self._items)

    def contains(self, item: str) -> bool:
        """Return whether `item` appears in the list using a linear search.

        Args:
            item: Username to search for.

        Returns:
            True when an equal username is found, otherwise False.
        """
        for stored_item in self._items:
            if stored_item == item:
                return True
        return False
