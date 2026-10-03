"""Deterministic hashing shared by the hash table and filters."""

import hashlib


def stable_hash(value: str | bytes, salt: int = 0) -> int:
    """Return a stable 64-bit hash of a string or bytes.

    Args:
        value: Username or fingerprint bytes to hash.
        salt: Integer that separates independent uses of the hash.

    Returns:
        An integer.
    """
    data = value.encode("utf-8") if isinstance(value, str) else value
    person = salt.to_bytes(8, "little")
    digest = hashlib.blake2b(data, digest_size=8, person=person).digest()
    return int.from_bytes(digest, "little")
