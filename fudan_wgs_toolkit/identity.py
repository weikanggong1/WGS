"""Lossless family and individual identifiers used throughout the toolkit."""
from __future__ import annotations

import json
import unicodedata
import numpy as np


def _identifier(value, *, where):
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8")
    if not isinstance(value, (str, np.str_)):
        raise ValueError(f"{where}: FID and IID must be strings")
    value = str(value)
    if not value or any(character.isspace() or unicodedata.category(character).startswith("C")
                        for character in value):
        raise ValueError(f"{where}: FID and IID must be nonempty strings without whitespace or control characters")
    return value


def validate_sample_pairs(values, *, where="sample pairs"):
    """Return unique ``[n,2]`` FID/IID strings without numeric normalization.

    Family ``0``, leading zeroes, letters and separators are preserved. Only
    duplicate complete pairs are rejected; the same IID in different families
    remains a different individual.
    """
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError(f"{where}: sample pairs must have shape [n,2] in FID,IID order")
    pairs, seen = [], set()
    for row in raw:
        pair = tuple(_identifier(value, where=where) for value in row)
        if pair in seen:
            raise ValueError(f"{where}: duplicate FID/IID pair")
        seen.add(pair)
        pairs.append(pair)
    return np.asarray(pairs, dtype=str).reshape(len(pairs), 2)


def sample_keys(values, *, where="sample pairs"):
    """Encode complete FID/IID pairs as collision-free canonical JSON keys."""
    pairs = validate_sample_pairs(values, where=where)
    return np.asarray([json.dumps(list(pair), ensure_ascii=False, separators=(",", ":"))
                       for pair in pairs], dtype=str)


def pairs_from_keys(values, *, where="sample keys"):
    """Decode canonical keys, rejecting ambiguous or noncanonical identities."""
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{where}: sample keys must be a one-dimensional vector")
    pairs = []
    for key in raw:
        if isinstance(key, (bytes, np.bytes_)):
            key = bytes(key).decode("utf-8")
        if not isinstance(key, (str, np.str_)):
            raise ValueError(f"{where}: sample keys must be strings")
        key = str(key)
        try:
            pair = json.loads(key)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{where}: invalid FID/IID JSON key") from error
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{where}: each key must encode [FID,IID]")
        pairs.append(pair)
    validated = validate_sample_pairs(np.asarray(pairs, dtype=object).reshape(len(pairs), 2), where=where)
    if not np.array_equal(sample_keys(validated, where=where), raw.astype(str)):
        raise ValueError(f"{where}: sample keys are not canonical")
    return validated


__all__ = ["validate_sample_pairs", "sample_keys", "pairs_from_keys"]
