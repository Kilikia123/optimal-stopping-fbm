"""Stable JSON digests, node identifiers and dependency versions."""

import hashlib
import importlib.metadata
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def node_key(h, mu):
    return digest([float(h), float(mu)])[:20]


def dependency_versions():
    return {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "scikit-learn", "iisignature")
    }
