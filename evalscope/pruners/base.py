# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Base class and in-module registry for benchmark pruning strategies.

A PruningStrategy receives a list of sample descriptors — each carrying the
sample's stable identifier, its per-model scores from a reference run, and
any benchmark-specific metadata — and returns the subset of identifiers that
should be kept.

The interface is intentionally benchmark-agnostic.  The same strategy class
can be used by LiveCodeBench, AA-LCR, and MMMU pruned adapters without
modification.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class PruningStrategy(ABC):
    """
    Abstract base for all pruning strategies.

    Parameters accepted by ``select`` are kept generic on purpose so that
    the same class can serve multiple benchmarks.  Benchmark-specific
    behaviour (e.g. treating ``img_type`` for MMMU or ``contest_date`` for
    LCB) is achieved by subclasses that inspect ``sample['metadata']``.
    """

    @abstractmethod
    def select(
        self,
        samples: List[Dict[str, Any]],
        prune_ratio: float,
        seed: int = 42,
    ) -> List[Any]:
        """Select a subset of samples to keep.

        Args:
            samples: List of sample descriptors.  Each dict must contain:
                - ``"id"``        – stable sample identifier (str or int).
                  Used to cross-reference with the dataset during loading.
                - ``"avg_score"`` – mean score across reference models (float).
                - ``"variance"``  – Bernoulli variance p*(1-p) (float).
                - ``"metadata"``  – arbitrary benchmark-specific dict (may be
                  empty).
            prune_ratio: Fraction of samples to retain.  Must be in (0, 1].
            seed: RNG seed for reproducibility.

        Returns:
            List of selected ``id`` values (same type as the ``"id"`` field in
            the input dicts).
        """
        ...


# ---------------------------------------------------------------------------
# In-module registry  (no external dependency required)
# ---------------------------------------------------------------------------

_PRUNING_REGISTRY: Dict[str, Type[PruningStrategy]] = {}


def register_pruning_strategy(name: str):
    """Class decorator that registers a PruningStrategy under *name*.

    Example::

        @register_pruning_strategy('my_strategy')
        class MyPruner(PruningStrategy):
            ...
    """
    def decorator(cls: Type[PruningStrategy]) -> Type[PruningStrategy]:
        if name in _PRUNING_REGISTRY:
            raise ValueError(
                f"Pruning strategy '{name}' is already registered.  "
                f"Choose a different name or remove the existing registration."
            )
        _PRUNING_REGISTRY[name] = cls
        return cls
    return decorator


def get_pruning_strategy(name: str) -> Type[PruningStrategy]:
    """Retrieve a registered PruningStrategy class by name.

    Args:
        name: The registered strategy name.

    Returns:
        The strategy *class* (not an instance).  Callers instantiate it
        themselves so they can pass constructor arguments.

    Raises:
        ValueError: If *name* is not registered.
    """
    if name not in _PRUNING_REGISTRY:
        available = sorted(_PRUNING_REGISTRY.keys())
        raise ValueError(
            f"Pruning strategy '{name}' not found.  "
            f"Available strategies: {available}"
        )
    return _PRUNING_REGISTRY[name]


def list_pruning_strategies() -> List[str]:
    """Return a sorted list of all registered strategy names."""
    return sorted(_PRUNING_REGISTRY.keys())
