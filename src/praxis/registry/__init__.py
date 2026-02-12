"""
Function Registry (§7).

Resolves string names from YAML configs to concrete implementations.
"category.name" → class instance.

Usage:
    registry = FunctionRegistry.instance()
    signal_fn = registry.resolve("signals", "sma_crossover")
    sizer = registry.resolve("sizing", "fixed_fraction")

All functions are registered at import time via register() or
bulk-loaded from a registry config dict.
"""

from __future__ import annotations

import importlib
import threading
from typing import Any, Optional


class FunctionRegistry:
    """
    §7.1: Global function registry. Singleton.

    Maps "category.name" → module.Class, with lazy instantiation
    and instance caching.
    """

    _instance: Optional["FunctionRegistry"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._registry: dict[str, str] = {}      # "category.name" → "module.Class"
        self._instances: dict[str, object] = {}   # Cached instances

    @classmethod
    def instance(cls) -> "FunctionRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton. For testing only."""
        with cls._lock:
            cls._instance = None

    # ── Registration ──────────────────────────────────────────────

    def register(self, category: str, name: str, module_path: str, class_name: str) -> None:
        """
        Register a function.

        Args:
            category: "signals", "sizing", "construction", etc.
            name: "sma_crossover", "fixed_fraction", etc.
            module_path: "praxis.signals.trend"
            class_name: "SMACrossover"
        """
        key = f"{category}.{name}"
        self._registry[key] = f"{module_path}.{class_name}"
        # Invalidate cached instance if re-registering
        self._instances.pop(key, None)

    def register_class(self, category: str, name: str, cls: type) -> None:
        """
        Register a class directly (no lazy import needed).
        Useful for built-in functions registered at import time.
        """
        key = f"{category}.{name}"
        self._registry[key] = f"{cls.__module__}.{cls.__name__}"
        # Pre-cache the instance
        self._instances[key] = cls()

    def register_instance(self, category: str, name: str, instance: object) -> None:
        """
        Register a pre-built instance directly.
        Useful for functions with constructor args.
        """
        key = f"{category}.{name}"
        module_path = f"{type(instance).__module__}.{type(instance).__name__}"
        self._registry[key] = module_path
        self._instances[key] = instance

    def register_from_config(self, config: dict[str, dict[str, dict]]) -> int:
        """
        §7.2: Bulk register from a registry config dict.

        Args:
            config: {category: {name: {module: str, class: str}}}

        Returns count of functions registered.
        """
        count = 0
        for category, functions in config.items():
            for name, spec in functions.items():
                self.register(
                    category, name,
                    spec["module"], spec["class"],
                )
                count += 1
        return count

    # ── Resolution ────────────────────────────────────────────────

    def resolve(self, category: str, name: str) -> object:
        """
        §7.1: Resolve a function by category and name.

        Lazy-loads the module and instantiates the class on first access.
        Subsequent calls return the cached instance.

        Raises:
            KeyError: If category.name is not registered.
            ImportError: If the module cannot be imported.
            AttributeError: If the class doesn't exist in the module.
        """
        key = f"{category}.{name}"

        if key in self._instances:
            return self._instances[key]

        if key not in self._registry:
            available = self.list_category(category)
            raise KeyError(
                f"Function '{name}' not found in category '{category}'. "
                f"Available: {', '.join(available) if available else 'none'}"
            )

        full_path = self._registry[key]
        module_path, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        instance = cls()
        self._instances[key] = instance
        return instance

    def has(self, category: str, name: str) -> bool:
        """Check if a function is registered."""
        return f"{category}.{name}" in self._registry

    # ── Introspection ─────────────────────────────────────────────

    def list_category(self, category: str) -> list[str]:
        """List all function names in a category."""
        prefix = f"{category}."
        return sorted(
            key[len(prefix):] for key in self._registry if key.startswith(prefix)
        )

    def list_categories(self) -> list[str]:
        """List all categories."""
        return sorted({key.split(".")[0] for key in self._registry})

    def describe(self) -> dict[str, list[str]]:
        """Full registry listing: {category: [names]}."""
        result: dict[str, list[str]] = {}
        for key in sorted(self._registry):
            category, name = key.split(".", 1)
            result.setdefault(category, []).append(name)
        return result

    @property
    def count(self) -> int:
        return len(self._registry)
