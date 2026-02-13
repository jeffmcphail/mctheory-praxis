"""
User Code Template Framework (Phase 3.11, §16.2-16.6).

Enables users to extend the platform with custom signals, instruments,
execution adapters, etc. — registered via registry.yaml without touching
platform source code.

Templates provide interface contracts. Users subclass and register.
Startup validation ensures compliance before any model runs.

Usage:
    loader = UserCodeLoader("./praxis_user")
    results = loader.load_and_validate()
    # Registered entries are now available in FunctionRegistry
"""

from __future__ import annotations

import importlib
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from praxis.logger.core import PraxisLogger
from praxis.registry import FunctionRegistry


# ═══════════════════════════════════════════════════════════════════
#  Template Interfaces (§16.3)
# ═══════════════════════════════════════════════════════════════════

class SignalTemplate(ABC):
    """Interface contract for user-defined signal generators."""

    @abstractmethod
    def generate(self, prices: Any, params: dict) -> Any:
        """Generate signal series from prices + params."""
        ...


class SizingTemplate(ABC):
    """Interface contract for user-defined position sizers."""

    @abstractmethod
    def size(self, signals: Any, params: dict) -> Any:
        """Compute position sizes from signals."""
        ...


class ExecutionAdapterTemplate(ABC):
    """Interface contract for user-defined execution adapters."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to venue."""
        ...

    @abstractmethod
    def submit_order(self, order: dict) -> str:
        """Submit order, return order_id."""
        ...


class DataSourceTemplate(ABC):
    """Interface contract for user-defined data sources."""

    @abstractmethod
    def fetch(self, tickers: list[str], start: str, end: str) -> Any:
        """Fetch price data for tickers in date range."""
        ...


# Category → required interface
TEMPLATE_INTERFACES: dict[str, type] = {
    "signals": SignalTemplate,
    "sizing": SizingTemplate,
    "execution": ExecutionAdapterTemplate,
    "data_sources": DataSourceTemplate,
}

# §16.6 Sandbox rules
PROHIBITED_MODULES = {
    "subprocess", "os.system", "exec", "eval",
    "socket", "http.client", "urllib.request",
}

# Execution adapters are the exception — they need network access
NETWORK_EXEMPT_CATEGORIES = {"execution"}


# ═══════════════════════════════════════════════════════════════════
#  Validation (§16.5)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """Result of validating one user registration."""
    name: str
    category: str
    passed: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class LoadResult:
    """Result of loading all user code."""
    validations: list[ValidationResult] = field(default_factory=list)
    registered: int = 0
    failed: int = 0

    @property
    def all_passed(self) -> bool:
        return self.failed == 0


def validate_interface(cls: type, category: str) -> list[str]:
    """
    §16.5 step 1: Check interface compliance.

    Verify the class implements all abstract methods from the
    category's template interface.
    """
    errors = []
    interface = TEMPLATE_INTERFACES.get(category)
    if interface is None:
        return errors  # No interface check for unknown categories

    required_methods = {
        name for name, method in inspect.getmembers(interface, predicate=inspect.isfunction)
        if getattr(method, "__isabstractmethod__", False)
    }

    for method_name in required_methods:
        if not hasattr(cls, method_name):
            errors.append(f"Missing required method: {method_name}")
        elif inspect.isabstract(cls) and method_name in {
            name for name, _ in inspect.getmembers(cls)
            if getattr(getattr(cls, name, None), "__isabstractmethod__", False)
        }:
            errors.append(f"Method {method_name} is still abstract")

    return errors


def validate_sandbox(module_path: str, category: str) -> list[str]:
    """
    §16.5 step 3 / §16.6: Sandbox check.

    Verify user code doesn't import prohibited modules.
    Execution adapters are exempt from network restrictions.
    """
    errors = []
    if category in NETWORK_EXEMPT_CATEGORIES:
        return errors  # Execution adapters can use network

    try:
        source_path = Path(module_path.replace(".", "/") + ".py")
        if source_path.exists():
            source = source_path.read_text()
            for prohibited in PROHIBITED_MODULES:
                if prohibited in source:
                    errors.append(
                        f"Prohibited import/call: {prohibited} "
                        f"(only execution adapters may access network)"
                    )
    except Exception:
        pass  # If we can't read source, skip sandbox check

    return errors


def validate_no_conflicts(name: str, category: str, registry: FunctionRegistry) -> list[str]:
    """
    §16.5 step 4: Conflict detection.

    Check if user registration collides with platform-provided name.
    """
    errors = []
    try:
        existing = registry.resolve(category, name)
        if existing is not None:
            errors.append(
                f"Conflict: '{name}' already registered in '{category}'. "
                f"User code cannot override platform defaults."
            )
    except (KeyError, ValueError):
        pass  # No conflict
    return errors


# ═══════════════════════════════════════════════════════════════════
#  Loader (§16.4)
# ═══════════════════════════════════════════════════════════════════

class UserCodeLoader:
    """
    Load and validate user code from registry.yaml.

    Scans user directory for registry.yaml, validates all entries,
    and registers valid ones into the FunctionRegistry.
    """

    def __init__(
        self,
        user_dir: str | Path = "./praxis_user",
        registry: FunctionRegistry | None = None,
    ):
        self._user_dir = Path(user_dir)
        self._registry = registry or FunctionRegistry.instance()
        self._log = PraxisLogger.instance()

    def load_and_validate(self) -> LoadResult:
        """
        Load registry.yaml, validate all entries, register valid ones.

        Returns:
            LoadResult with validation details.
        """
        result = LoadResult()
        registry_path = self._user_dir / "registry.yaml"

        if not registry_path.exists():
            self._log.debug(
                f"No user registry found at {registry_path}",
                tags={"registry", "user_code"},
            )
            return result

        try:
            with open(registry_path) as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            self._log.error(
                f"Failed to parse user registry: {e}",
                tags={"registry", "user_code"},
            )
            return result

        for category, entries in config.items():
            if not isinstance(entries, dict):
                continue

            for name, entry in entries.items():
                vr = self._validate_and_register(name, category, entry)
                result.validations.append(vr)
                if vr.passed:
                    result.registered += 1
                else:
                    result.failed += 1

        self._log.info(
            f"User code: {result.registered} registered, "
            f"{result.failed} failed",
            tags={"registry", "user_code"},
        )

        return result

    def _validate_and_register(
        self,
        name: str,
        category: str,
        entry: dict,
    ) -> ValidationResult:
        """Validate one entry and register if valid."""
        vr = ValidationResult(name=name, category=category, passed=False)

        module_path = entry.get("module")
        class_name = entry.get("class")
        function_name = entry.get("function")

        if not module_path:
            vr.errors.append("Missing 'module' in registry entry")
            return vr

        # Step 1: Import module
        try:
            mod = importlib.import_module(module_path)
        except ImportError as e:
            vr.errors.append(f"Cannot import module '{module_path}': {e}")
            return vr

        # Step 2: Resolve class or function
        target = None
        if class_name:
            target = getattr(mod, class_name, None)
            if target is None:
                vr.errors.append(f"Class '{class_name}' not found in '{module_path}'")
                return vr

            # Interface compliance
            iface_errors = validate_interface(target, category)
            vr.errors.extend(iface_errors)

        elif function_name:
            target = getattr(mod, function_name, None)
            if target is None:
                vr.errors.append(f"Function '{function_name}' not found in '{module_path}'")
                return vr
        else:
            vr.errors.append("Entry must specify 'class' or 'function'")
            return vr

        # Step 3: Sandbox check
        sandbox_errors = validate_sandbox(module_path, category)
        vr.errors.extend(sandbox_errors)

        # Step 4: Conflict detection
        conflict_errors = validate_no_conflicts(name, category, self._registry)
        vr.errors.extend(conflict_errors)

        # Register if no errors
        if not vr.errors:
            try:
                if class_name and inspect.isclass(target):
                    instance = target()
                    self._registry.register(category, name, instance)
                else:
                    self._registry.register(category, name, target)
                vr.passed = True
            except Exception as e:
                vr.errors.append(f"Registration failed: {e}")

        return vr
