"""
strategies/base.py
==================
Abstract base class for all MCb backtest strategies.
Each strategy consumes a DataFrame with MCb columns pre-computed and
returns the same DataFrame with entry/exit signal columns added.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import pandas as pd


@dataclass
class ParamSpec:
    """Describes a single configurable parameter for the kickoff UI."""
    name: str
    label: str
    type: str            # "float" | "int" | "bool" | "select"
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list | None = None
    help: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "type": self.type,
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "options": self.options,
            "help": self.help,
        }


class MCBStrategy(ABC):
    """
    Base class for all Market Cipher B backtest strategies.

    Subclasses must set class-level attributes and implement generate_signals().
    """

    id: str = ""
    name: str = ""
    description: str = ""
    param_specs: list[ParamSpec] = []

    def __init__(self, params: dict | None = None):
        # Merge caller params with defaults
        self.params: dict = {}
        for spec in self.param_specs:
            self.params[spec.name] = spec.default
        if params:
            self.params.update(params)

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add signal columns to df and return it.

        Required output columns:
            entry         : bool  — open a long position at this bar's close
            exit_signal   : bool  — close open long position at this bar's close
            signal_label  : str   — human-readable reason (shown in trade log)
        """
        pass

    @classmethod
    def to_info(cls) -> dict:
        return {
            "id": cls.id,
            "name": cls.name,
            "description": cls.description,
            "params": [s.to_dict() for s in cls.param_specs],
        }
