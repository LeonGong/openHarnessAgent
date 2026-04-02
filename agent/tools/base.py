from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


class Tool(ABC):
    name: str
    description: str
    input_schema: dict[str, Any]

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )

    @abstractmethod
    def run(self, context: dict[str, Any], args: dict[str, Any]) -> str:
        raise NotImplementedError
