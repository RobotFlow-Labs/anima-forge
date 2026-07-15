from __future__ import annotations

import importlib
import logging
from pathlib import Path

from forge.teachers.base import TeacherAdapter

logger = logging.getLogger(__name__)


class TeacherRegistry:
    """Singleton registry for teacher model adapters.

    Auto-discovers adapters in the teachers/ package.
    Supports manual registration for external adapters.

    Usage:
        registry = TeacherRegistry()
        registry.auto_discover()
        adapter = registry.create("openvla-7b")
        adapter.load(Path("/models/openvla--openvla-7b"))
        chunk = adapter.predict(image, "pick up the red block")
    """

    _instance: TeacherRegistry | None = None
    _adapters: dict[str, type[TeacherAdapter]]
    _discovered: bool

    def __new__(cls) -> TeacherRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._adapters = {}
            cls._instance._discovered = False
        return cls._instance

    def register(self, name: str, adapter_class: type[TeacherAdapter]) -> None:
        """Register a teacher adapter class."""
        self._adapters[name] = adapter_class
        logger.debug(f"Registered teacher adapter: {name}")

    def create(self, name: str) -> TeacherAdapter:
        """Create an instance of a registered adapter."""
        if not self._discovered:
            self.auto_discover()
        if name not in self._adapters:
            available = ", ".join(self._adapters.keys())
            raise KeyError(f"Unknown teacher '{name}'. Available: {available}")
        return self._adapters[name]()

    def list_teachers(self) -> list[str]:
        """List all registered teacher names."""
        if not self._discovered:
            self.auto_discover()
        return sorted(self._adapters.keys())

    def auto_discover(self) -> None:
        """Auto-discover adapter modules in the teachers package."""
        if self._discovered:
            return

        teachers_dir = Path(__file__).parent
        for py_file in teachers_dir.glob("*_adapter.py"):
            module_name = py_file.stem
            try:
                module = importlib.import_module(f"forge.teachers.{module_name}")
                # Each adapter module must have ADAPTER_NAME and ADAPTER_CLASS
                if hasattr(module, "ADAPTER_NAME") and hasattr(module, "ADAPTER_CLASS"):
                    self.register(module.ADAPTER_NAME, module.ADAPTER_CLASS)
            except Exception as e:
                logger.warning(f"Failed to discover {module_name}: {e}")

        self._discovered = True
        logger.info(f"Discovered {len(self._adapters)} teacher adapters: {self.list_teachers()}")

    def reset(self) -> None:
        """Reset registry (for testing)."""
        self._adapters.clear()
        self._discovered = False


def get_registry() -> TeacherRegistry:
    """Get the global teacher registry."""
    return TeacherRegistry()
