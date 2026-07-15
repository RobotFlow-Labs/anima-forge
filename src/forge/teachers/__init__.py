"""FORGE Teacher Adapters — Universal VLA teacher interface."""

from forge.teachers.base import ActionChunk, TeacherAdapter, TeacherInfo
from forge.teachers.registry import TeacherRegistry, get_registry

__all__ = ["ActionChunk", "TeacherAdapter", "TeacherInfo", "TeacherRegistry", "get_registry"]
