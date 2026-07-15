"""FORGE v2 VC Demo Dashboard & HTML Report.

Generates investor-ready HTML reports with architecture diagrams,
benchmark charts, and compression metrics.
"""

from forge.demo.pipeline import run_demo
from forge.demo.runner import DemoRunner

__all__ = ["DemoRunner", "run_demo"]
