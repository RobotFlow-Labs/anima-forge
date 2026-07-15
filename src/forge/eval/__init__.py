"""PRD-32: VLA Evaluation Harness — Native FORGE Integration.

Integrates Allen AI's vla-evaluation-harness for task-success-rate benchmarking
of FORGE student models. Eval is invoked via CLI (`forge eval`), not auto-run.

Usage:
    from forge.eval.model_server import ForgeModelServer
    from forge.eval.runner import EvalRunner
    from forge.eval.results import EvalResult, load_results
"""

from forge.eval.results import EvalResult

__all__ = ["EvalResult"]
