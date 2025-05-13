"""
SAT Solver package with unified interface.
"""

from .base import SolverBase, SolverResult, SolverStatus
from .config import SolverConfig, get_config, load_config
from .registry import SolverRegistry, register_solver

# Auto-discover and register all available solvers
try:
    SolverRegistry.auto_discover()
except ImportError:
    # This might happen during the initial import if there are circular dependencies
    pass

__all__ = [
    "SolverBase",
    "SolverResult",
    "SolverStatus",
    "SolverRegistry",
    "register_solver",
    "get_config",
    "load_config",
    "SolverConfig",
]
