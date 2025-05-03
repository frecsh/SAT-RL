"""
SAT Solver package with unified interface.
"""

from .base import SolverBase, SolverResult, SolverStatus
from .registry import SolverRegistry, register_solver
from .config import get_config, load_config, SolverConfig

# Auto-discover and register all available solvers
try:
    SolverRegistry.auto_discover()
except ImportError as e:
    # This might happen during the initial import if there are circular dependencies
    pass

__all__ = [
    'SolverBase',
    'SolverResult',
    'SolverStatus',
    'SolverRegistry',
    'register_solver',
    'get_config',
    'load_config',
    'SolverConfig'
]