"""
Registry for SAT solvers in the framework.
Implements a simple registry pattern for dynamically registering and accessing solvers.
"""

import importlib
import inspect
import logging
import os
import pkgutil
from collections.abc import Callable

from .base import SolverBase

# Set up logging
logger = logging.getLogger(__name__)


class SolverRegistry:
    """
    Registry for SAT solvers.
    Enables registering solvers by name and retrieving them later.
    """

    _registry: dict[str, type[SolverBase]] = {}
    _default_solver: str | None = None

    @classmethod
    def register(cls, name: str, solver_cls: type[SolverBase]) -> None:
        """
        Register a solver with the given name.

        Args:
            name: Name of the solver
            solver_cls: Solver class (must inherit from SolverBase)
        """
        if not issubclass(solver_cls, SolverBase):
            raise TypeError(
                f"Solver class {solver_cls.__name__} must inherit from SolverBase"
            )

        if name in cls._registry:
            logger.warning(f"Overriding existing solver registration for '{name}'")

        cls._registry[name] = solver_cls

        # If this is the first solver registered, make it the default
        if cls._default_solver is None:
            cls._default_solver = name

    @classmethod
    def register_as(cls, name: str) -> Callable[[type[SolverBase]], type[SolverBase]]:
        """
        Decorator to register a solver with the given name.

        Args:
            name: Name of the solver

        Returns:
            Decorator function that registers the solver
        """

        def decorator(solver_cls: type[SolverBase]) -> type[SolverBase]:
            cls.register(name, solver_cls)
            return solver_cls

        return decorator

    @classmethod
    def set_default(cls, name: str) -> None:
        """
        Set the default solver.

        Args:
            name: Name of the solver to use as default
        """
        if name not in cls._registry:
            raise ValueError(f"No solver registered with name '{name}'")
        cls._default_solver = name

    @classmethod
    def get(cls, name: str | None = None) -> type[SolverBase]:
        """
        Get a solver by name.

        Args:
            name: Name of the solver, or None to get the default solver

        Returns:
            Solver class
        """
        if name is None:
            if cls._default_solver is None:
                raise ValueError("No default solver set")
            return cls._registry[cls._default_solver]

        if name not in cls._registry:
            raise ValueError(f"No solver registered with name '{name}'")

        return cls._registry[name]

    @classmethod
    def list_solvers(cls) -> list[str]:
        """
        List all registered solvers.

        Returns:
            List of solver names
        """
        return list(cls._registry.keys())

    @classmethod
    def create(cls, name: str | None = None, **kwargs) -> SolverBase:
        """
        Create a new instance of the specified solver.

        Args:
            name: Name of the solver, or None to use the default solver
            **kwargs: Arguments to pass to the solver constructor

        Returns:
            Instance of the solver
        """
        solver_cls = cls.get(name)
        return solver_cls(**kwargs)

    @classmethod
    def auto_discover(cls) -> None:
        """
        Auto-discover and register all solvers in the solvers package.
        Looks for classes that inherit from SolverBase and automatically registers them.
        """
        import solvers  # Import the package containing solvers

        # Get the package directory
        package_dir = os.path.dirname(solvers.__file__)

        # Iterate over all modules in the package
        for _, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
            if is_pkg or module_name == "base" or module_name == "registry":
                continue  # Skip subpackages and the base/registry modules

            # Import the module
            module = importlib.import_module(f"solvers.{module_name}")

            # Find all solver classes in the module
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, SolverBase)
                    and obj != SolverBase
                    and not inspect.isabstract(obj)
                ):
                    # Register the solver with its class name or a provided name
                    solver_name = getattr(obj, "solver_name", name.lower())
                    cls.register(solver_name, obj)
                    logger.debug(f"Auto-discovered solver: {solver_name}")


# Register common decorator for more concise solver registration
register_solver = SolverRegistry.register_as
