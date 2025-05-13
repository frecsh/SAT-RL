"""
Adapter for legacy SAT solvers to make them compatible with the new interface.
This allows existing solver implementations to be used with the new unified architecture.
"""

import inspect
import logging
import time
from typing import Any

from .base import SolverBase, SolverResult, SolverStatus

# Set up logging
logger = logging.getLogger(__name__)


class LegacySolverAdapter(SolverBase):
    """
    Adapter for legacy SAT solvers to make them compatible with the SolverBase interface.
    This class wraps existing solver classes or functions to adapt them to the new interface.
    """

    def __init__(
        self,
        legacy_solver: Any,
        num_vars: int = 0,
        name: str = "legacy",
        **solver_kwargs,
    ):
        """
        Initialize the adapter with a legacy solver.

        Args:
            legacy_solver: The legacy solver to adapt (class or instance)
            num_vars: Number of variables in the problem
            name: Name of the legacy solver
            **solver_kwargs: Additional arguments to pass to the legacy solver constructor
        """
        self.name = name
        self.num_vars = num_vars
        self.clauses = []
        self.result = None
        self.statistics = {}
        self._initialize_legacy_solver(legacy_solver, **solver_kwargs)

    def _initialize_legacy_solver(self, legacy_solver: Any, **kwargs) -> None:
        """
        Initialize the legacy solver.

        Args:
            legacy_solver: The legacy solver to adapt (class or instance)
            **kwargs: Additional arguments to pass to the legacy solver constructor
        """
        # Determine if legacy_solver is a class or an instance
        if inspect.isclass(legacy_solver):
            # It's a class, so instantiate it
            try:
                self.legacy_solver = legacy_solver(num_vars=self.num_vars, **kwargs)
            except TypeError:
                # Try with different parameter names
                try:
                    self.legacy_solver = legacy_solver(n_vars=self.num_vars, **kwargs)
                except TypeError:
                    # Last resort, just pass all kwargs
                    self.legacy_solver = legacy_solver(**kwargs)
        else:
            # It's already an instance
            self.legacy_solver = legacy_solver

        # Detect available methods on the legacy solver
        self._has_add_clause = hasattr(self.legacy_solver, "add_clause")
        self._has_add_clauses = hasattr(self.legacy_solver, "add_clauses")
        self._has_solve = hasattr(self.legacy_solver, "solve")
        self._has_get_model = hasattr(self.legacy_solver, "get_model")
        self._has_get_statistics = hasattr(self.legacy_solver, "get_statistics")
        self._has_interrupt = hasattr(self.legacy_solver, "interrupt")
        self._has_configure = hasattr(self.legacy_solver, "configure")

    def add_clause(self, clause: list[int]) -> None:
        """
        Add a single clause to the solver.

        Args:
            clause: A list of integers representing literals in the clause.
        """
        self.clauses.append(clause)

        if self._has_add_clause:
            try:
                self.legacy_solver.add_clause(clause)
            except Exception as e:
                logger.error(f"Error adding clause to legacy solver: {e}")
                # Store the clause locally if adding to the legacy solver fails

    def add_clauses(self, clauses: list[list[int]]) -> None:
        """
        Add multiple clauses to the solver.

        Args:
            clauses: A list of clauses, where each clause is a list of integers.
        """
        self.clauses.extend(clauses)

        if self._has_add_clauses:
            try:
                self.legacy_solver.add_clauses(clauses)
            except Exception as e:
                logger.error(f"Error adding clauses to legacy solver: {e}")
                # Use add_clause as fallback
                if self._has_add_clause:
                    for clause in clauses:
                        try:
                            self.legacy_solver.add_clause(clause)
                        except Exception:
                            pass

    def solve(
        self, assumptions: list[int] | None = None, timeout: float | None = None
    ) -> SolverResult:
        """
        Attempt to solve the SAT instance.

        Args:
            assumptions: Optional list of literal assumptions
            timeout: Optional timeout in seconds

        Returns:
            SolverResult containing the solution status and other information
        """
        start_time = time.time()

        # Default result in case of error
        result = SolverResult(
            status=SolverStatus.ERROR,
            total_clauses=len(self.clauses),
            runtime=0.0,
            error_message="Solver method not available",
        )

        try:
            if self._has_solve:
                # Try to call the solve method with the appropriate parameters
                # First, inspect what parameters the legacy solver's solve method accepts
                solve_sig = inspect.signature(self.legacy_solver.solve)
                solve_params = solve_sig.parameters

                solve_kwargs = {}
                if "assumptions" in solve_params and assumptions is not None:
                    solve_kwargs["assumptions"] = assumptions
                if "timeout" in solve_params and timeout is not None:
                    solve_kwargs["timeout"] = timeout

                # Call the solve method
                legacy_result = self.legacy_solver.solve(**solve_kwargs)

                # Check if legacy_result is a boolean or a more complex result object
                if isinstance(legacy_result, bool):
                    is_sat = legacy_result
                    status = (
                        SolverStatus.SATISFIABLE
                        if is_sat
                        else SolverStatus.UNSATISFIABLE
                    )

                    # Try to get the model if available
                    solution = None
                    if is_sat and self._has_get_model:
                        solution = self.legacy_solver.get_model()

                    # Create a result object
                    result = SolverResult(
                        status=status,
                        solution=solution,
                        runtime=time.time() - start_time,
                        total_clauses=len(self.clauses),
                        satisfied_clauses=len(self.clauses) if is_sat else 0,
                    )
                else:
                    # Try to extract information from the complex result object
                    # This requires knowledge of the legacy solver's result format
                    try:
                        if hasattr(legacy_result, "is_sat") or hasattr(
                            legacy_result, "satisfiable"
                        ):
                            is_sat = getattr(
                                legacy_result,
                                "is_sat",
                                getattr(legacy_result, "satisfiable", False),
                            )
                            status = (
                                SolverStatus.SATISFIABLE
                                if is_sat
                                else SolverStatus.UNSATISFIABLE
                            )
                        else:
                            status = SolverStatus.UNKNOWN

                        solution = getattr(
                            legacy_result,
                            "model",
                            getattr(legacy_result, "solution", None),
                        )
                        runtime = getattr(
                            legacy_result, "runtime", time.time() - start_time
                        )
                        stats = getattr(legacy_result, "statistics", {})

                        result = SolverResult(
                            status=status,
                            solution=solution,
                            runtime=runtime,
                            total_clauses=len(self.clauses),
                            satisfied_clauses=len(self.clauses) if is_sat else 0,
                            statistics=stats,
                        )
                    except Exception as e:
                        logger.error(f"Error parsing legacy solver result: {e}")
                        result = SolverResult(
                            status=SolverStatus.ERROR,
                            runtime=time.time() - start_time,
                            error_message=f"Error parsing legacy result: {e}",
                        )
            else:
                # The legacy solver doesn't have a solve method
                # Try to use other methods that might be available
                method_found = False

                # Try alternative methods that might be available
                for method_name in ["run", "start", "execute", "search"]:
                    if hasattr(self.legacy_solver, method_name):
                        method = getattr(self.legacy_solver, method_name)
                        if callable(method):
                            try:
                                method()
                                # Now we need to interpret this result...
                                # This is highly dependent on the legacy solver's API
                                result = SolverResult(
                                    status=SolverStatus.UNKNOWN,
                                    runtime=time.time() - start_time,
                                    total_clauses=len(self.clauses),
                                )
                                method_found = True
                                break
                            except Exception as e:
                                logger.error(
                                    f"Error calling legacy solver method '{method_name}': {e}"
                                )

                if not method_found:
                    result = SolverResult(
                        status=SolverStatus.ERROR,
                        runtime=time.time() - start_time,
                        error_message="No solving method found in legacy solver",
                    )

        except Exception as e:
            result = SolverResult(
                status=SolverStatus.ERROR,
                runtime=time.time() - start_time,
                error_message=str(e),
            )

        # Store the result for later access
        self.result = result

        # Store some statistics
        self.statistics["runtime"] = result.runtime
        self.statistics["status"] = result.status.value

        return result

    def get_model(self) -> list[int] | None:
        """
        Get the satisfying assignment if one exists.

        Returns:
            List of literals representing the satisfying assignment,
            or None if problem is unsatisfiable
        """
        # First, check if we have a stored result
        if self.result and self.result.solution:
            return self.result.solution

        # Otherwise, try to get it from the legacy solver
        if self._has_get_model:
            try:
                return self.legacy_solver.get_model()
            except Exception as e:
                logger.error(f"Error getting model from legacy solver: {e}")

        # Check alternative method names
        for method_name in ["get_solution", "get_assignment", "solution", "model"]:
            if hasattr(self.legacy_solver, method_name):
                attr = getattr(self.legacy_solver, method_name)
                if callable(attr):
                    try:
                        return attr()
                    except Exception:
                        pass
                else:
                    return attr

        return None

    def get_statistics(self) -> dict[str, Any]:
        """
        Get solver statistics.

        Returns:
            Dictionary of statistics
        """
        # First, check if the legacy solver has a get_statistics method
        if self._has_get_statistics:
            try:
                legacy_stats = self.legacy_solver.get_statistics()
                if isinstance(legacy_stats, dict):
                    # Merge with our own statistics
                    return {**legacy_stats, **self.statistics}
            except Exception as e:
                logger.error(f"Error getting statistics from legacy solver: {e}")

        # Check alternative method names
        for method_name in ["statistics", "get_stats", "stats"]:
            if hasattr(self.legacy_solver, method_name):
                attr = getattr(self.legacy_solver, method_name)
                if callable(attr):
                    try:
                        stats = attr()
                        if isinstance(stats, dict):
                            return {**stats, **self.statistics}
                    except Exception:
                        pass
                elif isinstance(attr, dict):
                    return {**attr, **self.statistics}

        # Return our own statistics if nothing else is available
        return self.statistics

    def interrupt(self) -> None:
        """
        Interrupt the solving process.
        """
        if self._has_interrupt:
            try:
                self.legacy_solver.interrupt()
            except Exception as e:
                logger.error(f"Error interrupting legacy solver: {e}")

        # Check alternative method names
        for method_name in ["stop", "terminate", "cancel"]:
            if hasattr(self.legacy_solver, method_name):
                try:
                    method = getattr(self.legacy_solver, method_name)
                    if callable(method):
                        method()
                        return
                except Exception:
                    pass

    def configure(self, config: dict[str, Any]) -> None:
        """
        Configure the solver with the given parameters.

        Args:
            config: Dictionary of configuration parameters
        """
        if self._has_configure:
            try:
                self.legacy_solver.configure(config)
            except Exception as e:
                logger.error(f"Error configuring legacy solver: {e}")

        # Check alternative method names
        for method_name in ["set_parameters", "set_config", "config", "set_options"]:
            if hasattr(self.legacy_solver, method_name):
                try:
                    method = getattr(self.legacy_solver, method_name)
                    if callable(method):
                        method(config)
                        return
                except Exception:
                    pass

        # Try to set attributes directly
        for key, value in config.items():
            try:
                setattr(self.legacy_solver, key, value)
            except Exception:
                pass
