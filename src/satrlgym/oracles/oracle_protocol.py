"""
Oracle protocol for communication between environments and oracles.

This module defines the standard formats for queries to oracles and their responses.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class QueryType(Enum):
    """Types of queries that can be sent to oracles."""

    NEXT_VARIABLE = "next_variable"
    EVALUATE_SOLUTION = "evaluate_solution"
    SOLVE = "solve"
    UNIT_PROPAGATION = "unit_propagation"
    ANALYZE_CONFLICT = "analyze_conflict"
    VARIABLE_ASSIGNMENT = "variable_assignment"  # Used in tests
    CLAUSE_SATISFACTION = "clause_satisfaction"
    PARTIAL_ASSIGNMENT = "partial_assignment"
    FORMULA_SATISFIABILITY = "formula_satisfiability"
    SOLUTION_COUNT = "solution_count"
    BACKBONE_VARIABLES = "backbone_variables"
    MARGINALS = "marginals"


class OracleProtocol(ABC):
    """
    Abstract protocol for oracle interactions.

    This defines the standard interface for communication between
    environments and SAT oracles.
    """

    @abstractmethod
    def query(self, query_type: QueryType, data: dict[str, Any]) -> dict[str, Any]:
        """
        Send a query to the oracle.

        Args:
            query_type: Type of the query
            data: Query data

        Returns:
            Response data
        """


class OracleQuery:
    """
    A standardized query to an oracle.

    Attributes:
        query_type: Type of query (e.g., "next_variable", "evaluate_solution")
        current_assignment: Current variable assignments
        options: Additional options for the query
    """

    def __init__(
        self,
        query_type: str,
        current_assignment: dict[int, bool],
        options: dict[str, Any] | None = None,
    ):
        """
        Initialize an oracle query.

        Args:
            query_type: Type of query (e.g., "next_variable", "evaluate_solution")
            current_assignment: Current variable assignments
            options: Additional options for the query
        """
        self.query_type = query_type
        self.current_assignment = current_assignment
        self.options = options or {}

    def validate(self) -> bool:
        """
        Validate that the query is well-formed.

        Returns:
            True if the query is valid, otherwise raises ValueError

        Raises:
            ValueError: If the query is invalid
        """
        if not self.query_type:
            raise ValueError("Query type is required")

        if not isinstance(self.current_assignment, dict):
            raise ValueError("Current assignment must be a dictionary")

        return True

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the query to a dictionary for serialization.

        Returns:
            Dictionary representation of the query
        """
        return {
            "query_type": self.query_type,
            "current_assignment": self.current_assignment,
            "options": self.options,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OracleQuery":
        """
        Create a query from a dictionary.

        Args:
            data: Dictionary representation of the query

        Returns:
            An OracleQuery instance
        """
        return cls(
            query_type=data["query_type"],
            current_assignment=data["current_assignment"],
            options=data.get("options", {}),
        )


class OracleResponse:
    """
    A standardized response from an oracle.

    Attributes:
        query_type: Type of query that was processed
        success: Whether the query was successful
        result: The result of the query (if successful)
        metrics: Performance metrics or other metadata
        message: Optional message (for backwards compatibility)
        metadata: Optional metadata (for backwards compatibility)
    """

    def __init__(
        self,
        query_type: str,
        success: bool,
        result: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize an oracle response.

        Args:
            query_type: Type of query that was processed
            success: Whether the query was successful
            result: The result of the query (if successful)
            metrics: Performance metrics or other metadata
            message: Optional message (for backwards compatibility)
            metadata: Optional metadata (for backwards compatibility)
        """
        self.query_type = query_type
        self.success = success
        self.result = result
        self.metrics = metrics or {}
        self.message = message
        self.metadata = metadata or {}

    def validate(self) -> bool:
        """
        Validate that the response is well-formed.

        Returns:
            True if the response is valid, otherwise raises ValueError

        Raises:
            ValueError: If the response is invalid
        """
        if self.success is None:
            raise ValueError("Success flag is required")

        if self.success and self.result is None:
            raise ValueError("Result is required when success is True")

        return True

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the response to a dictionary for serialization.

        Returns:
            Dictionary representation of the response
        """
        return {
            "query_type": self.query_type,
            "success": self.success,
            "result": self.result,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OracleResponse":
        """
        Create a response from a dictionary.

        Args:
            data: Dictionary representation of the response

        Returns:
            An OracleResponse instance
        """
        return cls(
            query_type=data["query_type"],
            success=data["success"],
            result=data.get("result"),
            metrics=data.get("metrics", {}),
        )

    def __getitem__(self, key):
        """
        Allow dictionary-like access to response attributes.

        This enables backwards compatibility with code that expects a dict.

        Args:
            key: Attribute name

        Returns:
            The attribute value

        Raises:
            KeyError: If the attribute doesn't exist
        """
        if key == "suggested_assignments" and self.result and "variable" in self.result:
            # For backwards compatibility with tests that expect suggested_assignments
            return {self.result["variable"]: self.result["value"]}
        elif (
            key == "satisfied_clauses"
            and self.result
            and "num_satisfied" in self.result
        ):
            # For tests expecting satisfied_clauses
            return self.result.get("satisfied_indices", [])
        elif self.result and key in self.result:
            return self.result[key]
        elif hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(key)

    def __contains__(self, key):
        """
        Check if key exists in the response.

        Args:
            key: Key to check

        Returns:
            True if the key exists, False otherwise
        """
        if (
            key == "satisfied_clauses"
            and self.result
            and "num_satisfied" in self.result
        ):
            return True
        elif self.result and key in self.result:
            return True
        elif (
            key == "suggested_assignments" and self.result and "variable" in self.result
        ):
            return True
        elif hasattr(self, key):
            return True
        return False

    def __iter__(self):
        """Support iteration like a dictionary"""
        if self.result:
            yield from self.result

    def __len__(self):
        """Support len() like a dictionary"""
        return len(self.result) if self.result else 0

    def __bool__(self):
        """Boolean evaluation (always True to avoid errors)"""
        return True

    def keys(self):
        """Return the keys like a dictionary"""
        return self.result.keys() if self.result else []

    def values(self):
        """Return the values like a dictionary"""
        return self.result.values() if self.result else []

    def items(self):
        """Return the items like a dictionary"""
        return self.result.items() if self.result else []

    def get(self, key, default=None):
        """Get a value with a default like a dictionary"""
        try:
            return self[key]
        except KeyError:
            return default
