"""
Custom exception classes for SAT solving operations.

This module defines specialized exceptions for various failure modes
in SAT solving to make error handling more precise and informative.
"""


class SATBaseException(Exception):
    """Base exception class for all SAT solver related exceptions."""
    pass


class UnsatisfiableError(SATBaseException):
    """
    Raised when a SAT problem is proven to be unsatisfiable.
    
    This is not strictly an error condition but a valid result.
    It's separated as an exception for control flow purposes.
    """
    def __init__(self, message="Problem is unsatisfiable", clause_analysis=None):
        self.clause_analysis = clause_analysis or {}
        self.message = message
        super().__init__(self.message)


class SolverTimeoutError(SATBaseException):
    """
    Raised when a solver exceeds its allocated time budget.
    
    Attributes:
        time_spent: Time spent before timeout in seconds
        partial_assignment: Any partial variable assignment found before timeout
        satisfied_clauses: Count or percentage of clauses satisfied in partial solution
    """
    def __init__(self, message="Solver exceeded time limit", time_spent=None, 
                 partial_assignment=None, satisfied_clauses=None):
        self.time_spent = time_spent
        self.partial_assignment = partial_assignment
        self.satisfied_clauses = satisfied_clauses
        self.message = message
        super().__init__(self.message)
        
    def __str__(self):
        details = []
        if self.time_spent is not None:
            details.append(f"time_spent={self.time_spent:.2f}s")
        if self.satisfied_clauses is not None:
            details.append(f"satisfied_clauses={self.satisfied_clauses}")
        
        detail_str = ", ".join(details)
        return f"{self.message} ({detail_str})" if details else self.message


class InconsistentAssignmentError(SATBaseException):
    """
    Raised when a solver detects that an assignment is inconsistent 
    (e.g., a variable is assigned both True and False).
    """
    def __init__(self, message="Inconsistent variable assignment detected", variable=None, 
                 assignment_history=None):
        self.variable = variable
        self.assignment_history = assignment_history
        self.message = message
        if variable is not None:
            self.message = f"{message} for variable {variable}"
        super().__init__(self.message)


class InvalidClauseError(SATBaseException):
    """
    Raised when an invalid clause is detected (e.g., empty clause or invalid literals).
    """
    def __init__(self, message="Invalid clause detected", clause=None):
        self.clause = clause
        self.message = message
        if clause is not None:
            self.message = f"{message}: {clause}"
        super().__init__(self.message)


class ConfigurationError(SATBaseException):
    """
    Raised when there's a problem with solver configuration.
    """
    pass