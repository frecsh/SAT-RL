"""
Custom exceptions for the SAT+RL project.

This module defines exception classes specific to SAT solving and reinforcement learning,
allowing for more detailed error handling and reporting.
"""

from typing import Dict, List, Optional, Any


class SATBaseException(Exception):
    """Base class for all SAT+RL specific exceptions."""
    
    def __init__(self, message: str = None):
        """
        Initialize the exception.
        
        Args:
            message: Optional error message
        """
        self.message = message
        super().__init__(message)


class UnsatisfiableError(SATBaseException):
    """
    Exception raised when a SAT problem is proven to be unsatisfiable.
    
    This exception can include analysis of the unsatisfiable core.
    """
    
    def __init__(self, message: str = "Problem is unsatisfiable", 
                clause_analysis: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            clause_analysis: Optional analysis of the unsatisfiable core
        """
        self.clause_analysis = clause_analysis
        super().__init__(message)


class SolverTimeoutError(SATBaseException):
    """
    Exception raised when a SAT solver exceeds its time limit.
    
    This exception can include details about the partial solution found.
    """
    
    def __init__(self, message: str = "Solver exceeded time limit", 
                time_spent: float = None, partial_assignment: List[int] = None,
                satisfied_clauses: int = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            time_spent: Time spent before timeout (seconds)
            partial_assignment: Partial variable assignment at timeout
            satisfied_clauses: Number of clauses satisfied at timeout
        """
        self.time_spent = time_spent
        self.partial_assignment = partial_assignment
        self.satisfied_clauses = satisfied_clauses
        
        # Enhance the message with details if available
        if time_spent is not None or satisfied_clauses is not None:
            details = []
            if time_spent is not None:
                details.append(f"time_spent={time_spent:.2f}s")
            if satisfied_clauses is not None:
                details.append(f"satisfied_clauses={satisfied_clauses}")
            
            if details:
                message = f"{message} ({', '.join(details)})"
        
        super().__init__(message)


class InconsistentAssignmentError(SATBaseException):
    """
    Exception raised when an inconsistent variable assignment is detected.
    
    This occurs when a variable is assigned both True and False.
    """
    
    def __init__(self, message: str = "Inconsistent variable assignment detected", 
                variable: int = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            variable: The variable with inconsistent assignment
        """
        self.variable = variable
        
        # Enhance the message with the variable if available
        if variable is not None:
            message = f"{message} for variable {variable}"
        
        super().__init__(message)


class InvalidClauseError(SATBaseException):
    """
    Exception raised when an invalid clause is detected.
    
    This occurs when a clause has invalid literals or structure.
    """
    
    def __init__(self, message: str = "Invalid clause detected", clause: List[int] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            clause: The invalid clause
        """
        self.clause = clause
        
        # Enhance the message with the clause if available
        if clause is not None:
            message = f"{message}: {clause}"
        
        super().__init__(message)


class EnvironmentError(SATBaseException):
    """Exception raised when there's an error in the SAT environment."""
    
    def __init__(self, message: str = "Environment error occurred"):
        """
        Initialize the exception.
        
        Args:
            message: Error message
        """
        super().__init__(message)


class AgentError(SATBaseException):
    """Exception raised when there's an error in an agent."""
    
    def __init__(self, message: str = "Agent error occurred", agent_id: str = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            agent_id: Identifier of the agent that caused the error
        """
        self.agent_id = agent_id
        
        # Enhance the message with the agent ID if available
        if agent_id is not None:
            message = f"{message} in agent {agent_id}"
        
        super().__init__(message)