#!/usr/bin/env python3
"""
SAT RL Logger - Comprehensive logging utility for SAT RL agents

This module provides a specialized logger for tracking the behavior of
reinforcement learning agents solving SAT problems, including:
- State transitions
- Action selection
- Rewards
- Performance metrics
- Training progress
"""

import os
import time
import json
import csv
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
import functools
import matplotlib.pyplot as plt


def wrap_agent_step(func):
    """
    Decorator for agent step functions to log their activity.
    
    This decorator should be applied to the step method of RL agents
    to automatically log state transitions, actions, and rewards.
    
    Args:
        func: The step function to wrap
        
    Returns:
        Wrapped function that logs activity
    """
    def wrapper(self, state, *args, **kwargs):
        # Call the original step function
        action, next_state, reward, done, info = func(self, state, *args, **kwargs)
        
        # Log the step if logging is enabled
        if hasattr(self, 'enable_logging') and self.enable_logging and hasattr(self, 'logger'):
            self.logger.log_step(state, action, reward, next_state, done, info)
            
        return action, next_state, reward, done, info
    return wrapper


class SATRLLogger:
    """
    Logger for SAT reinforcement learning experiments
    """
    
    def __init__(self, logs_dir=None, log_to_file=True, max_entries=10000):
        """
        Initialize the logger.
        
        Args:
            logs_dir: Directory to save logs to
            log_to_file: Whether to log to a file
            max_entries: Maximum number of trace entries to keep in memory
        """
        self.log_to_file = log_to_file
        self.logs_dir = logs_dir
        
        # Create logs directory if needed
        if log_to_file and logs_dir:
            os.makedirs(logs_dir, exist_ok=True)
        
        # Initialize trace storage
        # Ensure max_entries is an integer
        try:
            self.max_entries = int(max_entries)
        except (ValueError, TypeError):
            print(f"Warning: Invalid max_entries value '{max_entries}', using default of 10000")
            self.max_entries = 10000
            
        self.traces = deque(maxlen=self.max_entries)