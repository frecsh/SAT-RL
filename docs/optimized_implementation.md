# SymbolicGym Optimized Implementation

This directory contains optimized implementations of the SAT environment classes for improved performance on large-scale SAT problems.

## Overview

The optimized SAT environment implementation focuses on the following performance improvements:

1. **Incremental Clause Satisfaction Updates**: Only recalculate the satisfaction state of clauses affected by a variable change, rather than recalculating all clauses.
2. **Efficient Data Structures**: Specialized data structures for quick lookup of clauses containing specific variables.
3. **Optimized Variable Tracking**: Efficient storage and access to variable assignments.
4. **Caching**: Avoids recreating observation arrays unnecessarily.

## Implementation Details

### Key Files

- `optimized_utils.py`: Contains optimized data structures and utility classes.
- `optimized_env.py`: Contains the optimized SAT environment implementation.

### Classes in `optimized_utils.py`

#### `ClauseIndexer`

Maps variables to clauses for efficient lookup:

```python
indexer = ClauseIndexer(clauses)
affected_clauses = indexer.get_affected_clauses(var_idx)
```

- **Purpose**: Quickly identify which clauses contain a specific variable.
- **Implementation**: Pre-computes maps from variables to clause indices during initialization.
- **Performance Impact**: Reduces lookup time from O(num_clauses \* avg_clause_length) to O(1).

#### `SatisfactionTracker`

Efficiently tracks and updates clause satisfaction state:

```python
tracker = SatisfactionTracker(clauses)
num_satisfied, newly_satisfied, newly_unsatisfied = tracker.update_assignment(var_idx, value)
```

- **Purpose**: Maintain which clauses are satisfied and update incrementally when variables change.
- **Implementation**: Tracks satisfied clauses and the variables that satisfy each clause.
- **Performance Impact**: Reduces satisfaction update time from O(num_clauses \* avg_clause_length) to O(num_affected_clauses).

#### `EfficientVariableMap`

Optimized storage and access to variable assignments:

```python
var_map = EfficientVariableMap(num_vars)
var_map.set(var_idx, value)
value = var_map.get(var_idx)
```

- **Purpose**: Provide efficient storage and access to variable assignments.
- **Implementation**: Stores only assigned variables to save memory.
- **Performance Impact**: Reduces memory usage and improves access time for large problems.

### `OptimizedSymbolicSatEnv` Class

Extends the base `SymbolicSatEnv` with optimizations:

- **Incremental Updates**: Updates only clauses affected by variable changes.
- **Observation Caching**: Avoids recreating observation arrays unnecessarily.
- **Compatible Interface**: Maintains compatibility with the original environment API.

## Performance Gains

The optimized implementation provides significant performance improvements, especially for large-scale SAT problems:

- **Step Time**: Improved performance for the `step` method, especially as the problem size increases.
- **Memory Usage**: More efficient memory usage for large problems.
- **Scalability**: Better scaling behavior for problems with hundreds or thousands of variables.

## Usage

To use the optimized environment, simply replace `SatGymEnv` with `OptimizedSymbolicSatEnv` in your code:

```python
from symbolicgym.domains.sat.env import OptimizedSymbolicSatEnv  # TODO: Update to actual import if available

# Create the environment
formula = ...  # Your SAT formula
env = OptimizedSymbolicSatEnv(formula=formula, reward_mode="sparse")

# Use the environment as usual
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

## Future Optimizations

Additional optimizations that could be implemented in the future:

1. **Parallelization**: Multi-threaded satisfaction checking for very large problems.
2. **More Sophisticated Caching**: More comprehensive caching strategies for frequently accessed data.
3. **Memory-Mapped Storage**: For extremely large problems that don't fit in memory.
4. **CUDA Acceleration**: GPU acceleration for large problems (requires significant changes).
