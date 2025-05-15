# SymbolicGym API Documentation

This document provides detailed information about the core components of SymbolicGym.

## Core Environment

### SymbolicSatEnv

The main environment class for SAT problems.

**Class**: `symbolicgym.domains.sat.env.SymbolicSatEnv`

#### Constructor Parameters

| Parameter   | Type        | Default  | Description                                             |
| ----------- | ----------- | -------- | ------------------------------------------------------- |
| formula     | dict        | required | SAT formula dict with 'clauses' and 'num_vars'          |
| reward_mode | str         | "sparse" | Type of reward function ('sparse', 'dense', 'learning') |
| max_steps   | int or None | None     | Maximum number of steps per episode                     |
| render_mode | str or None | None     | Rendering mode ('human', 'ansi', 'rgb_array')           |

#### Properties

| Property          | Type     | Description                        |
| ----------------- | -------- | ---------------------------------- |
| action_space      | Discrete | Space of valid actions             |
| observation_space | Dict     | Space of valid observations        |
| num_vars          | int      | Number of variables in the formula |
| num_clauses       | int      | Number of clauses in the formula   |

#### Methods

| Method               | Parameters              | Return Type                          | Description                                |
| -------------------- | ----------------------- | ------------------------------------ | ------------------------------------------ |
| reset                | seed=None, options=None | tuple(dict, dict)                    | Reset the environment to initial state     |
| step                 | action                  | tuple(dict, float, bool, bool, dict) | Execute one step in the environment        |
| render               | mode="human"            | Any                                  | Render the environment                     |
| compute_satisfaction | assignment              | tuple(list, float)                   | Compute clause satisfaction for assignment |
| get_valid_actions    | None                    | list[int]                            | Get list of valid actions                  |
| get_observation      | None                    | dict                                 | Get current observation                    |

#### Observation Space

The observation is a dictionary with the following components:

```python
{
    'variables': np.ndarray,  # Current variable assignments (-1, 0, 1)
    'clauses': np.ndarray,  # Boolean array of satisfied clauses
    'variable_assignment': dict,  # Dictionary mapping var indices to values
    'clause_satisfaction': list[bool]  # List of satisfied clauses
}
```

#### Action Space

Integer actions representing the variable to flip (0-indexed).

## Reward Functions

**Module**: `symbolicgym.domains.sat.rewards`

| Function        | Parameters                                                                       | Description                               |
| --------------- | -------------------------------------------------------------------------------- | ----------------------------------------- |
| sparse_reward   | satisfied_clauses, num_clauses, prev_satisfaction, solved, step_count, max_steps | Only rewards solving the problem          |
| dense_reward    | satisfied_clauses, num_clauses, prev_satisfaction, solved, step_count, max_steps | Rewards based on satisfaction improvement |
| learning_reward | satisfied_clauses, num_clauses, prev_satisfaction, solved, step_count, max_steps | Shaped rewards for learning               |

## Oracle Integration

### OracleBase

**Class**: `symbolicgym.domains.sat.oracles.base_oracle.OracleBase`

Abstract base class for SAT oracles.

#### Constructor Parameters

| Parameter     | Type            | Default  | Description              |
| ------------- | --------------- | -------- | ------------------------ |
| clauses       | list[list[int]] | required | List of clauses          |
| num_vars      | int             | required | Number of variables      |
| oracle_config | dict or None    | None     | Configuration parameters |

#### Methods

| Method | Parameters               | Return Type    | Description          |
| ------ | ------------------------ | -------------- | -------------------- |
| query  | query_or_type, data=None | OracleResponse | Process oracle query |

### SimpleDPLLOracle

**Class**: `symbolicgym.domains.sat.oracles.simple_oracle.SimpleDPLLOracle`

Oracle based on DPLL algorithm.

#### Constructor Parameters

| Parameter     | Type            | Default  | Description              |
| ------------- | --------------- | -------- | ------------------------ |
| clauses       | list[list[int]] | required | List of clauses          |
| num_vars      | int             | required | Number of variables      |
| oracle_config | dict or None    | None     | Configuration parameters |

#### Methods

| Method           | Parameters                                | Return Type    | Description                    |
| ---------------- | ----------------------------------------- | -------------- | ------------------------------ |
| query            | query_or_type, data=None                  | OracleResponse | Process oracle query           |
| get_unit_clause  | clauses, assignment                       | int or None    | Find unit clause               |
| get_pure_literal | clauses, assignment, variable_occurrences | int or None    | Find pure literal              |
| choose_variable  | clauses, assignment, heuristic="VSIDS"    | int            | Choose variable with heuristic |

## Utility Functions

### CNF Utilities

**Module**: `symbolicgym.domains.sat.utils`

| Function                  | Parameters          | Return Type        | Description                               |
| ------------------------- | ------------------- | ------------------ | ----------------------------------------- |
| load_cnf_file             | filename            | dict               | Load CNF file in DIMACS format            |
| parse_dimacs              | dimacs_string       | dict               | Parse DIMACS format string                |
| compute_satisfied_clauses | clauses, assignment | tuple(list, float) | Compute clause satisfaction               |
| is_satisfiable            | clauses, assignment | bool               | Check if assignment satisfies all clauses |

## Experience Storage

### ExperienceStorage

**Class**: `symbolicgym.domains.sat.storage.base_storage.ExperienceStorage`

Abstract base class for storing agent experiences.

#### Methods

| Method           | Parameters                                   | Return Type | Description             |
| ---------------- | -------------------------------------------- | ----------- | ----------------------- |
| store_transition | state, action, reward, next_state, done      | None        | Store transition        |
| store_episode    | states, actions, rewards, next_states, dones | None        | Store complete episode  |
| load             | identifier                                   | Any         | Load stored experiences |
| save             | identifier                                   | None        | Save stored experiences |

### Specific Storage Implementations

| Class         | Description                          |
| ------------- | ------------------------------------ |
| NPZStorage    | NumPy compressed storage             |
| HDF5Storage   | HDF5 file storage for large datasets |
| SQLiteStorage | SQL database storage                 |

## Visualization

### DataVisualizer

**Class**: `symbolicgym.domains.sat.visualization.data_viz.DataVisualizer`

Tools for visualizing agent behavior and statistics.

#### Constructor Parameters

| Parameter       | Type         | Default  | Description                 |
| --------------- | ------------ | -------- | --------------------------- |
| experiment_path | str          | required | Path to experiment data     |
| config          | dict or None | None     | Visualization configuration |

#### Methods

| Method                    | Parameters             | Return Type     | Description                         |
| ------------------------- | ---------------------- | --------------- | ----------------------------------- |
| plot_clause_satisfaction  | episode=None, ax=None  | matplotlib.Axes | Plot clause satisfaction over time  |
| plot_variable_assignments | episode=None, ax=None  | matplotlib.Axes | Plot variable assignments over time |
| plot_reward_curve         | smoothing=0.0, ax=None | matplotlib.Axes | Plot reward curve                   |
| create_dashboard          | save_path=None         | None            | Create comprehensive dashboard      |
