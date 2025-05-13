# SatRLGym: Reinforcement Learning for Boolean Satisfiability

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SatRLGym is a Gym-compatible environment for training reinforcement learning agents on boolean satisfiability (SAT) problems. It provides a standardized interface for experimenting with RL approaches to solving SAT instances, offering tools for verification, visualization, and performance benchmarking.

## Background

Boolean satisfiability (SAT) problems involve determining if there exists an assignment of boolean variables that makes a given formula evaluate to true. As the first problem proven to be NP-complete, SAT is fundamental to computational complexity theory and has applications in formal verification, planning, and circuit design.

Traditional SAT solvers use hand-crafted heuristics, but reinforcement learning offers an opportunity to learn effective strategies from experience. SatRLGym provides the tools needed to explore this intersection of classical computational problems and modern machine learning techniques.

## Key Features

- **Gymnasium-compatible Environment**: Standard RL interface for SAT problem-solving
- **DRAT Proof Verification**: Validate UNSAT proofs with rewards for correct proofs
- **Multiple Reward Functions**: Choose from sparse, dense, or learning-oriented rewards
- **Oracle Integration**: Use traditional SAT solvers as oracles for guidance
- **Visualization Tools**: Analyze agent behavior and problem structures
- **Flexible Storage Backends**: Support for various experience storage formats

## Installation

```bash
# Clone the repository
git clone https://github.com/frecsh/SatRLGym.git
cd SatRLGym

# Install locally in development mode
pip install -e .

# Install with optional dependencies
pip install -e ".[torch,solvers,proof]"
```

## Quick Start

```python
import gymnasium as gym
import satrlgym

# Create a SAT environment
env = gym.make("SatRLGym-v0", cnf_file="path/to/problem.cnf")
obs, info = env.reset(seed=42)

# Run a random agent
done = False
while not done:
    action = env.action_space.sample()  # Replace with your agent
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        print(f"Problem solved: {info['solved']}")
```

## Environment Representation

### Observation Space

The default environment provides observations as dictionaries containing:

- `variables`: Array representing current variable assignments (-1.0 for false, 1.0 for true, 0 for unassigned)
- `clauses`: Array indicating which clauses are currently satisfied
- `variable_assignment`: Dictionary mapping variable indices to boolean values
- `clause_satisfaction`: Boolean array indicating which clauses are satisfied

### Action Space

Actions are integer indices corresponding to variables to flip (0-indexed, with the environment converting to 1-indexed variables internally).

## Reward Functions

SatRLGym provides several reward function modes:

- **Sparse**: Reward only on problem solution (+1 for solving, 0 otherwise)
- **Dense**: Incremental rewards based on clause satisfaction changes
- **Learning**: Shaped rewards that balance exploration and exploitation

Example of selecting a reward mode:

```python
# Create environment with dense rewards
env = gym.make("SatRLGym-v0", cnf_file="problem.cnf", reward_mode="dense")
```

## Supported Environments

SatRLGym provides several environment configurations:

| Environment          | Description                                         |
| -------------------- | --------------------------------------------------- |
| `SatRLGym-v0`        | Core environment with variable flipping actions     |
| `SatRLGym-Guided-v0` | Environment with CDCL oracle guidance               |
| `SatRLGym-UNSAT-v0`  | Environment with rewards for UNSAT proof generation |

## Oracle Guidance

SatRLGym includes an oracle system that allows integration with traditional SAT solving heuristics:

- **What are Oracles?**: Components that provide expert guidance for variable selection and evaluation
- **Types of Oracles**:
  - Simple DPLL Oracle: Implements core DPLL algorithm heuristics
  - External Solver Oracle: Wraps external SAT solvers like MiniSAT

### Oracle Integration Example

```python
from satrlgym.oracles import SimpleDPLLOracle

# Create environment and oracle
env = gym.make("SatRLGym-v0", cnf_file="problem.cnf")
oracle = SimpleDPLLOracle(env.clauses, env.num_vars)

# Use oracle for guidance
obs, _ = env.reset()
query = {"assignment": obs["variable_assignment"]}
guidance = oracle.query(query)

# Select action based on oracle suggestion
if "suggested_assignments" in guidance:
    var_idx = list(guidance["suggested_assignments"].keys())[0]
    action = int(var_idx) - 1
else:
    action = env.action_space.sample()
```

## Implementing Custom Agents

Here's a simple example of implementing a greedy agent that prioritizes variables appearing in the most unsatisfied clauses:

```python
class GreedySATAgent:
    def __init__(self, env):
        self.env = env

    def choose_action(self, observation):
        # Count variable occurrences in unsatisfied clauses
        var_counts = defaultdict(int)
        clauses = observation["clauses"]

        for i, satisfied in enumerate(observation["clause_satisfaction"]):
            if not satisfied:
                for literal in self.env.clauses[i]:
                    var_counts[abs(literal)] += 1

        # Select variable with highest occurrence count
        if var_counts:
            best_var = max(var_counts.items(), key=lambda x: x[1])[0]
            # Convert to 0-indexed action
            return best_var - 1
        else:
            return self.env.action_space.sample()
```

## Proof Verification

SatRLGym includes DRAT proof verification for unsatisfiable instances:

```python
from satrlgym.proofs.verification import ProofVerificationManager

# Verify a DRAT proof
verifier = ProofVerificationManager()
cnf = "p cnf 1 2\n1 0\n-1 0\n"  # Simple UNSAT formula
proof = "d 1 0\n"  # DRAT proof
valid = verifier.verify_solution(cnf, proof)
```

## Environment Integration

Use proof verification within your environment for enhanced rewards:

```python
from satrlgym.proofs.verification import ProofVerificationManager

# In your environment wrapper
verifier = ProofVerificationManager()
if verifier.is_available():
    if verifier.verify_solution(cnf_formula, agent_proof):
        reward += 10  # Bonus reward for correct proof
    else:
        reward -= 5   # Penalty for invalid proof
```

## Visualization Tools

SatRLGym includes tools for visualizing agent behavior:

```python
from satrlgym.visualization import DataVisualizer

visualizer = DataVisualizer(experiment_path="path/to/experiment")
visualizer.plot_clause_satisfaction()
visualizer.plot_variable_assignments()
visualizer.plot_reward_curve()
```

## Using Standard Benchmarks

SatRLGym supports standard DIMACS CNF benchmark files:

```python
# Using a standard benchmark from SATLIB
env = gym.make("SatRLGym-v0", cnf_file="benchmarks/uf50-01.cnf")

# Or directly from DIMACS string
dimacs_string = """
p cnf 3 3
1 2 0
-1 3 0
-2 -3 0
"""
from satrlgym.utils import parse_dimacs
formula = parse_dimacs(dimacs_string)
env = gym.make("SatRLGym-v0", formula=formula)
```

Common benchmark sources:

- [SATLIB](https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html)
- [SAT Competition](http://www.satcompetition.org/)

## Performance Benchmarking

```bash
# Benchmark storage backends
python scripts/benchmark_cli.py run --backends npz hdf5 parquet

# Benchmark solver performance
python scripts/benchmark_solvers.py --time_limit 60
```

## Project Structure

```
src/satrlgym/
├── __init__.py               # Package initialization and registration
├── envs/                     # Environment implementations
│   ├── core.py               # Core environment classes
│   └── rewards.py            # Reward function implementations
├── oracles/                  # Oracle implementations for guidance
│   ├── base_oracle.py        # Abstract oracle base class
│   └── sat_oracle.py         # SAT solver oracle implementation
├── proofs/                   # Proof verification components
│   ├── drat.py               # DRAT proof checker implementation
│   └── verification.py       # Verification management utilities
├── utils/                    # Utility functions and tools
├── visualization/            # Visualization components
└── experience/               # Experience storage backends
```

## Troubleshooting

### Common Issues

- **ImportError: No module named 'drat-trim'**: The proof verification dependencies are missing. Install with `pip install -e ".[proof]"`.
- **Oracle returns empty guidance**: Ensure you're using the correct query format - some oracles expect OracleQuery objects rather than dictionaries.
- **Memory errors with large formulas**: Try using the memory-mapped storage backend for large problems.

For more help, check the full documentation or open an issue in the GitHub repository.

## Documentation

For detailed documentation, see the [`docs`](docs) directory:

- Installation Guide
- Environment API
- Oracle Integration
- Proof Verification

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

## Citation

If you use SatRLGym in your research, please cite:

```bibtex
@software{satrlgym2025,
  title = {SatRLGym: Reinforcement Learning for Boolean Satisfiability Problems},
  author = {SAT+RL Project Contributors},
  year = {2025},
  url = {https://github.com/frecsh/SatRLGym}
}
```

## License

MIT License
