# SymbolicGym: A Unified RL Platform for Symbolic Reasoning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/frecsh/SymbolicGym/actions/workflows/ci.yml/badge.svg)](https://github.com/frecsh/SymbolicGym/actions/workflows/ci.yml)
[![codecov](https://åo/gh/frecsh/SymbolicGym/branch/main/graph/badge.svg)](https://codecov.io/gh/frecsh/SymbolicGym)

SymbolicGym is a general-purpose, extensible Gym-compatible platform for reinforcement learning on symbolic reasoning tasks. It supports SAT, SymPy, and Z3 domains, with rich latent-space feedback, cross-domain learning, interpretability tooling, and robust risk mitigation. All major roadmap features are implemented, tested, and verified.

## Architecture Overview

![SymbolicGym Architecture](docs/architecture_diagram.svg)

The diagram above illustrates how the core environment, symbolic domains (SAT, SymPy, Z3), domain registry, feedback and representation layers, agents (DQN, PPO, GNN, CTDE, GRPO, MoE, Imitation), and interpretability/visualization tools interact within SymbolicGym.

## Key Contributions

- **Unified RL platform for SAT, SymPy, and Z3 symbolic domains**
- **Multi-dimensional latent feedback and graph-based state representations**
- **Cross-domain curriculum learning and shared encoder for generalization**
- **Interpretability dashboard, latent projector, and feedback interpreter**
- **Risk mitigation: oracle imitation, VecEnv, distributed runner, external storage**
- **Extensible domain registry and integration template for new domains**

## Research Goals

- Investigate generalization and transfer learning across symbolic domains
- Develop interpretable RL agents for symbolic reasoning
- Reduce sample complexity and improve robustness via curriculum and oracles
- Enable scalable, reproducible experiments in symbolic RL

## Related Publications

- See `docs/research_questions.md` for open research questions.

## Background

Symbolic reasoning tasks—such as SAT solving, symbolic algebra, and SMT solving—are foundational in computer science, mathematics, and AI. SymbolicGym enables RL agents to interact with and learn from these domains using both scalar and high-dimensional feedback, supporting research in generalization, transfer, interpretability, and robust evaluation.

## Supported Domains

- **SAT**: Boolean satisfiability (phase transition, industrial, crafted problems)
- **SymPy**: Symbolic algebra (simplification, factoring, equation solving)
- **Z3**: Satisfiability Modulo Theories (SMT) tasks

## Features

- **Pluggable symbolic-feedback backends** for easy domain extension (see `docs/domain_integration.md`)
- **Latent-space feedback**: Multi-dimensional signals for richer learning
- **Graph and matrix state representations** for advanced agents
- **Multi-agent and CTDE support**: Centralized training, decentralized execution
- **Curriculum learning**: Domain-agnostic and per-domain curricula
- **Oracle and proof integration**: Use traditional solvers and proof checkers as oracles
- **Visualization and interpretability tools**: Clause attention, feedback vector analysis, dashboard, latent projector
- **Risk mitigation**: Oracle imitation pre-training, shared encoder, VecEnv, distributed runner, external storage
- **Extensible domain registry**: Add new domains with minimal boilerplate

## Installation

```bash
# Clone the repository
git clone https://github.com/frecsh/SymbolicGym.git
cd SymbolicGym

# Install locally in development mode
pip install -e .

# Install with optional dependencies
pip install -e ".[torch,solvers,proof]"
```

## Quick Start

```python
import gymnasium as gym
import symbolicgym

# Create a SAT environment
# You can specify either a CNF file (using 'cnf_file'), or directly provide a formula dict (using 'formula').
# 'cnf_file' is a convenience option that loads and parses the file into a formula dict internally.
env = gym.make("SymbolicGym-v0", cnf_file="path/to/problem.cnf")
# Equivalent usage with a formula dict:
# formula = {"clauses": [[1, 2], [-1, -2], [1, -2], [-1, 2]], "num_vars": 2}
# env = gym.make("SymbolicGym-v0", formula=formula)
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

For more detailed agent and usage examples, see the 'examples/' directory in the repository. This includes DQN agents, custom agent implementations, environment enhancements, and interpretability tools.

For additional code samples and advanced usage, refer to the 'examples/' directory and the 'notebooks/' folder for interactive guides.

## Results and Benchmarks

See the `results/` directory for sample learning curves, cross-domain transfer plots, ablation tables, and a summary of key findings. Performance benchmarking scripts and profiling tools are provided for large-scale and distributed runs.

## Reproducibility

- Ready-to-run experiment scripts: see `scripts/train_sat_dqn.py` and `scripts/train_crossdomain_gnn.py`.
- All experiments are seedable and config-driven for reproducibility.
- See the `tests/` directory for expanded test coverage, including cross-domain and performance tests.

## Quickstart Notebook

A quickstart notebook is available in `notebooks/Quickstart.ipynb` for new users. It demonstrates installation, environment setup, running agents, and visualization tools for SAT, SymPy, and Z3 domains.

### Using with Problem Generators

You can also generate random SAT problems for training:

```python
from symbolicgym.utils.generators import generate_random_ksat

# Generate a random 3-SAT formula with 20 variables and 85 clauses
formula = generate_random_ksat(n_vars=20, n_clauses=85, k=3, seed=42)
env = gym.make("SymbolicGym-v0", formula=formula)

# Or generate a formula with a specific clause-to-variable ratio
formula = generate_random_ksat(n_vars=100, clause_ratio=4.2, k=3, seed=42)
```

## Environment Representation

### Observation Space

The default environment provides observations as dictionaries containing:

- `variables`: Array representing current variable assignments (-1, 0, 1; -1 for false, 1 for true, 0 for unassigned)
- `clauses`: Array indicating which clauses are currently satisfied
- `variable_assignment`: Dictionary mapping variable indices to boolean values
- `clause_satisfaction`: Boolean array indicating which clauses are satisfied

### Action Space

Actions are integer indices corresponding to variables to flip (0-indexed, with the environment converting to 1-indexed variables internally).

## Reward Functions

SymbolicGym provides several reward function modes:

- **Sparse**: Reward only on problem solution (+1 for solving, 0 otherwise)
- **Dense**: Incremental rewards based on clause satisfaction changes
- **Learning**: Shaped rewards that balance exploration and exploitation

Example of selecting a reward mode:

```python
# Create environment with dense rewards
env = gym.make("SymbolicGym-v0", cnf_file="problem.cnf", reward_mode="dense")
```

## Supported Environments

SymbolicGym provides several environment configurations:

| Environment             | Description                                                                                                |
| ----------------------- | ---------------------------------------------------------------------------------------------------------- |
| `SymbolicGym-v0`        | Core environment with variable flipping actions                                                            |
| `SymbolicGym-Guided-v0` | Environment with CDCL oracle guidance _(experimental, may not be available in all releases)_               |
| `SymbolicGym-UNSAT-v0`  | Environment with rewards for UNSAT proof generation _(experimental, may not be available in all releases)_ |

## Oracle Guidance

SymbolicGym includes an oracle system that allows integration with traditional SAT solving heuristics:

- **What are Oracles?**: Components that provide expert guidance for variable selection and evaluation
- **Types of Oracles**:
  - Simple DPLL Oracle: Implements core DPLL algorithm heuristics
  - External Solver Oracle: Wraps external SAT solvers like MiniSAT

### Oracle Integration Example

```python
from symbolicgym.oracles import SimpleDPLLOracle

# Create environment and oracle
env = gym.make("SymbolicGym-v0", cnf_file="problem.cnf")
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

SymbolicGym includes DRAT proof verification for unsatisfiable instances:

```python
from symbolicgym.proofs.verification import ProofVerificationManager

# Verify a DRAT proof
verifier = ProofVerificationManager()
cnf = "p cnf 1 2\n1 0\n-1 0\n"  # Simple UNSAT formula
proof = "d 1 0\n"  # DRAT proof
valid = verifier.verify_solution(cnf, proof)
```

## Environment Integration

Use proof verification within your environment for enhanced rewards:

```python
from symbolicgym.proofs.verification import ProofVerificationManager

# In your environment wrapper
verifier = ProofVerificationManager()
if verifier.is_available():
    if verifier.verify_solution(cnf_formula, agent_proof):
        reward += 10  # Bonus reward for correct proof
    else:
        reward -= 5   # Penalty for invalid proof
```

## Visualization Tools

SymbolicGym includes tools for visualizing agent behavior:

```python
from symbolicgym.visualization import DataVisualizer

visualizer = DataVisualizer(experiment_path="path/to/experiment")
visualizer.plot_clause_satisfaction()
visualizer.plot_variable_assignments()
visualizer.plot_reward_curve()
```

## Using Standard Benchmarks

SymbolicGym supports standard DIMACS CNF benchmark files:

```python
# Using a standard benchmark from SATLIB
env = gym.make("SymbolicGym-v0", cnf_file="benchmarks/uf50-01.cnf")

# Or directly from DIMACS string
dimacs_string = """
p cnf 3 3
1 2 0
-1 3 0
-2 -3 0
"""
from symbolicgym.utils import parse_dimacs
formula = parse_dimacs(dimacs_string)
env = gym.make("SymbolicGym-v0", formula=formula)
```

Common benchmark sources:

- [SATLIB](https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html)
- [SAT Competition](http://www.satcompetition.org/)

### Training with Different Reward Modes

SymbolicGym provides several reward modes suitable for different training scenarios:

```python
# Sparse reward mode - only rewards on complete solution
env_sparse = gym.make("SymbolicGym-v0", formula=formula, reward_mode="sparse")

# Dense reward mode - rewards progress in satisfying clauses
env_dense = gym.make("SymbolicGym-v0", formula=formula, reward_mode="dense")

# Learning reward mode - shaped rewards for better learning signal
env_learning = gym.make("SymbolicGym-v0", formula=formula, reward_mode="learning")
```

Each reward mode changes the learning dynamics:

| Reward Mode | When to Use          | Characteristics                              |
| ----------- | -------------------- | -------------------------------------------- |
| Sparse      | For simple problems  | +1 only when problem is solved               |
| Dense       | For faster learning  | Rewards for each additional satisfied clause |
| Learning    | For complex problems | Balances exploration and exploitation        |

### Integrating with Neural Networks

When working with neural networks, you'll need to preprocess observations:

```python
def create_state_representation(observation):
    """Convert observation dict to a flat vector for neural networks."""
    # Extract observation components
    variables = observation['variables']  # Variable assignments
    clauses = observation['clauses']     # Clause satisfaction

    # You can also include other features like:
    # - Number/percentage of satisfied clauses
    # - Variable occurrence statistics
    # - Recent variable flip history

    # Create combined feature vector
    features = np.concatenate([
        variables,  # Current variable assignments
        clauses,    # Current clause satisfactions
        [np.mean(clauses)],  # Percentage of satisfied clauses
    ])

    return features
```

## Performance Benchmarking

```bash
# Benchmark storage backends
python scripts/benchmark_cli.py run --backends npz hdf5 parquet

# Benchmark solver performance
python scripts/benchmark_solvers.py --time_limit 60
```

## Project Structure

```
src/symbolicgym/
├── __init__.py               # Package initialization and registration
├── envs/                     # Environment implementations
│   ├── core.py               # Core environment classes
│   └── rewards.py            # Reward function implementations
├── oracles/                  # Oracle implementations for guidance
│   ├── base_oracle.py        # Abstract oracle base class
│   └── simple_oracle.py      # SAT solver oracle implementation (was previously listed as sat_oracle.py)
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

If you use SymbolicGym in your research, please cite:

```bibtex
@software{symbolicgym2025,
  title = {SymbolicGym: Reinforcement Learning for Symbolic Reasoning Domains},
  author = {Symbolic Reasoning Project Contributors},
  year = {2025},
  url = {https://github.com/frecsh/SymbolicGym}
}
```

## License

MIT License
