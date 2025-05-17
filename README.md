# SymbolicGym: Unified RL Platform for Symbolic Reasoning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/frecsh/SymbolicGym/actions/workflows/ci.yml/badge.svg)](https://github.com/frecsh/SymbolicGym/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/frecsh/SymbolicGym/branch/main/graph/badge.svg)](https://codecov.io/gh/frecsh/SymbolicGym)

**SymbolicGym** is a general-purpose, extensible Gym-compatible platform for reinforcement learning on symbolic reasoning tasks. It supports SAT, SymPy, and Z3 domains, with rich latent-space feedback, cross-domain learning, interpretability tooling, and robust risk mitigation. All major roadmap features are implemented, tested, and verified.

---

## Key Features

- **Unified RL for Symbolic Domains:** SAT, SymPy, Z3
- **Multi-Agent & Single-Agent Support:**
  - Native multi-agent environments (per-agent state/action/obs)
  - PettingZoo-style wrappers, CTDE agents, modular communication
  - Bandwidth penalty, conflict resolution, per-agent reward shaping
- **Cross-Domain Generalization:**
  - Abstract API for graph/vector/dict observations
  - Pluggable feedback backends, shared encoder (MLP/GNN)
- **GNN-based RL Pipeline:** PPO pipeline, multi-agent/cross-domain scripts
- **Comprehensive Test Suite:** Multi-agent, cross-domain, modularity, and benchmarking
- **Documentation & Examples:** Feedback metrics, architecture, usage, and advanced features

---

## Architecture Overview

![SymbolicGym Architecture](docs/architecture_diagram.svg)

---

## Quick Start

```python
import gymnasium as gym
import symbolicgym

# Create a SAT environment (from CNF file or formula dict)
env = gym.make("SymbolicGym-v0", cnf_file="path/to/problem.cnf")
# or:
# formula = {"clauses": [[1, 2], [-1, -2], [1, -2], [-1, 2]], "num_vars": 2}
# env = gym.make("SymbolicGym-v0", formula=formula)
obs, info = env.reset(seed=42)

# Run a random agent
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"Solved: {info['solved']}")
        break
```

---

## Multi-Agent Example

```python
# Multi-agent SAT environment
env = gym.make("SymbolicGym-v0", multi_agent_mode=True, n_agents=2)
obs, info = env.reset()
actions = {aid: env.action_space[aid].sample() for aid in obs}
next_obs, rewards, dones, truncated, infos = env.step(actions)
```

---

## Cross-Domain & GNN Example

```python
from symbolicgym.models.shared_encoder import SharedEncoder
env = gym.make("SymbolicGym-v0", observation_mode="graph")
obs, info = env.reset()
encoder = SharedEncoder(mode="gnn", input_dim=..., hidden_dim=...)
features = encoder(obs)
```

---

## Installation

```bash
git clone https://github.com/frecsh/SymbolicGym.git
cd SymbolicGym
pip install -e .
# Optional dependencies:
pip install -e ".[torch,solvers,proof]"
```

---

## Project Structure

```
src/symbolicgym/
├── envs/           # Environment implementations (SAT, SymPy, Z3, base)
├── agents/         # RL agents (DQN, PPO, GNN, CTDE, Comm, etc.)
├── oracles/        # Oracle guidance modules
├── proofs/         # Proof verification (DRAT, verification utils)
├── utils/          # Utilities (generators, parsing, metrics)
├── visualization/  # Visualization tools
├── experience/     # Experience storage backends
```

---

## Supported Domains

- **SAT**: Boolean satisfiability (phase transition, industrial, crafted)
- **SymPy**: Symbolic algebra (simplification, factoring, equation solving)
- **Z3**: Satisfiability Modulo Theories (SMT)

---

## Reward Modes

- **Sparse**: +1 on solution, 0 otherwise
- **Dense**: Incremental reward for clause satisfaction
- **Learning**: Shaped reward for exploration/exploitation

```python
env = gym.make("SymbolicGym-v0", formula=formula, reward_mode="dense")
```

---

## Feedback Metrics (SAT Example)

- **clause_satisfaction**: Fraction of satisfied clauses
- **variable_decisiveness**: Fraction of assigned variables
- **search_diversity**: Std. dev. of assignments
- **constraint_tension**: Avg. abs. sum of literals per clause
- **proof_progress**: clause_satisfaction × variable_decisiveness

---

## Example: Custom Greedy Agent

```python
class GreedySATAgent:
    def __init__(self, env):
        self.env = env
    def choose_action(self, observation):
        var_counts = defaultdict(int)
        for i, satisfied in enumerate(observation["clause_satisfaction"]):
            if not satisfied:
                for literal in self.env.clauses[i]:
                    var_counts[abs(literal)] += 1
        if var_counts:
            best_var = max(var_counts.items(), key=lambda x: x[1])[0]
            return best_var - 1
        return self.env.action_space.sample()
```

---

## Benchmarks & Results

- See `results/` for learning curves, transfer plots, ablation tables
- Benchmarking scripts and profiling tools for large-scale runs

---

## Reproducibility

- All experiments are seedable and config-driven
- See `scripts/train_sat_dqn.py`, `scripts/train_crossdomain_gnn.py`
- Expanded test coverage in `tests/` (multi-agent, cross-domain, performance)

---

## Documentation & Contribution

- See [`docs/`](docs) for API, integration, and advanced usage
- Contributions welcome! See `CONTRIBUTING.md`

---

## Citation

```bibtex
@software{symbolicgym2025,
  title = {SymbolicGym: Reinforcement Learning for Symbolic Reasoning Domains},
  author = {Symbolic Reasoning Project Contributors},
  year = {2025},
  url = {https://github.com/frecsh/SymbolicGym}
}
```

---

## License

MIT License
