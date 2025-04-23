# SAT+RL: Reinforcement Learning for SAT Problem Solving

This project explores different multi-agent reinforcement learning approaches for solving Boolean satisfiability (SAT) problems, working toward a novel hybrid framework that integrates RL with other techniques for constraint satisfaction problems.

## Research Overview

This project is part of broader research into hybrid approaches for constraint-satisfying solution space exploration. Traditional SAT solvers, while powerful, are limited by search-based heuristics and scalability constraints. Our work seeks to bridge this gap by:

- Using intelligent agents to generate candidate solutions
- Employing SAT solvers as verifiers to validate constraints
- Providing reinforcement learning-based feedback to optimize agent behavior over time
- Implementing multi-agent communications to explore distinct regions of the solution space

The current implementation focuses on reinforcement learning approaches, with plans to incorporate generative models in future work.

## Approaches Implemented

- **Cooperative Agents**: Multiple agents with shared rewards
- **Competitive Agents**: Agents competing for rewards
- **Communicative Agents**: Agents that share experiences with each other
- **Oracle-Guided Agents**: Agents receiving feedback from traditional SAT solvers

## Project Structure

- `main.py` - Core SAT environment
- `sat_problems.py` - Library of SAT problem definitions
- `multi_q_sat.py` - Cooperative Q-learning implementation
- `multi_q_sat_comp.py` - Competitive Q-learning implementation
- `multi_q_sat_comm.py` - Communicative Q-learning implementation
- `multi_q_sat_oracle.py` - Oracle-guided Q-learning implementation
- `sat_oracle.py` - Traditional SAT solver oracle
- `compare.py` - Comparison script for different agent approaches
- `compare_thresholds.py` - Analysis of communication thresholds
- `compare_oracle_weights.py` - Analysis of oracle influence weights
- `parameter_sweep.py` - Hyperparameter optimization
- `visualize_communication.py` - Visualization of agent communication patterns
- `research.md` - Research proposal and theoretical framework

## Getting Started

### Prerequisites

```
python>=3.8
numpy
matplotlib
seaborn
python-sat (optional, for oracle functionality)
```

### Installation

```bash
pip install -r requirements.txt
```

### Running Experiments

```bash
# Compare cooperative vs competitive approaches
python compare.py

# Test different communication thresholds
python compare_thresholds.py

# Test oracle-guided learning with different weights
python multi_q_sat_oracle.py
python compare_oracle_weights.py

# Visualize communication patterns
python visualize_communication.py
```

## Results

The project demonstrates several key findings:

1. **Cooperative vs Competitive Dynamics**: Cooperative agents generally outperform competitive ones in SAT solving, particularly on simpler problems.

2. **Communication Benefits**: Shared experience between agents improves learning speed and solution quality, with an optimal communication threshold around 0.5 (moderate selectivity).

3. **Oracle Guidance**: Traditional SAT solvers can effectively guide RL agents, helping them overcome difficult clauses.

4. **Problem Difficulty**: Performance gaps between approaches widen as problems approach the phase transition zone (clause-to-variable ratio ~4.2-4.3).

## Future Work

This implementation is an initial step toward our broader research goals. Future plans include:

- Integrating Generative Adversarial Networks (GANs) to intelligently generate candidate solutions
- Expanding the multi-agent communication protocols based on constraint satisfaction quality
- Designing more sophisticated reward functions that balance exploration vs. exploitation
- Testing on larger-scale problems including those in the phase transition zone
- Applying the hybrid approach to related constraint satisfaction problems

## Applications

The techniques developed here have potential applications in:
- Cryptographic key generation with constraints
- Hardware verification optimization
- Complex scheduling and resource allocation problems
- Energy systems and smart grid optimization
- Automated drug discovery and bioinformatics

## License

MIT