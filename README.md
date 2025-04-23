# SAT+RL: Reinforcement Learning for SAT Problem Solving

This project explores different multi-agent reinforcement learning approaches for solving Boolean satisfiability (SAT) problems.

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

- Extend to more complex SAT problem families
- Implement hybrid approaches combining cooperation and competition
- Explore more sophisticated communication protocols
- Apply to other constraint satisfaction problems

## License

MIT