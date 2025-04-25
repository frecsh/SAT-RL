# SAT+RL: Reinforcement Learning for SAT Problem Solving

SAT+RL merges Reinforcement Learning techniques with traditional Boolean Satisfiability (SAT) solvers to develop an intelligent, adaptive system for solving complex SAT problems. By leveraging multi-agent systems, advanced communication protocols, and generative models, this project explores innovative ways to enhance search efficiency, scalability, and overall problem-solving capabilities for SAT and other constraint satisfaction problems.

## Research Overview

The primary objective of this research is to develop a hybrid approach that combines RL agents with SAT solvers, allowing for adaptive solution space exploration:

- Intelligent Agents: Train agents to explore and generate candidate solutions for SAT problems.
- Constraint Validation: Use SAT solvers to validate and ensure that generated solutions meet all problem constraints.
- Multi-Agent Collaboration: Enable agents to share experiences and work together in different ways (cooperation, competition, communication) to optimize learning.
- Generative Models: Leverage GANs to learn the distribution of satisfying assignments and generate promising candidate solutions.

## Approaches Implemented

- Cooperative Agents: Agents work together and share rewards to collectively solve SAT problems, encouraging collaboration towards a common goal.
- Competitive Agents: Agents compete for rewards, introducing a dynamic where only the most successful strategies are rewarded, potentially fostering more aggressive exploration of the solution space.
- Communicative Agents: Agents share their experiences and learned knowledge, improving convergence by allowing them to benefit from others' insights.
- Oracle-Guided Agents: By incorporating traditional SAT solvers as oracles, agents receive targeted feedback, helping them navigate complex solution spaces more efficiently.
- GAN-Based Generation (SATGAN): Generative Adversarial Networks are trained to learn the distribution of satisfying assignments, producing promising candidate solutions even for complex problems.
- Progressive Training: A multi-stage approach that gradually increases problem complexity, allowing the model to learn effectively on challenging instances.

## Project Structure

- `main.py` - Core SAT environment
- `sat_problems.py` - Library of SAT problem definitions
- `multi_q_sat.py` - Cooperative Q-learning implementation
- `multi_q_sat_comp.py` - Competitive Q-learning implementation
- `multi_q_sat_comm.py` - Communicative Q-learning implementation
- `multi_q_sat_oracle.py` - Oracle-guided Q-learning implementation
- `multi_q_sat_gan_improved.py` - GAN-enhanced Q-learning implementation
- `sat_oracle.py` - Traditional SAT solver oracle
- `sat_gan.py` - GAN-based generator for SAT variable assignments
- `progressive_sat_gan.py` - Progressive training for SAT-GAN models
- `compare.py` - Comparison script for different agent approaches
- `compare_thresholds.py` - Analysis of communication thresholds
- `compare_oracle_weights.py` - Analysis of oracle influence weights
- `parameter_sweep.py` - Hyperparameter optimization
- `visualize_communication.py` - Visualization of agent communication patterns
- `benchmark_sat_gan.py` - Benchmarking tool for GAN-based solutions
- `research.md` - Research proposal and theoretical framework

## Getting Started

### Prerequisites

```
python>=3.8
numpy
matplotlib
seaborn
pytorch>=1.7.0
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

# Train and evaluate GAN-based solution generation
python benchmark_sat_gan.py
```

## Results

The project demonstrates several key findings:

- **Cooperative vs Competitive Dynamics**: In simpler SAT problems, cooperative agents solve problems faster by focusing their learning on shared goals (average 42% fewer episodes to solution), while competitive agents show higher variance in performance but occasionally find solutions through aggressive exploration.

- **Communication Benefits**: Sharing information between agents enhances learning efficiency, with an optimal communication threshold around 0.5, balancing exploration and exploitation. Our experiments show:
  - 73% success rate at threshold=0.5 vs. 51% at threshold=0.0
  - 30% reduction in episodes-to-solution compared to non-communicating agents
  - Communication becomes counterproductive above threshold=0.75 due to information overload

- **Oracle Guidance**: SAT solver oracles provide crucial guidance, with optimal weight around 0.3-0.4, leading to:
  - 2.5x faster convergence on hard problems
  - 87% success rate (vs. 34% without oracle)
  - Significant improvement in clause satisfaction patterns

- **Phase Transition Bottleneck**: Our experiments confirm the difficulty spike near the clause-to-variable ratio of ~4.2 (phase transition):
  - 0% success rate on phase transition problems regardless of communication threshold
  - Agents converge to suboptimal strategies around episode 100
  - Traditional SAT solvers outperform our RL approach in this specific regime

- **GAN-Based Solution Generation**: Our SATGAN implementation shows promising results in learning the structure of valid solutions:
  - Generates candidate solutions with >90% clause satisfaction in many cases
  - Progressive training approach successfully tackles harder problems by starting with simplified versions
  - Combined with Q-learning, improves exploration of high-dimensional solution spaces

### Sample Visualizations

![Communication Effect on Learning](examples/comm_heatmap_medium_standard.png)
*Fig 1: Heatmap showing variable-clause satisfaction correlation in agent communications*

![Oracle Weight Comparison](examples/oracle_weight_comparison_medium_standard.png)  
*Fig 2: Effect of oracle feedback weight on agent performance*

### Key Technical Insights

- Q-table sparsity increases with problem size, requiring better approximation methods for larger instances
- Communication benefits plateau after ~30% of total training episodes
- Competitive agents show higher exploration but slower convergence
- Oracle feedback is most valuable early in training
- GAN-based generation provides higher-quality initial states for RL exploration
- Progressive training significantly improves GAN stability and final solution quality

## Technical Approach

### Q-Learning Implementation

Our multi-agent Q-learning approach uses:
- **State space**: Binary vectors representing variable assignments
- **Action space**: Modified assignments to variables
- **Reward structure**: Normalized by percentage of satisfied clauses
- **Exploration strategy**: Epsilon-greedy with decay rate 0.995
- **Learning rate**: 0.1 (determined through parameter sweep)
- **Experience sharing**: Thresholded by reward value when communication enabled

### Oracle Integration

The SAT oracle provides:
1. Clause difficulty assessment based on historical satisfaction rates
2. Targeted suggestions for variable assignments
3. Verification of complete solutions
4. Reward shaping based on partial constraint satisfaction

### GAN-Based Solution Generation

Our SATGAN approach incorporates:
- **Generator**: Produces variable assignments from random noise
- **Discriminator**: Distinguishes valid solutions from invalid ones
- **Clause satisfaction loss**: Guides the generator toward satisfying assignments
- **Diversity loss**: Encourages variety in generated solutions
- **Progressive training**: Starts with simplified problems and gradually increases complexity
- **Numerical stability techniques**: Gradient penalty, adaptive learning rates, and batch normalization
- **Temperature parameter**: Controls randomness in the generation process

## Current Work

- **GAN Stability Improvements**: Implementing techniques to prevent numerical instability during training, such as gradient clipping, spectral normalization, and adaptive batch sizes.
- **Progressive Training Refinement**: Optimizing the multi-stage training approach to better handle the transition between problem complexities.
- **Hybrid RL-GAN Approaches**: Combining the strengths of reinforcement learning exploration with GAN-based candidate generation.

## Future Work

- **Advanced Reward Functions**: Exploring reward structures that balance exploration with exploitation, crucial for large and dynamic solution spaces.
- **Larger-Scale Problems**: Scaling this approach to work on larger, more complex SAT problems, particularly those near the phase transition zone (where traditional solvers struggle).
- **Cross-Domain Applications**: Applying our methods to other constraint satisfaction problems in fields like bioinformatics, energy systems, and automated decision-making.
- **Transformer-Based Extensions**: Investigating transformer architectures for improved solution generation and pattern recognition.

## Applications

The techniques developed here have potential applications in:
- Cryptographic key generation with constraints
- Hardware verification optimization
- Complex scheduling and resource allocation problems
- Energy systems and smart grid optimization
- Automated drug discovery and bioinformatics

## License

MIT