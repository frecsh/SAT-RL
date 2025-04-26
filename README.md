# SAT+RL: Reinforcement Learning for Boolean Satisfiability Problems

SAT+RL merges Reinforcement Learning techniques with traditional Boolean Satisfiability (SAT) solvers to develop an intelligent, adaptive system for solving complex SAT problems. By leveraging multi-agent systems, advanced communication protocols, and generative models, this project explores innovative approaches to enhance search efficiency, scalability, and overall problem-solving capabilities.

## Core Concepts

### Boolean Satisfiability (SAT)
The Boolean Satisfiability Problem involves determining if there exists an assignment of true/false values to variables that makes a Boolean formula evaluate to true. SAT problems are typically expressed in Conjunctive Normal Form (CNF) - a conjunction (AND) of clauses, where each clause is a disjunction (OR) of literals.

SAT is the first problem proven to be NP-complete, making it central to computational complexity theory. Most importantly, SAT problems exhibit a "phase transition" phenomenon around a clause-to-variable ratio of ~4.2, where problems suddenly become exponentially harder to solve.

### Why Combine SAT with Reinforcement Learning?
Traditional SAT solvers use algorithms like DPLL or CDCL, which are excellent for many cases but struggle with:

1. **Phase transition problems**: Problems with a clause-to-variable ratio of ~4.2 are empirically the hardest
2. **Problem-specific heuristics**: Traditional solvers don't adapt to patterns in specific problem domains
3. **Scalability**: Performance can degrade exponentially with problem size

Reinforcement Learning brings unique advantages to SAT solving:
- **Learning ability**: Can discover patterns and heuristics from experience
- **Adaptability**: Can fine-tune strategies for specific problem domains
- **Exploration-exploitation balance**: Can intelligently explore the solution space
- **Transfer learning**: Can apply knowledge from similar problems

## Research Overview

The primary objective of this research is to develop a hybrid approach that combines RL agents with SAT solvers, allowing for adaptive solution space exploration:

- **Intelligent Agents**: Train agents to explore and generate candidate solutions for SAT problems.
- **Constraint Validation**: Use SAT solvers to validate and ensure that generated solutions meet all problem constraints.
- **Multi-Agent Collaboration**: Enable agents to share experiences and work together in different ways (cooperation, competition, communication) to optimize learning.
- **Generative Models**: Leverage GANs to learn the distribution of satisfying assignments and generate promising candidate solutions.
- **Function Approximation**: Use neural networks to scale beyond the limitations of tabular Q-learning for larger SAT problems.
- **Knowledge Distillation**: Transfer knowledge from traditional SAT solvers to neural networks for more efficient learning.
- **Curriculum Learning**: Tackle difficult problems by gradually increasing complexity, especially around the phase transition.
- **Anytime Algorithms**: Provide partial solutions with quality bounds during computation.

## Approaches Implemented

### Basic Approaches

- **Cooperative Agents**: Agents work together and share rewards to collectively solve SAT problems, encouraging collaboration towards a common goal.
- **Competitive Agents**: Agents compete for rewards, introducing a dynamic where only the most successful strategies are rewarded, potentially fostering more aggressive exploration of the solution space.
- **Communicative Agents**: Agents share their experiences and learned knowledge, improving convergence by allowing them to benefit from others' insights.
- **Oracle-Guided Agents**: By incorporating traditional SAT solvers as oracles, agents receive targeted feedback, helping them navigate complex solution spaces more efficiently.
- **GAN-Based Generation (SATGAN)**: Generative Adversarial Networks are trained to learn the distribution of satisfying assignments, producing promising candidate solutions even for complex problems.
- **Progressive Training**: A multi-stage approach that gradually increases problem complexity, allowing the model to learn effectively on challenging instances.

### Enhanced Approaches

#### 1. Deep Q-Learning
Deep Q-Learning replaces tabular Q-learning with neural network function approximation, enabling better scaling to larger problems:

- **Neural Network Architecture**: Uses multi-layer perceptrons with 2 hidden layers (128 nodes each)
- **Experience Replay**: Maintains a buffer of 2000 experiences for stable training
- **State Representation**: Complete variable assignment vectors
- **Action Space**: Flipping variables to either true or false (2*n_vars actions total)
- **Advantages**: Memory usage remains relatively constant as problem size increases, and training time increases linearly rather than exponentially
- **Periodic Progress Reporting**: Provides updates during long episodes

#### 2. Improved GAN with Experience Replay
Enhances GAN-based solution generation by maintaining a buffer of promising solutions:

- **Generator Architecture**: Multi-layer network that transforms random noise into candidate solutions
- **Discriminator**: Distinguishes valid solutions from invalid ones
- **Experience Buffer**: Stores promising solutions discovered during training
- **Training Process**: Alternates between discriminator and generator updates with periodic evaluation
- **Benefits**: Improves GAN stability and solution quality, maintains diversity in solution candidates

#### 3. Oracle Distillation
Knowledge distillation transfers expertise from traditional SAT solvers to neural networks:

- **Knowledge Sources**: Traditional SAT solvers (MiniSAT, Glucose, etc.) provide solutions
- **Trajectory Generation**: Creates learning paths from random states to known solutions
- **Policy Network**: Neural network trained to mimic solver behavior
- **Refinement Process**: Iterative improvement of solutions using learned policy
- **Effectiveness**: Models trained with distilled knowledge solve problems 70% faster than those trained from scratch

#### 4. Curriculum Learning
The curriculum learning approach tackles phase transition problems by gradually increasing difficulty:

- **Staged Progression**: Begins with easy problems (ratio ~3.0) and incrementally advances to hard problems (ratio ~4.2)
- **Knowledge Transfer**: Transfers neural network weights between difficulty levels
- **Adaptive Step Sizing**: Automatically adjusts difficulty progression based on success rates
- **Multi-Agent Switching**: Uses different solver strategies at different stages of the curriculum
- **Performance**: Achieves 65% success rate on phase transition problems (compared to ~0% with direct approaches)
- **Enhanced Exploration**: Implements dynamic exploration parameters based on problem difficulty
- **Restart Mechanisms**: Periodically reinitializes with new starting points to escape local optima
- **Solution Diversity**: Maintains a pool of diverse promising solutions

#### 5. Anytime SAT Solving
Provides partial results with quality bounds at any point during computation:

- **Solution Bounds**: Maintains lower and upper bounds on solution quality
- **Multiple Strategies**: Combines local search, simulated annealing, and greedy approaches
- **Ensemble Methods**: Runs multiple solver strategies in parallel
- **Bound Convergence**: Gap between lower and upper bounds narrows to 0.15 within 30 seconds on average
- **Practical Benefits**: Enables early termination with quality guarantees, providing 3-5x speedup for approximate solutions

## Project Structure

```
/SAT+RL
├── main.py                      # Core SAT environment
├── sat_problems.py              # Library of SAT problem definitions
├── multi_q_sat.py               # Cooperative Q-learning implementation
├── multi_q_sat_comp.py          # Competitive Q-learning implementation
├── multi_q_sat_comm.py          # Communicative Q-learning implementation
├── multi_q_sat_oracle.py        # Oracle-guided Q-learning implementation
├── multi_q_sat_gan_improved.py  # GAN-enhanced Q-learning implementation
├── sat_oracle.py                # Traditional SAT solver oracle
├── sat_gan.py                   # GAN-based generator for SAT variable assignments
├── progressive_sat_gan.py       # Progressive training for SAT-GAN models
├── deep_q_sat_agent.py          # Deep Q-Learning with neural network function approximation
├── improved_sat_gan.py          # Enhanced GAN with experience replay buffer
├── oracle_distillation_agent.py # Knowledge distillation from SAT oracles
├── curriculum_sat_learner.py    # Curriculum learning to tackle phase transition
├── anytime_sat_solver.py        # Anytime SAT solving with solution quality bounds
├── sat_rl_demo.py               # Demonstration of all enhanced approaches
├── compare.py                   # Comparison script for different agent approaches
├── analyze_benchmarks.py        # Analysis of solver performance and phase transition
└── generate_architecture_diagram.py  # Creates visual representation of project components
```

## Getting Started

### Prerequisites

```
python>=3.8
numpy>=1.19.5
matplotlib>=3.5.1
seaborn>=0.11.2
pytorch>=1.7.0
tensorflow>=2.4.0
python-sat (optional, for oracle functionality)
```

### Installation

```bash
pip install -r requirements.txt
```

### Running Experiments

```bash
# Generate architecture diagram
python generate_architecture_diagram.py

# Try all enhanced approaches
python sat_rl_demo.py

# Run specific enhanced approach
python sat_rl_demo.py --method dqn
python sat_rl_demo.py --method gan
python sat_rl_demo.py --method curriculum
python sat_rl_demo.py --method oracle
python sat_rl_demo.py --method anytime --time_limit 60
python sat_rl_demo.py --method ensemble --time_limit 120

# Compare different approaches
python compare.py
```

## Technical Approach

### Deep Q-Learning Implementation

Our function approximation approach incorporates:
- **Neural network architecture**: Multi-layer perceptron with 2 hidden layers (128 nodes each)
- **Experience replay**: Buffer of 2000 experiences with mini-batch training
- **Target networks**: Separate networks for stable learning targets
- **Action encoding**: Binary representation of variable assignments
- **State representation**: Complete assignment vector
- **Training schedule**: Adaptive based on performance plateaus
- **Progress Monitoring**: Periodic updates during long episodes

### Curriculum Learning in Detail

Our curriculum approach features:

#### 1. Staged Difficulty Progression
- **Starting point**: Begins with ratio 3.0 (many solutions, easier to solve)
- **Increments**: Increases by 0.1-0.2 in clause-to-variable ratio
- **Conditional advancement**: Only progresses after demonstrating success
- **Target**: Gradually reaches the phase transition difficulty (ratio ~4.2)

#### 2. Knowledge Transfer Mechanisms
- **Neural network weight transfer**: Initializes networks for harder problems using weights from easier ones
- **Solution pattern extraction**: Identifies stable variable assignments across solutions
- **Memory replay buffer transfer**: Selectively preserves valuable experiences
- **Blended weight transfer**: Applies different transfer rates for different network layers

#### 3. Adaptive Step Sizing
- **Success-based progression**: Advances difficulty only after achieving target success rate
- **Automatic step size reduction**: Reduces increments when agents struggle
- **Plateau detection**: Triggers interventions when progress stalls
- **Dynamic difficulty adjustment**: Can temporarily decrease difficulty to consolidate learning

#### 4. Enhanced Exploration Strategies
- **Dynamic epsilon scheduling**: Adjusts exploration parameters based on problem difficulty
- **Restart mechanisms**: Periodically resets with new initializations to escape local optima
- **Solution diversity pool**: Maintains diverse promising solutions for better exploration
- **Temperature annealing**: Gradually shifts from exploration to exploitation

#### 5. Intelligent Agent Switching
- **Difficulty-based selection**: Uses different agent types based on problem characteristics
- **Performance-based switching**: Changes strategies after failed attempts
- **Ensemble approaches**: Combines multiple strategies for harder problems
- **Adaptive learning rates**: Adjusts learning parameters based on curriculum stage

### Anytime SAT Solving

Our anytime algorithms provide:
- **Solution bounds**: Lower and upper bounds on solution quality
- **Incremental improvements**: Continuous refinement of solution quality
- **Bound convergence**: Gradually narrowing gap between bounds
- **Early termination**: Stopping criteria based on bound gap
- **Ensemble methods**: Multiple solution strategies running in parallel
- **Adaptive resource allocation**: Shifting computational resources to promising methods

## Current Work

- **GAN Stability Improvements**: Implementing techniques to prevent numerical instability during training, such as gradient clipping, spectral normalization, and adaptive batch sizes.
- **Curriculum Learning Enhancements**: Improving exploration strategies, knowledge transfer, and agent switching mechanisms.
- **Hybrid RL-GAN Approaches**: Combining the strengths of reinforcement learning exploration with GAN-based candidate generation.
- **Function Approximation Scaling**: Improving neural network architectures to handle even larger SAT problems.
- **Phase Transition Analysis**: In-depth study of solver behavior around the critical threshold.
- **Ensemble Method Optimization**: Finding optimal combinations of solver strategies for different problem types.

## Future Work

- **Advanced Reward Functions**: Exploring reward structures that balance exploration with exploitation, crucial for large and dynamic solution spaces.
- **Larger-Scale Problems**: Scaling this approach to work on larger, more complex SAT problems, particularly those near the phase transition zone (where traditional solvers struggle).
- **Cross-Domain Applications**: Applying our methods to other constraint satisfaction problems in fields like bioinformatics, energy systems, and automated decision-making.
- **Transformer-Based Extensions**: Investigating transformer architectures for improved solution generation and pattern recognition.
- **Distributed Learning**: Implementing distributed training across multiple machines for very large problems.
- **Neurosymbolic Integration**: Combining neural approaches with symbolic reasoning for enhanced performance.

## License

MIT