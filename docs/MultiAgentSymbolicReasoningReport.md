# Multi-Agent Deep RL & Symbolic Reasoning for SAT Solving Near Phase Transition

## System Architecture Overview

We propose a **multi-agent SAT-solving framework** in which specialized RL agents collaborate under guidance from symbolic modules:

1. **SAT Environment**

   - Encodes the CNF formula and current partial assignment
   - Enforces transitions (unit propagation, backtracking on conflicts, etc.)

2. **Agents**

   - **Branching Agent (DQN-based):** Selects next variable and polarity
   - **Conflict Agent (Actor-Critic):** Performs conflict-driven clause learning and backjumping

3. **Symbolic Reasoning Module (SAT Oracle)**

   - Validates partial assignments
   - Generates proof certificates
   - Suggests heuristic hints to agents

4. **Communication Bus**

   - Exchanges compact messages (e.g., partial assignments, learned-clause summaries)
   - Enables cooperation or competition among agents

5. **Reward-Shaping Engine**

   - Dynamically adjusts rewards based on problem difficulty and agent progress

6. **Curriculum Controller**
   - Schedules SAT instances of increasing hardness toward the critical clause density (~4.3 for 3-SAT)

Together, these components form a **hybrid neurosymbolic architecture**: neural agents make local decisions while symbolic modules inject structure and interpretability.

---

## Core Components

- **Agents**

  - _Value-Based (DQN) Agent_: Learns Q-values for discrete variable selections
  - _Actor-Critic Agent_: Manages continuous or high-level decisions (e.g., restart strategies)
  - _Additional Agents_: Monitor unit propagation or propose sub-problem splits
  - _Reward Modes_: Cooperative (shared reward) or competitive (individual credit)

- **Environment Interface**

  - **Observations**: CNF clauses, assignment history, learned clauses
  - **Actions**: assign variable, backtrack, add clause, etc.
  - **Rewards**: +1 for full solution, intermediate rewards for milestones (e.g., unit propagation)

- **Symbolic Module (SAT Oracle)**

  - Runs in parallel to serve as a “teacher”
  - Provides **knowledge distillation**: labels states with optimal moves or proof insights

- **Reward Shaping & Curriculum Engine**
  - Monitors clause-to-variable ratio and agent performance
  - Applies bonus rewards for difficult milestones
  - Implements a curriculum from easy to hard SAT instances

---

## Information Flow & Interaction

1. **State Emission**: Environment emits current assignment, clause set, and conflict flags.
2. **Agent Action**:
   - Branching Agent picks a variable and polarity via Q-network.
   - Environment applies action, triggers unit propagation or conflict.
3. **Conflict Handling**:
   - Conflict Agent chooses a learned clause to add or a backjump level.
4. **Communication**:
   - Agents share key events (e.g., “variable X caused a conflict”).
5. **Reward Computation**:
   - Reward Engine assigns +0.1 per clause eliminated, +1.0 for solution, negative for deep conflicts.
   - Dynamically adjusted by curriculum controller.
6. **Knowledge Distillation**:
   - Periodically, SAT Oracle solves (sub)instance, produces proofs.
   - Distilled into policy updates or replay-buffer augmentation.
7. **Training Loop**:
   - Agents learn from both RL updates (Q-learning or actor-critic) and symbolic feedback.

---

## Agent Specialization

- **Variable-Selection Agent (DQN)**
  Learns a Q-function over (state, variable) to choose branching variables (optionally via a GNN on the CNF).

- **Conflict-Resolution Agent (Actor-Critic)**
  Chooses which learned clause to add or where to backjump when conflicts occur.

- **Knowledge Agent**
  Observes agent trajectories, consults SAT Oracle, distills high-level patterns into symbolic rules or reward signals.

- **Curriculum Controller**
  Adjusts problem difficulty based on agent success rates, smoothing the phase-transition cliff.

Agents train under a **centralized training, decentralized execution (CTDE)** scheme, encouraging specialized expertise.

---

## Phased Development Roadmap

1. **Baseline Single-Agent RL**

   - Milestone: Solve toy SAT instances (~20 vars) with DQN
   - Benchmark against vanilla CDCL

2. **Multi-Agent Coordination**

   - Milestone: Two+ agents working cooperatively
   - Compare cooperative vs competitive reward schemes

3. **Actor-Critic Integration**

   - Milestone: Introduce Actor-Critic for clause learning policies

4. **Symbolic Knowledge Distillation**

   - Milestone: Oracle-guided policy distillation, accelerated convergence

5. **Curriculum & Reward Shaping**

   - Milestone: Master phase-transition instances with dynamic difficulty and rewards

6. **Scaling & Evaluation**
   - Milestone: Benchmark on larger SAT problems, analyze interpretability gains

---

## Theoretical Risks & Failure Modes

- **Non-Stationarity & Credit Assignment**
  Agents may destabilize each other’s learning; credit for shared success becomes unclear.

- **Communication Bottlenecks**
  Overhead from message exchange can slow training; too little communication prevents coordination.

- **Reward Shaping Pitfalls**
  Mis-shaped rewards risk rewarding suboptimal strategies or gaming intermediate goals.

- **Curriculum Forgetting**
  Agents may forget simpler tasks when advancing to harder instances; require revisiting easy problems.

- **Symbolic Noise & Misguidance**
  Oracle feedback may be noisy or suboptimal, potentially misleading agents.

- **Interpretability vs Performance**
  Rich symbolic representations improve transparency but may hamper scalability and speed.

- **Compute Constraints**
  Multi-agent and hybrid training can be resource-intensive on standard lab hardware.

---

## Citations & Further Reading

- **Interpretable RL with Neural Symbolic Logic**
  https://openreview.net/forum?id=M_gk45ItxIp
- **RLSF: Reinforcement Learning via Symbolic Feedback**
  https://openreview.net/forum?id=vf8iou7FNF
- **Curriculum-Driven Deep RL for Hard Planning Instances**
  https://www.ijcai.org/proceedings/2020/0304.pdf
- **Neuro-symbolic RL at NeurIPS 2020**
  https://proceedings.neurips.cc/paper/2020/file/6d70cb65d15211726dcce4c0e971e21c-Paper.pdf

---

_This README outlines a research-oriented blueprint for a hybrid multi-agent RL and symbolic reasoning approach to solving SAT near its phase transition. It is intended as a living document to guide prototype development, experimentation, and eventual publication._
