# Enhanced SatRLGym Conversion Checklist

A structured task list for converting the current SAT-RL project into a reusable, locally-installable Gymnasium environment: **SatRLGym v0.1.0**.

"SatRLGym is a Gym-compatible environment for training RL agents on SAT problems and symbolic reasoning tasks. Think OpenAI Gym meets formal methods."

## 0. Safety Net

- [ ] Create backup tag `pre-satrlgym-backup`
  ```bash
  git tag pre-satrlgym-backup
  git push origin pre-satrlgym-backup
  ```

## 1. Repo & Package Structure

- [x] Rename GitHub repo to SatRLGym
- [x] Move runtime code to src/satrlgym/
- [x] Update all sat_rl imports to satrlgym
- [x] Add **init**.py files with version + main exports
- [x] Create proper package structure:
  ```
  src/satrlgym/
  ├── __init__.py               # Added with Gymnasium registration
  ├── envs/
  │   ├── __init__.py
  │   ├── core.py               # Implemented with SatGymEnv
  │   └── rewards.py            # Implemented with reward functions
  ├── oracles/
  │   ├── __init__.py
  │   ├── base_oracle.py        # Created with OracleBase abstract class
  │   ├── simple_oracle.py      # Implemented with SimpleDPLLOracle
  │   └── sat_oracle.py         # Created with SATOracle implementation
  ├── proofs/                   # Created for proof verification
  │   ├── __init__.py
  │   └── drat.py               # DRAT proof checker
  ├── utils/
  │   ├── __init__.py
  │   ├── cnf.py                # For handling CNF formulas
  │   ├── metadata.py           # Added for package metadata
  │   └── sat_utils.py          # Generic SAT utilities
  ├── visualization/            # Created as specified
  │   ├── __init__.py
  │   └── (components planned)
  ├── experience/               # Created for experience data storage
  │   ├── __init__.py
  │   ├── experience_indexing.py # Moved from original codebase
  │   └── (other components planned)
  ├── benchmarks/               # Created for benchmarking tools
  │   └── __init__.py
  └── data/                     # Data storage location
      └── test_problems.py      # Test SAT instances
  ```
- [x] Create enhanced `pyproject.toml` with dependencies:

  ```toml
  [build-system]
  requires = ["setuptools>=42", "wheel"]
  build-backend = "setuptools.build_meta"

  [project]
  name = "satrlgym"
  version = "0.1.0"
  description = "Gymnasium environment for SAT problems with RL hooks"
  readme = "README.md"
  authors = [{name = "Your Name", email = "your.email@example.com"}]
  license = {text = "MIT"}
  classifiers = [
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: MIT License",
      "Topic :: Scientific/Engineering :: Artificial Intelligence",
  ]
  requires-python = ">=3.8"
  dependencies = [
      "gymnasium>=0.28.0",
      "numpy>=1.20.0",
      "matplotlib>=3.5.0",
      "networkx>=2.6.0",  # For clause-variable graphs
      "pyarrow>=8.0.0",   # For experience storage
      "h5py>=3.6.0",      # Alternative storage backend
      "seaborn>=0.11.0",  # For advanced visualizations
  ]

  [project.optional-dependencies]
  dev = [
      "pytest>=6.0",
      "pytest-cov>=2.12.0",  # For coverage reporting
      "black",
      "ruff",
      "mypy",
  ]
  torch = [
      "torch>=1.10.0",
  ]
  tensorflow = [
      "tensorflow>=2.8.0",
  ]
  solvers = [
      "python-sat>=0.1.7.dev3",  # For PySAT integration
  ]
  proof = [
      # DRAT verification is handled by our internal wrapper around drat-trim.c
      # No Python package dependency required
  ]

  [project.urls]
  "Homepage" = "https://github.com/yourusername/satrlgym"
  "Bug Tracker" = "https://github.com/yourusername/satrlgym/issues"
  ```

## 2. Public API Design

- [ ] Define `SatGymEnv` in satrlgym/envs/core.py:
  - [ ] Implement standard Gymnasium methods (`reset()`, `step()`)
  - [ ] Define `observation_space` and `action_space`
  - [ ] Document with proper docstrings
- [ ] Implement reward modes:
  - [ ] "sparse" (1 for solved, 0 for unsolved)
  - [ ] "dense" (based on clause satisfaction progress)
  - [ ] "learning" (with shaping for exploration)
- [ ] Add Oracle protocol in `oracles/oracle_protocol.py`:
  - [ ] Define `oracle.query()` interface for symbolic guidance
  - [ ] Implement standard response format
  - [ ] Add CDCL solver integration
- [ ] Create OracleWrapper class for environment integration:
  - [ ] Define standardized query/response flow between env and oracle
  - [ ] Implement helper methods for common oracle queries
  - [ ] Add metrics collection for oracle usage
- [ ] Add Gym registration in satrlgym/**init**.py:

  ```python
  from gymnasium.envs.registration import register

  register(
      id="SatRLGym-v0",
      entry_point="satrlgym.envs:SatGymEnv",
  )
  ```

- [ ] Ensure satrlgym.make("SatRLGym-v0") works correctly
- [ ] Start API documentation from the beginning:
  - [ ] Add detailed docstrings with examples
  - [ ] Include type hints for all public interfaces

## 3. Enhanced Testing & CI

- [ ] Set up comprehensive tests/ folder with pytest unit tests:
  - [ ] `test_environment.py`: Test environment step/reset behavior
  - [ ] `test_rewards.py`: Test all reward functions
  - [ ] `test_oracles.py`: Test oracle integration
  - [ ] `test_cnf.py`: Test CNF parser functionality
  - [ ] `test_experience.py`: Test experience storage
  - [ ] `test_visualization.py`: Test visualization components
  - [ ] `test_benchmarks/`: Performance benchmarks
- [ ] Add integration tests between components:
  - [ ] `test_oracle_env_integration.py`: Test oracle-environment integration
  - [ ] `test_experience_visualization.py`: Test experience storage-visualization pipeline
- [ ] Add performance tests:
  - [ ] `test_memory_usage.py`: Test memory usage on large SAT instances
  - [ ] `test_step_time.py`: Measure environment step time overhead
- [ ] Test specific environment behaviors:
  - [ ] Test determinism (same seed → same results)
  - [ ] Test observation and action space validation
  - [ ] Test edge cases (all clauses satisfied, no solution, etc.)
- [ ] Add coverage threshold requirements (90%+)
- [ ] Add GitHub Actions workflows:
  - [ ] pytest for unit testing
  - [ ] black, ruff, mypy for code quality
  - [ ] Add CI badge in README
  - [ ] Add matrix testing for Python 3.8, 3.9, 3.10
- [ ] Create test fixtures with small SAT instances

## 4. Baseline Examples & Visualization

- [ ] Create examples/ folder with demos:
  - [ ] examples/run_random_policy.py (baseline performance)
  - [ ] examples/run_minisat.py (oracle integration)
  - [ ] examples/run_ppo.py (RL agent)
- [ ] Include small SATLIB subset (data/random3sat)
- [ ] Create Jupyter notebook tutorial:
  - [ ] Show how to create and reset env
  - [ ] Step through an episode
  - [ ] Visualize clause satisfaction
  - [ ] Plot learning curves
- [ ] Implement visualization components:
  - [ ] S1: Clause-Variable Graph (NetworkX+Matplotlib)
  - [ ] S2: Variable-Assignment Timeline (Seaborn heatmap)
  - [ ] T1: Episode Trace Viewer (Tabular log)
  - [ ] T2: Reward & Clause Count Curves (Line plots)
  - [ ] P2: Action Entropy Curve (Simple metric plot)
- [ ] Add benchmarking utilities:
  - [ ] Compare against CDCL, MiniSAT baseline
  - [ ] Generate performance metrics
  - [ ] Create visualization for benchmark results

## 5. Documentation

- [ ] Set up MkDocs or Sphinx with sphinx-autodoc
- [ ] Add core documentation pages:
  - [ ] Installation guide
  - [ ] API reference (auto-generated from docstrings)
  - [ ] Environment specification
  - [ ] Reward functions
  - [ ] Observation space details
  - [ ] Action space details
  - [ ] Oracle integration guide
  - [ ] Visualization guide
  - [ ] Experience format specification
  - [ ] Benchmarking utilities
- [ ] Add user journey tutorials:
  - [ ] "Getting Started" guide with complete workflow
  - [ ] "Extending SatRLGym" tutorial for custom components
  - [ ] Migration guide from sat-rl to satrlgym
- [ ] Configure GitHub Pages deployment
- [ ] Create docstrings for all public classes and methods
- [ ] Document breaking changes and workarounds

## 6. Proof Checker

- [x] Implement DRAT wrapper in satrlgym.proofs.drat
- [x] Add UNSAT example with verified proof
- [x] Add documentation on proof verification
- [x] Add integration test between environment and proof checker

## 7. Experience Format Implementation

- [ ] Implement Core Format Enhancements:
  - [ ] Type system improvements (explicit dtypes)
  - [ ] Support for complex observation spaces
  - [ ] Create validation utilities
- [ ] Implement Storage Backends:
  - [ ] Apache Arrow/Parquet support
  - [ ] HDF5 backend option
  - [ ] Memory-mapped file support
- [ ] Add Metadata Enrichment:
  - [ ] Git repository tracking
  - [ ] Per-episode random seeds for reproducibility
  - [ ] Hardware fingerprinting
- [ ] Framework Integration:
  - [ ] PyTorch Dataset implementation
  - [ ] TensorFlow data pipeline generators
  - [ ] Common processing functions

## 8. Local Package Installation

- [ ] Define version numbering strategy:
  - [ ] 0.1.x for bug fixes
  - [ ] 0.2.0 for new features
  - [ ] 1.0.0 for stable API
- [ ] Install locally in development mode:
  ```bash
  pip install -e .
  ```
- [ ] Create build script for local wheel:
  ```bash
  python -m pip install --upgrade build
  python -m build
  ```
- [ ] Test installation from local build:
  ```bash
  pip install dist/satrlgym-0.1.0-py3-none-any.whl
  ```
- [ ] Tag v0.1.0 for local versioning:
  ```bash
  git tag v0.1.0
  ```
- [ ] Add environment setup scripts:
  ```bash
  # Create env_setup.sh
  #!/bin/bash
  python -m venv venv
  source venv/bin/activate
  pip install -e .
  ```

## 9. README Polish

- [ ] Update title and add badges (CI, coverage)
- [ ] Add concise description with key features
- [ ] Include a GIF or image of the environment
- [ ] Add local installation instructions:

  ```bash
  # Clone the repository
  git clone https://github.com/yourusername/satrlgym.git
  cd satrlgym

  # Install locally
  pip install -e .
  ```

- [ ] Include a simple "hello world" example:

  ```python
  import gymnasium as gym
  import satrlgym

  env = gym.make("SatRLGym-v0", cnf_file="path/to/problem.cnf")
  obs, info = env.reset(seed=42)

  done = False
  while not done:
      action = env.action_space.sample()  # Replace with your agent
      obs, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated
      if done:
          print(f"Problem solved: {info['solved']}")
  ```

- [ ] Link to documentation site
- [ ] Add installation and dependency information
- [ ] Add section on visualization capabilities
- [ ] Include benchmarking results comparing with baseline solvers

## 10. Feedback Loop

- [ ] Create internal review and feedback process
- [ ] Document known limitations and future enhancements
- [ ] Set up git hooks for pre-commit checks
- [ ] Create v0.1.1 milestone for tracking improvements

## Updated Timeline (6 Weeks)

| Week | Focus                                                                 |
| ---- | --------------------------------------------------------------------- |
| 1    | Sections 0-2: Backup, repo structure, API design, early documentation |
| 2    | Core visualization components (S1, S2)                                |
| 3    | Experience format + remaining visualization                           |
| 4    | Sections 3-4: Tests, CI setup, examples, benchmarking                 |
| 5    | Sections 5-6: Documentation site, proof checker                       |
| 6    | Sections 7-10: Package, publish, announce                             |

## Post-Launch Enhancements (Optional)

- [ ] Docker image with SAT solvers pre-installed
- [ ] Additional wrappers (e.g., MaxSAT rewards)
- [ ] Dataset utility: `satrlgym.datasets.fetch_satlib()`
- [ ] Benchmark suite for comparing agents
- [ ] Integration with more RL frameworks (Stable Baselines3, RLlib)
- [ ] Additional visualizations from Phase 1 plan (S3/T3/P1/M1/M2)
