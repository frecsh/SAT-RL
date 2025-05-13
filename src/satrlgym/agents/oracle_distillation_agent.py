"""
Oracle Distillation Agent for SAT problems.
This agent learns from solutions provided by traditional SAT solvers
and transfers that knowledge to a neural network policy.
"""

import os
import subprocess
import tempfile

import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class OracleDistillationAgent:
    def __init__(self, n_vars, clauses):
        """
        Initialize an Oracle Distillation Agent for SAT problems.

        Args:
            n_vars: Number of variables in the SAT problem
            clauses: List of clauses in CNF form
        """
        self.n_vars = n_vars
        self.clauses = clauses

        # Build policy network
        self.model = self._build_policy_network()

        # Store oracle solutions
        self.oracle_solutions = []
        self.solution_trajectories = []

        # Track best solution found
        self.best_assignment = None
        self.best_satisfied = 0

    def _build_policy_network(self):
        """Build neural network for policy approximation"""
        # Use Input layer explicitly instead of passing input_shape to Dense
        inputs = Input(shape=(self.n_vars,))
        x = Dense(128, activation="relu")(inputs)
        x = Dropout(0.2)(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.2)(x)
        # Output one logit for each variable (probability of setting to True)
        outputs = Dense(self.n_vars, activation="sigmoid")(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy")
        return model

    def _create_dimacs_file(self, filename):
        """Create a DIMACS CNF file for the SAT problem"""
        with open(filename, "w") as f:
            # Header: p cnf <num_vars> <num_clauses>
            f.write(f"p cnf {self.n_vars} {len(self.clauses)}\n")

            # Write clauses
            for clause in self.clauses:
                # Each clause ends with 0
                f.write(" ".join(map(str, clause)) + " 0\n")

    def call_oracle_solver(self, solver_cmd="minisat"):
        """Call an external SAT solver (oracle) to get a solution"""
        with tempfile.NamedTemporaryFile(suffix=".cnf", delete=False) as tmp:
            cnf_filename = tmp.name

        # Create CNF file
        self._create_dimacs_file(cnf_filename)

        # Output file for the solution
        out_filename = cnf_filename + ".out"

        # Call the SAT solver
        try:
            subprocess.run(
                [solver_cmd, cnf_filename, out_filename],
                check=True,
                capture_output=True,
            )

            # Parse solution
            solution = None
            with open(out_filename) as f:
                lines = f.readlines()
                if lines and lines[0].strip() == "SAT":
                    # Solution found
                    sol_line = lines[1].strip().split()
                    # Convert to 0/1 array, ignoring the trailing 0
                    solution = np.zeros(self.n_vars, dtype=int)
                    for lit in sol_line:
                        if lit == "0":
                            break
                        var = int(lit)
                        if var > 0:
                            solution[var - 1] = 1  # Variables are 1-indexed in DIMACS

            # Clean up
            os.unlink(cnf_filename)
            os.unlink(out_filename)

            return solution

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error calling SAT solver: {e}")
            # Clean up
            if os.path.exists(cnf_filename):
                os.unlink(cnf_filename)
            if os.path.exists(out_filename):
                os.unlink(out_filename)
            return None

    def count_satisfied_clauses(self, assignment):
        """Count number of clauses satisfied by an assignment"""
        satisfied = 0
        for clause in self.clauses:
            for literal in clause:
                var = abs(literal) - 1  # Convert to 0-indexed
                val = assignment[var]
                if (literal > 0 and val == 1) or (literal < 0 and val == 0):
                    satisfied += 1
                    break

        # Update best assignment if current is better
        if satisfied > self.best_satisfied:
            self.best_satisfied = satisfied
            self.best_assignment = assignment.copy()

        return satisfied

    def _generate_trajectories(
        self, solution, n_trajectories=10, steps_per_trajectory=20
    ):
        """Generate learning trajectories from a known solution"""
        trajectories = []

        for _ in range(n_trajectories):
            # Start with a random assignment
            current = np.random.randint(0, 2, size=self.n_vars)
            trajectory = [current.copy()]

            # Gradually move towards the solution
            for step in range(steps_per_trajectory):
                # Choose variables to flip based on difference from solution
                diff_indices = np.where(current != solution)[0]

                if len(diff_indices) == 0:
                    break  # Already reached the solution

                # Flip one randomly selected differing variable
                idx = np.random.choice(diff_indices)
                current[idx] = 1 - current[idx]

                trajectory.append(current.copy())

            trajectories.append(trajectory)

        return trajectories

    def collect_oracle_data(self, n_solutions=5, n_trajectories=10):
        """Collect solutions and learning trajectories from an oracle solver"""
        # Try to get multiple solutions from the oracle
        for _ in range(n_solutions):
            solution = self.call_oracle_solver()

            if solution is not None:
                self.oracle_solutions.append(solution)

                # Generate learning trajectories from this solution
                trajectories = self._generate_trajectories(
                    solution, n_trajectories=n_trajectories
                )
                self.solution_trajectories.extend(trajectories)

                # Update best solution if needed
                sat_count = self.count_satisfied_clauses(solution)
                print(
                    f"Oracle solution found with {sat_count}/{len(self.clauses)} satisfied clauses"
                )
            else:
                print("Oracle failed to find a solution")

        print(
            f"Collected {len(self.oracle_solutions)} oracle solutions and "
            f"{len(self.solution_trajectories)} learning trajectories"
        )

    def distill_knowledge(self, epochs=50, batch_size=32):
        """Distill knowledge from oracle solutions into policy network"""
        if not self.solution_trajectories:
            print("No learning trajectories available. Call collect_oracle_data first.")
            return False

        # Prepare training data
        states = []
        targets = []

        # Process all trajectories
        for trajectory in self.solution_trajectories:
            for t in range(len(trajectory) - 1):
                state = trajectory[t]
                next_state = trajectory[t + 1]

                # Use the next state as learning target
                states.append(state)
                targets.append(next_state)

        # Convert to numpy arrays
        states = np.array(states)
        targets = np.array(targets)

        # Train the policy network
        print(f"Training policy network on {len(states)} state-target pairs...")
        history = self.model.fit(
            states, targets, epochs=epochs, batch_size=batch_size, verbose=1
        )

        print("Knowledge distillation complete")
        return True

    def predict_assignment(self, state=None):
        """Predict variable assignment using the trained policy network"""
        if state is None:
            # Start with random state if none provided
            state = np.random.randint(0, 2, size=self.n_vars)

        # Get probabilities from policy network
        probs = self.model.predict(np.array([state]), verbose=0)[0]

        # Convert to binary assignment
        assignment = (probs > 0.5).astype(int)

        # Count satisfied clauses
        satisfied = self.count_satisfied_clauses(assignment)

        return assignment, satisfied

    def iterative_refinement(self, max_iterations=100):
        """Iteratively refine a solution using the policy network"""
        # Start with random assignment
        current = np.random.randint(0, 2, size=self.n_vars)
        current_satisfied = self.count_satisfied_clauses(current)

        for i in range(max_iterations):
            # Get prediction from policy network
            probs = self.model.predict(np.array([current]), verbose=0)[0]

            # Create candidate by sampling from probabilities
            candidate = (np.random.random(self.n_vars) < probs).astype(int)

            # Evaluate candidate
            candidate_satisfied = self.count_satisfied_clauses(candidate)

            # Accept if better
            if candidate_satisfied > current_satisfied:
                current = candidate
                current_satisfied = candidate_satisfied
                print(
                    f"Iteration {i}: Improved to {current_satisfied}/{len(self.clauses)} satisfied clauses"
                )

            # Exit if solution found
            if current_satisfied == len(self.clauses):
                print(f"Solution found after {i+1} iterations!")
                break

        return self.best_assignment, self.best_satisfied

    def solve(self):
        """Solve SAT problem using oracle distillation"""
        # Step 1: Collect oracle data
        self.collect_oracle_data()

        # Step 2: Distill knowledge into policy network
        success = self.distill_knowledge()

        if not success:
            print("Failed to distill knowledge. Returning best solution found so far.")
            return self.best_assignment, self.best_satisfied

        # Step 3: Use trained policy for iterative refinement
        return self.iterative_refinement()
