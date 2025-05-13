#!/usr/bin/env python3
"""
Example demonstrating how to use proof verification in an RL environment for SAT solving.
This example shows:
1. Environment setup with proof verification
2. A custom reward function that uses proof verification
3. Performance optimizations with caching
4. Advanced metrics collection
"""

import logging
import os
import sys
import time

# Add the source directory to the path if running from project root
sys.path.insert(0, os.path.abspath("src"))

try:
    import gymnasium as gym

    from satrlgym.proofs.verification import ProofVerificationManager
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Make sure you've installed the package with 'pip install -e .'")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SatRLGym-ProofDemo")


class ProofVerifiedEnv:
    """
    Wrapper that adds proof verification capabilities to a SAT environment.
    This demonstrates how to integrate proof verification with an RL environment.
    """

    def __init__(self, env, proof_bonus=5.0, proof_penalty=-2.0):
        """
        Initialize the wrapper with a SAT environment.

        Args:
            env: The SatRLGym environment to wrap
            proof_bonus: Reward bonus for a valid proof
            proof_penalty: Reward penalty for an invalid proof
        """
        self.env = env
        self.proof_manager = ProofVerificationManager(use_cache=True)
        self.proof_bonus = proof_bonus
        self.proof_penalty = proof_penalty
        self.cnf_formula = None
        self.metrics = {
            "verification_calls": 0,
            "valid_proofs": 0,
            "invalid_proofs": 0,
            "verification_time_total": 0.0,
            "verification_time_avg": 0.0,
        }

        # Get the CNF formula from the environment if available
        if hasattr(env, "formula"):
            self.cnf_formula = env.formula
        elif hasattr(env, "cnf"):
            self.cnf_formula = env.cnf

    def reset(self, **kwargs):
        """Reset the environment and clear the proof verification cache."""
        obs, info = self.env.reset(**kwargs)

        # Get the CNF formula from the reset environment if it changed
        if hasattr(self.env, "formula"):
            self.cnf_formula = self.env.formula
        elif hasattr(self.env, "cnf"):
            self.cnf_formula = self.env.cnf

        # Add proof verification availability to info
        info["proof_verification_available"] = self.proof_manager.is_available()

        return obs, info

    def step(self, action):
        """
        Take a step in the environment and apply proof verification if a proof is provided.

        Args:
            action: The action to take in the environment

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.env.get_state() if hasattr(self.env, "get_state") else {}

        # Take the step in the underlying environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check if a proof was provided in the info dict
        proof = info.get("proof", None)

        if proof and self.proof_manager.is_available() and self.cnf_formula:
            # Measure verification time
            start_time = time.time()

            # Verify the proof
            try:
                verification_result = self.proof_manager.verify_with_metrics(
                    self.cnf_formula, proof
                )

                # Track metrics
                verification_time = time.time() - start_time
                self.metrics["verification_calls"] += 1
                self.metrics["verification_time_total"] += verification_time
                self.metrics["verification_time_avg"] = (
                    self.metrics["verification_time_total"]
                    / self.metrics["verification_calls"]
                )

                # Add verification results to info
                info["proof_verified"] = verification_result["valid"]
                info["verification_time"] = verification_time
                info["proof_steps"] = verification_result.get("proof_steps", 0)

                # Adjust reward based on proof verification
                if verification_result["valid"]:
                    reward += self.proof_bonus
                    self.metrics["valid_proofs"] += 1
                    logger.info(f"Valid proof verified in {verification_time:.4f}s")
                else:
                    reward += self.proof_penalty
                    self.metrics["invalid_proofs"] += 1
                    logger.warning(
                        f"Invalid proof rejected in {verification_time:.4f}s"
                    )

            except Exception as e:
                logger.error(f"Proof verification error: {e}")
                info["proof_verification_error"] = str(e)

        # Add metrics to info
        info["proof_verification_metrics"] = self.metrics.copy()

        return obs, reward, terminated, truncated, info

    def verify_batch_proofs(self, cnf_proofs, parallel=True):
        """
        Verify multiple proofs at once, optionally in parallel.

        Args:
            cnf_proofs: List of (cnf, proof) tuples
            parallel: Whether to use parallel processing

        Returns:
            List of boolean verification results
        """
        if not self.proof_manager.is_available():
            raise RuntimeError("Proof verification not available")

        return self.proof_manager.verify_batch(cnf_proofs, parallel=parallel)

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped environment."""
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env, name)


def generate_simple_unsat_problem():
    """Generate a simple UNSAT problem with a known proof."""
    # Simple unsatisfiable formula: (a) AND (NOT a)
    cnf = "p cnf 1 2\n1 0\n-1 0\n"

    # Simple DRAT proof
    proof = "d 1 0\n"

    return cnf, proof


def demo_verification_in_environment():
    """Demonstrate proof verification in an environment."""
    logger.info("Starting proof verification environment demo")

    # Create a simple environment (mock for this example)
    # In a real scenario, you would use gym.make("SatRLGym-v0")
    env = gym.make("CartPole-v1")  # Just a placeholder
    logger.info("Created mock environment for demonstration")

    # Create the proof verification wrapper
    verified_env = ProofVerifiedEnv(env, proof_bonus=10.0)
    logger.info("Created proof verification wrapper")

    # Generate a simple UNSAT problem and proof
    cnf, proof = generate_simple_unsat_problem()
    verified_env.cnf_formula = cnf

    # Reset the environment
    obs, info = verified_env.reset()
    logger.info(f"Reset environment: {info}")

    # Take a few steps with proof verification
    for i in range(5):
        # In a real scenario, you would get the action from your agent
        action = env.action_space.sample()

        # Add the proof to the info dict to trigger verification
        # In a real scenario, this would come from your agent
        custom_info = {"proof": proof if i % 2 == 0 else "invalid proof"}

        # We're mocking the step method here since this is a dummy environment
        # In a real environment, you would just call step(action)
        original_step = env.step

        def mocked_step(action):
            obs, reward, done, truncated, info = original_step(action)
            info.update(custom_info)
            return obs, reward, done, truncated, info

        env.step = mocked_step

        # Take a step with the proof
        obs, reward, terminated, truncated, info = verified_env.step(action)

        # Restore original step function
        env.step = original_step

        logger.info(f"Step {i+1} reward: {reward}")
        logger.info(f"Proof verified: {info.get('proof_verified', 'N/A')}")

        if terminated:
            logger.info("Episode terminated")
            break

    # Show metrics
    logger.info(f"Verification metrics: {verified_env.metrics}")

    # Clean up
    env.close()


def demo_batch_verification():
    """Demonstrate batch verification of multiple proofs."""
    logger.info("Starting batch verification demo")

    manager = ProofVerificationManager()

    if not manager.is_available():
        logger.error("DRAT verifier not available")
        return

    # Generate multiple proofs
    batch_size = 10
    cnf_proofs = []

    # Generate a batch of identical problems for demonstration
    cnf, valid_proof = generate_simple_unsat_problem()
    invalid_proof = "d 2 0\n"

    for i in range(batch_size):
        # Alternate between valid and invalid proofs
        proof = valid_proof if i % 2 == 0 else invalid_proof
        cnf_proofs.append((cnf, proof))

    logger.info(f"Verifying batch of {batch_size} proofs")

    # Verify serially
    start_time = time.time()
    serial_results = manager.verify_batch(cnf_proofs, parallel=False)
    serial_time = time.time() - start_time
    logger.info(f"Serial verification time: {serial_time:.4f}s")

    # Verify in parallel
    start_time = time.time()
    parallel_results = manager.verify_batch(cnf_proofs, parallel=True)
    parallel_time = time.time() - start_time
    logger.info(f"Parallel verification time: {parallel_time:.4f}s")

    # Compare results
    logger.info(f"Results match: {serial_results == parallel_results}")
    logger.info(f"Valid proofs: {sum(serial_results)}/{batch_size}")
    logger.info(f"Speedup from parallelism: {serial_time/parallel_time:.2f}x")

    # Demonstrate caching
    logger.info("Demonstrating cache effectiveness")
    manager.clear_cache()

    # First run without cache
    start_time = time.time()
    manager.verify_solution(cnf, valid_proof)
    first_time = time.time() - start_time

    # Second run with cache
    start_time = time.time()
    manager.verify_solution(cnf, valid_proof)
    cached_time = time.time() - start_time

    logger.info(f"First verification: {first_time:.6f}s")
    logger.info(f"Cached verification: {cached_time:.6f}s")
    logger.info(f"Cache speedup: {first_time/cached_time:.2f}x")


if __name__ == "__main__":
    logger.info("Starting proof verification demos")

    try:
        # Run the demos
        demo_verification_in_environment()
        demo_batch_verification()
        logger.info("All demos completed successfully")
    except Exception as e:
        logger.error(f"Error in demo: {e}", exc_info=True)
