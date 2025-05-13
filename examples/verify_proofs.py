#!/usr/bin/env python3
"""
Example script demonstrating how to use the DRAT proof verification in SatRLGym.
"""

import argparse
import os
import sys

# Add the source directory to the path if running from project root
sys.path.insert(0, os.path.abspath("src"))

try:
    from satrlgym.proofs.drat import DRATVerifier
    from satrlgym.proofs.verification import ProofVerificationManager
except ImportError:
    print("Failed to import satrlgym proof verification modules.")
    print("Make sure you've installed the package with 'pip install -e .'")
    sys.exit(1)


def create_example_files():
    """Create example CNF and proof files for demonstration."""
    # Simple unsatisfiable formula: (x1) AND (NOT x1)
    cnf_content = "p cnf 1 2\n1 0\n-1 0\n"

    # DRAT proof: delete clause (x1)
    proof_content = "d 1 0\n"

    with open("example.cnf", "w") as f:
        f.write(cnf_content)

    with open("example.drat", "w") as f:
        f.write(proof_content)

    return "example.cnf", "example.drat"


def basic_verification():
    """Demonstrate basic proof verification."""
    print("\n=== Basic Verification ===")

    try:
        verifier = DRATVerifier()
        print("✓ Successfully initialized DRAT verifier")
    except FileNotFoundError:
        print("✗ Failed to initialize DRAT verifier: drat-trim.c not found")
        return
    except RuntimeError as e:
        print(f"✗ Failed to initialize DRAT verifier: {e}")
        return

    # Create example files
    cnf_file, proof_file = create_example_files()
    print(f"Created example files: {cnf_file} and {proof_file}")

    # Verify from files
    print("\nVerifying from files...")
    result = verifier.verify(cnf_file, proof_file)
    print(f"Verification result: {'✓ Valid' if result else '✗ Invalid'}")

    # Verify from strings
    print("\nVerifying from strings...")
    cnf_content = "p cnf 1 2\n1 0\n-1 0\n"
    proof_content = "d 1 0\n"
    result = verifier.verify(cnf_content, proof_content)
    print(f"Verification result: {'✓ Valid' if result else '✗ Invalid'}")

    # Try an invalid proof
    print("\nTrying an invalid proof...")
    invalid_proof = "d 2 0\n"  # Trying to delete a non-existent clause
    result = verifier.verify(cnf_content, invalid_proof)
    print(
        f"Verification result (should be invalid): {'✓ Valid' if result else '✗ Invalid'}"
    )


def using_verification_manager():
    """Demonstrate using the verification manager."""
    print("\n=== Using Verification Manager ===")

    manager = ProofVerificationManager()
    if not manager.is_available():
        print("✗ DRAT verification not available")
        return

    print("✓ DRAT verification is available")

    # Simple verification
    cnf_content = "p cnf 1 2\n1 0\n-1 0\n"
    proof_content = "d 1 0\n"

    result = manager.verify_solution(cnf_content, proof_content)
    print(f"Verification result: {'✓ Valid' if result else '✗ Invalid'}")

    # Show integration with a reward function
    print("\nExample reward function integration:")

    def reward_function(cnf, proof, is_valid):
        """Example reward function using proof verification."""
        base_reward = -0.1  # Small penalty for each step
        if is_valid:
            print("  Proof verified successfully: +5.0 reward")
            return base_reward + 5.0
        else:
            print("  Proof verification failed: -2.0 penalty")
            return base_reward - 2.0

    # Valid proof
    reward = reward_function(
        cnf_content, proof_content, manager.verify_solution(cnf_content, proof_content)
    )
    print(f"  Total reward: {reward}")

    # Invalid proof
    reward = reward_function(
        cnf_content, "d 2 0\n", manager.verify_solution(cnf_content, "d 2 0\n")
    )
    print(f"  Total reward: {reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRAT proof verification example.")
    parser.add_argument(
        "--basic", action="store_true", help="Run basic verification example"
    )
    parser.add_argument(
        "--manager", action="store_true", help="Run verification manager example"
    )
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    # Default to running all examples if no arguments provided
    if not (args.basic or args.manager or args.all):
        args.all = True

    print("DRAT Proof Verification Example")
    print("==============================")

    if args.basic or args.all:
        basic_verification()

    if args.manager or args.all:
        using_verification_manager()

    print(
        "\nExample completed. If no errors occurred, the DRAT verification is working correctly."
    )
