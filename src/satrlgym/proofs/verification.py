"""Utilities for proof verification in SAT environments."""

import logging
from typing import Any

from satrlgym.proofs.drat import DRATVerifier


class ProofVerificationManager:
    """Manager for verifying proofs in SAT environments."""

    def __init__(self, debug=False, use_cache=True, max_cache_size=1000):
        """Initialize the manager.

        Args:
            debug: If True, print debug information during verification
            use_cache: Whether to use caching for verification results
            max_cache_size: Maximum number of entries in the cache
        """
        try:
            self.verifier = DRATVerifier(
                debug=debug, use_cache=use_cache, max_cache_size=max_cache_size
            )
            self.available = True
        except (FileNotFoundError, RuntimeError) as e:
            self.available = False
            logging.warning(f"DRAT verifier not available: {e}")

    def verify_solution(self, cnf_formula, proof, options=None):
        """
        Verify a solution proof for a SAT instance.

        Args:
            cnf_formula: CNF formula as string or path
            proof: DRAT proof as string or path
            options: Additional command line options for DRAT-trim

        Returns:
            bool: True if proof is valid, False otherwise
        """
        if not self.available:
            raise RuntimeError("DRAT verifier not available")

        return self.verifier.verify(cnf_formula, proof, options)

    def verify_batch(
        self,
        cnf_proofs: list[tuple[str, str]],
        parallel: bool = True,
        num_workers: int | None = None,
    ) -> list[bool]:
        """
        Verify multiple proofs, optionally in parallel.

        Args:
            cnf_proofs: List of (cnf_formula, proof) tuples
            parallel: Whether to use parallel processing
            num_workers: Number of worker processes/threads (defaults to CPU count)

        Returns:
            List[bool]: List of verification results, True for valid proofs
        """
        if not self.available:
            raise RuntimeError("DRAT verifier not available")

        # Pass the use_cache setting from the verifier
        return DRATVerifier.verify_batch(
            cnf_proofs,
            parallel=parallel,
            num_workers=num_workers,
            use_cache=self.verifier.use_cache,
        )

    def clear_cache(self):
        """Clear the verification cache"""
        if self.available:
            DRATVerifier.clear_cache()

    def is_available(self):
        """Check if proof verification is available."""
        return self.available

    def verify_with_metrics(self, cnf_formula, proof) -> dict[str, Any]:
        """
        Verify a proof and return detailed metrics about the verification.

        Args:
            cnf_formula: CNF formula as string or path
            proof: DRAT proof as string or path

        Returns:
            Dict containing verification results and metrics
        """
        if not self.available:
            raise RuntimeError("DRAT verifier not available")

        import time

        # Measure verification time
        start_time = time.time()
        is_valid = self.verifier.verify(cnf_formula, proof)
        verification_time = time.time() - start_time

        # Count proof steps (very simple approximation)
        proof_steps = 0
        if isinstance(proof, str):
            for line in proof.splitlines():
                if line.strip() and not line.startswith("c"):
                    proof_steps += 1

        return {
            "valid": is_valid,
            "verification_time": verification_time,
            "proof_steps": proof_steps,
        }
