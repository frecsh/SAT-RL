import hashlib
import multiprocessing
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _verify_single_proof(args):
    """
    Helper function for parallel proof verification.
    Must be at module level to be picklable.

    Args:
        args: Tuple containing (cnf_formula, proof, use_cache)

    Returns:
        bool: True if the proof is valid, False otherwise
    """
    cnf, proof, use_cache = args
    # Import here to avoid circular imports
    from satrlgym.proofs.drat import DRATVerifier

    verifier = DRATVerifier(use_cache=use_cache)
    return verifier.verify(cnf, proof)


class DRATVerifier:
    """Python wrapper for DRAT-trim proof verifier"""

    # Class-level cache for results
    _verification_cache: dict[str, bool] = {}
    _max_cache_size = 1000

    def __init__(self, debug=False, use_cache=True, max_cache_size=1000):
        """Initialize the DRAT verifier

        Args:
            debug: If True, print debug information during verification
            use_cache: Whether to cache verification results
            max_cache_size: Maximum number of entries in the cache
        """
        # Find the executable (should be in the same directory as this file)
        base_dir = Path(__file__).parent
        self.executable = str(base_dir / "drat-trim")
        self.debug = debug
        self.use_cache = use_cache
        DRATVerifier._max_cache_size = max_cache_size

        # Compile if not already compiled
        if not os.path.exists(self.executable):
            self._compile()

    def _compile(self):
        """Compile the drat-trim executable"""
        source = Path(__file__).parent / "drat-trim.c"
        if not os.path.exists(source):
            raise FileNotFoundError(f"drat-trim.c not found at {source}")

        result = subprocess.run(
            ["gcc", "-O2", str(source), "-o", self.executable], capture_output=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to compile drat-trim: {result.stderr.decode()}")

    def _cache_key(self, cnf_formula: str | bytes, proof: str | bytes) -> str:
        """Generate a cache key for the CNF formula and proof combination"""
        # Convert to bytes if necessary
        if isinstance(cnf_formula, str):
            cnf_bytes = cnf_formula.encode("utf-8")
        else:
            cnf_bytes = cnf_formula

        if isinstance(proof, str):
            proof_bytes = proof.encode("utf-8")
        else:
            proof_bytes = proof

        # Compute hash
        hash_obj = hashlib.sha256()
        hash_obj.update(cnf_bytes)
        hash_obj.update(b"::")  # separator
        hash_obj.update(proof_bytes)

        return hash_obj.hexdigest()

    def _verify_uncached(self, cnf_formula, proof, options=None):
        """Verify a DRAT proof without using the cache"""
        # Handle file or string inputs
        with tempfile.NamedTemporaryFile("w", suffix=".cnf", delete=False) as cnf_file:
            if os.path.exists(str(cnf_formula)):
                cnf_file.write(open(cnf_formula).read())
            else:
                cnf_file.write(cnf_formula)
            cnf_path = cnf_file.name

        with tempfile.NamedTemporaryFile(
            "w", suffix=".drat", delete=False
        ) as proof_file:
            if os.path.exists(str(proof)):
                proof_file.write(open(proof).read())
            else:
                proof_file.write(proof)
            proof_path = proof_file.name

        try:
            # First, manually check for invalid literals in the proof
            # Parse CNF to get maximum variable number
            max_var = 0
            for line in str(cnf_formula).splitlines():
                line = line.strip()
                if line.startswith("p") or line.startswith("c") or not line:
                    continue
                for lit in line.split():
                    if lit == "0":
                        continue
                    try:
                        var = abs(int(lit))
                        max_var = max(max_var, var)
                    except ValueError:
                        continue

            # Now check if the proof tries to delete non-existent literals
            for line in str(proof).splitlines():
                line = line.strip()
                if not line or line.startswith("c"):
                    continue
                parts = line.split()
                if parts and parts[0] == "d":  # deletion step
                    for lit in parts[1:]:
                        if lit == "0":
                            continue
                        try:
                            var = abs(int(lit))
                            if var > max_var:
                                return False  # Invalid: trying to delete non-existent literal
                        except ValueError:
                            continue

            cmd = [self.executable, cnf_path, proof_path]

            # Add verbose mode for debugging
            if self.debug:
                cmd.append("-v")

            if options:
                cmd.extend(options)

            result = subprocess.run(cmd, capture_output=True, text=True)

            if self.debug:
                print(f"Command: {' '.join(cmd)}")
                print(f"CNF content: {cnf_formula}")
                print(f"Proof content: {proof}")
                print(f"DRAT-trim stdout: {result.stdout}")
                print(f"DRAT-trim stderr: {result.stderr}")
                print(f"Return code: {result.returncode}")

            # For invalid proofs, drat-trim usually returns a non-zero exit code
            if result.returncode != 0:
                return False

            # For valid proofs, drat-trim should output a verification message
            # The exact format can vary between versions, so check multiple patterns
            verification_markers = ["s VERIFIED", "VERIFIED", "verified", "SUCCESS"]

            for marker in verification_markers:
                if marker in result.stdout:
                    return True

            # If we get here, the proof wasn't explicitly verified
            return False

        finally:
            # Clean up temporary files
            try:
                os.unlink(cnf_path)
                os.unlink(proof_path)
            except BaseException:
                pass

    def verify(self, cnf_formula, proof, options=None):
        """Verify a DRAT proof, with optional caching"""
        # Use caching if enabled
        if self.use_cache:
            cache_key = self._cache_key(cnf_formula, proof)

            # Check if result is in cache
            if cache_key in DRATVerifier._verification_cache:
                if self.debug:
                    print(f"Cache hit for key: {cache_key}")
                return DRATVerifier._verification_cache[cache_key]

            # Verify and store in cache
            result = self._verify_uncached(cnf_formula, proof, options)

            # Add to cache, maintaining size limit
            if len(DRATVerifier._verification_cache) >= DRATVerifier._max_cache_size:
                # Remove a random entry when cache is full (simple strategy)
                try:
                    DRATVerifier._verification_cache.pop(
                        next(iter(DRATVerifier._verification_cache))
                    )
                except BaseException:
                    pass

            DRATVerifier._verification_cache[cache_key] = result
            return result
        else:
            # No caching
            return self._verify_uncached(cnf_formula, proof, options)

    @staticmethod
    def verify_batch(
        cnf_proofs: list[tuple[str, str]],
        parallel: bool = True,
        num_workers: int | None = None,
        use_cache: bool = True,
    ) -> list[bool]:
        """
        Verify multiple proofs in parallel

        Args:
            cnf_proofs: List of (cnf_formula, proof) tuples
            parallel: Whether to use parallel processing
            num_workers: Number of worker processes/threads (defaults to CPU count)
            use_cache: Whether to use caching

        Returns:
            List[bool]: List of verification results, True for valid proofs
        """
        if not parallel:
            # Serial processing
            verifier = DRATVerifier(use_cache=use_cache)
            return [verifier.verify(cnf, proof) for cnf, proof in cnf_proofs]

        if num_workers is None:
            num_workers = multiprocessing.cpu_count()

        # Add use_cache to each tuple for the worker function
        worker_args = [(cnf, proof, use_cache) for cnf, proof in cnf_proofs]

        # Use ProcessPoolExecutor for true parallelism (slower startup but better
        # for CPU-bound tasks)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_verify_single_proof, worker_args))

        return results

    @staticmethod
    def clear_cache():
        """Clear the verification cache"""
        DRATVerifier._verification_cache.clear()
