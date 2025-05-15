# Proof Verification in SymbolicGym

SymbolicGym includes functionality for verifying proofs of unsatisfiability using the DRAT-trim proof checker.

## Requirements

- GCC compiler must be installed on the system
- The package must be installed with the `proof` optional dependencies

## Basic Usage

```python
import symbolicgym  # TODO: Update to actual import if available
from symbolicgym.proofs.drat import DRATVerifier  # TODO: Update to actual import if available

# Initialize the verifier
verifier = DRATVerifier()

# Example CNF formula: (a) AND (NOT a)
cnf = "p cnf 1 2\n1 0\n-1 0\n"

# Example DRAT proof
proof = "d 1 0\n"

# Verify the proof
is_valid = verifier.verify(cnf, proof)
print(f"Proof valid: {is_valid}")
```

## Integration with Environments

The proof verification can be used within environments to validate agent solutions:

```python
from symbolicgym.proofs.verification import ProofVerificationManager  # TODO: Update to actual import if available

# In your environment
verifier = ProofVerificationManager()

if verifier.is_available():
    # Agent found a proof
    if verifier.verify_solution(cnf_formula, agent_proof):
        reward += 10  # Bonus reward for correct proof
    else:
        reward -= 5   # Penalty for invalid proof
```

## Technical Details

### What is DRAT?

DRAT (Delete Resolution Asymmetric Tautology) is a proof system for showing that a Boolean formula is unsatisfiable. DRAT proofs are a sequence of clause additions and deletions that, when applied to the original formula, eventually lead to an empty clause (representing a contradiction).

### DRAT-trim

DRAT-trim is a tool developed by Marijn Heule at the University of Texas at Austin. It is a C program that:

1. Verifies DRAT proofs
2. Can trim DRAT proofs to remove unnecessary steps
3. Can produce optimized proofs

### Integration with SymbolicGym

SymbolicGym includes:

1. A Python wrapper around the DRAT-trim binary
2. Automatic compilation of the C code
3. A simple API for verifying proofs
4. Integration with the RL environment

### Error Handling

The verification system includes robust error handling:

```python
import symbolicgym  # TODO: Update to actual import if available
from symbolicgym.proofs.drat import DRATVerifier  # TODO: Update to actual import if available

try:
    verifier = DRATVerifier()
    result = verifier.verify(cnf, proof)
except FileNotFoundError:
    print("DRAT-trim source file not found")
except RuntimeError as e:
    print(f"Compilation error: {e}")
```

## Advanced Usage

### Customizing Compilation

```python
from symbolicgym.proofs.drat import DRATVerifier  # TODO: Update to actual import if available

verifier = DRATVerifier(
    source_path="/path/to/custom/drat-trim.c",
    compiler="clang",
    compiler_flags=["-O3", "-march=native"]
)
```

### Working with Files

```python
# Verify using file paths
from symbolicgym.proofs.drat import DRATVerifier  # TODO: Update to actual import if available

verifier = DRATVerifier()
result = verifier.verify("path/to/formula.cnf", "path/to/proof.drat")
```

### Integration with Training Loops

```python
# Example in a training loop
from symbolicgym.proofs.verification import ProofVerificationManager  # TODO: Update to actual import if available

verifier = ProofVerificationManager()

def custom_reward_function(env_state, action, next_state, proof=None):
    base_reward = -0.1  # Small negative reward for each step

    if next_state.is_solved:
        base_reward += 10  # Substantial reward for solving

    # If a proof was provided and verification is available
    if proof and verifier.is_available():
        if verifier.verify_solution(env_state.cnf, proof):
            base_reward += 5  # Additional reward for valid proof
        else:
            base_reward -= 2  # Penalty for invalid proof

    return base_reward
```
