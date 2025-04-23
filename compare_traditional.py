import time
from sat_problems import PROBLEM_COLLECTION

def compare_with_traditional(problems=None):
    """Compare with traditional SAT solver"""
    if problems is None:
        problems = PROBLEM_COLLECTION
    
    try:
        from pysat.solvers import Glucose3
        
        for problem in problems:
            print(f"\nTesting traditional solver on {problem['name']}")
            
            # Convert problem to dimacs format
            dimacs_clauses = problem["clauses"].copy()
            
            # Measure traditional solver performance
            start_time = time.time()
            g = Glucose3()
            for clause in dimacs_clauses:
                g.add_clause(clause)
            sat = g.solve()
            model = g.get_model() if sat else None
            trad_runtime = time.time() - start_time
            
            print(f"  Result: {'SAT' if sat else 'UNSAT'}")
            print(f"  Runtime: {trad_runtime:.6f}s")
            if model:
                assignment = {i: val > 0 for i, val in enumerate(model, 1) if abs(val) <= problem['num_vars']}
                print(f"  Solution: {assignment}")
            
    except ImportError:
        print("PySAT not installed. Use 'pip install python-sat'")

if __name__ == "__main__":
    compare_with_traditional()