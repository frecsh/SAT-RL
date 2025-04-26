#!/usr/bin/env python3
"""
Demonstration of SAT+RL enhancements.
This script shows how to use the improved SAT solving techniques.
"""

import numpy as np
import random
import time
import argparse
from deep_q_sat_agent import DeepQLearningAgent
from improved_sat_gan import ImprovedSATGAN
from oracle_distillation_agent import OracleDistillationAgent
from curriculum_sat_learner import CurriculumSATLearner
from anytime_sat_solver import AnytimeSATSolver, AnytimeEnsembleSolver


def generate_random_sat_problem(n_vars, ratio, unique_clauses=True):
    """Generate a random 3-SAT problem with given clause-to-variable ratio"""
    n_clauses = int(ratio * n_vars)
    clauses = []
    clause_set = set()
    
    while len(clauses) < n_clauses:
        # Generate a random clause with 3 literals (3-SAT)
        vars_in_clause = random.sample(range(1, n_vars + 1), 3)
        
        # Randomly negate some variables
        clause = [var if random.random() > 0.5 else -var for var in vars_in_clause]
        
        # Convert to tuple for set operations
        clause_tuple = tuple(sorted(clause, key=abs))
        
        # Add clause if unique or if not enforcing uniqueness
        if not unique_clauses or clause_tuple not in clause_set:
            clauses.append(clause)
            if unique_clauses:
                clause_set.add(clause_tuple)
    
    return clauses


def run_deep_q_learning(n_vars, clauses, max_episodes=500):
    """Run the Deep Q-Learning approach"""
    print("\n=== Deep Q-Learning Agent ===")
    start_time = time.time()
    
    agent = DeepQLearningAgent(n_vars, clauses)
    solution, stats = agent.solve(max_episodes=max_episodes)
    
    elapsed = time.time() - start_time
    
    print(f"Deep Q-Learning completed in {elapsed:.2f} seconds")
    print(f"Episodes: {stats['episodes']}")
    print(f"Best satisfied: {stats['best_satisfied']}/{len(clauses)}")
    
    return solution, stats


def run_improved_gan(n_vars, clauses, epochs=100):
    """Run the Improved SATGAN approach with experience replay"""
    print("\n=== Improved SATGAN with Experience Replay ===")
    start_time = time.time()
    
    gan = ImprovedSATGAN(n_vars, clauses, epochs=epochs)
    gan.train_with_experience_replay()
    solution = gan.solve(max_generations=50)
    
    elapsed = time.time() - start_time
    
    # Count satisfied clauses
    satisfied = 0
    for clause in clauses:
        for literal in clause:
            var = abs(literal) - 1  # Convert to 0-indexed
            val = solution[var]
            if (literal > 0 and val == 1) or (literal < 0 and val == 0):
                satisfied += 1
                break
    
    print(f"Improved SATGAN completed in {elapsed:.2f} seconds")
    print(f"Satisfied: {satisfied}/{len(clauses)}")
    
    return solution, {'satisfied': satisfied}


def run_oracle_distillation(n_vars, clauses):
    """Run the Oracle Distillation approach"""
    print("\n=== Oracle Distillation Agent ===")
    start_time = time.time()
    
    agent = OracleDistillationAgent(n_vars, clauses)
    solution, satisfied = agent.solve()
    
    elapsed = time.time() - start_time
    
    print(f"Oracle Distillation completed in {elapsed:.2f} seconds")
    print(f"Satisfied: {satisfied}/{len(clauses)}")
    
    return solution, {'satisfied': satisfied}


def run_curriculum_learning(n_vars, target_ratio=4.2):
    """Run the Curriculum Learning approach"""
    print("\n=== Curriculum Learning ===")
    start_time = time.time()
    
    learner = CurriculumSATLearner(n_vars, initial_ratio=3.0, target_ratio=target_ratio)
    solution, best_satisfied = learner.solve_with_curriculum()
    
    elapsed = time.time() - start_time
    
    print(f"Curriculum Learning completed in {elapsed:.2f} seconds")
    print(f"Final satisfied: {best_satisfied}/{len(learner.target_clauses)}")
    
    return solution, {'satisfied': best_satisfied}


def run_anytime_solver(n_vars, clauses, time_limit=30):
    """Run the Anytime SAT Solver"""
    print("\n=== Anytime SAT Solver ===")
    
    solver = AnytimeSATSolver(n_vars, clauses, time_limit=time_limit)
    solution, stats = solver.solve_with_local_search(max_iterations=100000)
    
    print("\nGenerating visualization...")
    solver.visualize_progress()
    
    return solution, stats


def run_ensemble_solver(n_vars, clauses, time_limit=30):
    """Run the Ensemble Anytime Solver"""
    print("\n=== Ensemble Anytime Solver ===")
    
    solver = AnytimeEnsembleSolver(n_vars, clauses, time_limit=time_limit)
    solution, stats = solver.solve()
    
    print("\nGenerating visualization...")
    solver.visualize_progress()
    
    return solution, stats


def main():
    parser = argparse.ArgumentParser(description="SAT+RL Enhanced Demonstration")
    parser.add_argument("--n_vars", type=int, default=20, help="Number of variables")
    parser.add_argument("--ratio", type=float, default=4.2, help="Clause-to-variable ratio")
    parser.add_argument("--method", type=str, 
                        choices=['dqn', 'gan', 'oracle', 'curriculum', 'anytime', 'ensemble', 'all'],
                        default='all', help="Method to run")
    parser.add_argument("--time_limit", type=int, default=30, 
                        help="Time limit for anytime solvers (seconds)")
    
    args = parser.parse_args()
    
    # Generate problem
    print(f"Generating 3-SAT problem with {args.n_vars} variables and ratio {args.ratio}")
    clauses = generate_random_sat_problem(args.n_vars, args.ratio)
    print(f"Generated {len(clauses)} clauses")
    
    # Run selected method(s)
    if args.method == 'dqn' or args.method == 'all':
        run_deep_q_learning(args.n_vars, clauses)
    
    if args.method == 'gan' or args.method == 'all':
        run_improved_gan(args.n_vars, clauses)
    
    if args.method == 'oracle' or args.method == 'all':
        run_oracle_distillation(args.n_vars, clauses)
    
    if args.method == 'curriculum' or args.method == 'all':
        run_curriculum_learning(args.n_vars, args.ratio)
    
    if args.method == 'anytime' or args.method == 'all':
        run_anytime_solver(args.n_vars, clauses, args.time_limit)
    
    if args.method == 'ensemble' or args.method == 'all':
        run_ensemble_solver(args.n_vars, clauses, args.time_limit)
    

if __name__ == "__main__":
    main()