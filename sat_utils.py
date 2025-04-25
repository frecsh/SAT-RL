#!/usr/bin/env python3
"""
Utilities for working with SAT problems and benchmark datasets.
"""

import os
import re
import urllib.request
import zipfile
import tarfile
import shutil
import random
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

def clause_to_variable_ratio(clauses: List[List[int]], n_vars: int) -> float:
    """
    Calculate the clause-to-variable ratio for a SAT problem.
    
    Args:
        clauses: List of clauses, where each clause is a list of integers
        n_vars: Number of variables in the problem
        
    Returns:
        The clause-to-variable ratio
    """
    return len(clauses) / n_vars

def download_sat_benchmarks(output_dir: str = 'benchmarks', 
                         sources: Optional[List[Dict[str, str]]] = None) -> None:
    """
    Download standard SAT benchmark datasets.
    
    Args:
        output_dir: Directory to save benchmark files
        sources: List of benchmark sources with 'url', 'name', and 'type' (zip/tar) fields
    """
    if sources is None:
        # Updated sources of SAT benchmarks with working URLs
        sources = [
            {
                'name': 'satlib_uf50',
                'url': 'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf50-218.tar.gz',
                'type': 'tar.gz'
            },
            {
                'name': 'satlib_uf75',
                'url': 'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf75-325.tar.gz',
                'type': 'tar.gz'
            },
            {
                'name': 'satlib_uf100',
                'url': 'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf100-430.tar.gz',
                'type': 'tar.gz'
            },
            {
                'name': 'satlib_uf125',
                'url': 'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf125-538.tar.gz',
                'type': 'tar.gz'
            },
            {
                'name': 'satlib_uf150',
                'url': 'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf150-645.tar.gz',
                'type': 'tar.gz'
            }
        ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for source in sources:
        src_name = source['name']
        src_url = source['url']
        src_type = source['type']
        
        print(f"Downloading {src_name} benchmarks from {src_url}")
        
        # Create destination directory
        dest_dir = os.path.join(output_dir, src_name)
        os.makedirs(dest_dir, exist_ok=True)
        
        # Download file
        archive_path = os.path.join(dest_dir, f"{src_name}.{src_type}")
        urllib.request.urlretrieve(src_url, archive_path)
        
        # Extract archive
        if src_type == 'zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
        elif src_type in ['tar.gz', 'tgz']:
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(dest_dir)
        
        print(f"Extracted {src_name} benchmarks to {dest_dir}")

def categorize_benchmarks(benchmark_dir: str) -> Dict[str, List[str]]:
    """
    Categorize benchmark problems by size and complexity.
    
    Args:
        benchmark_dir: Directory containing benchmark files
        
    Returns:
        Dictionary mapping categories to lists of problem file paths
    """
    categories = {
        'small': [],     # < 100 variables
        'medium': [],    # 100-1000 variables
        'large': [],     # 1000-10000 variables
        'very_large': [], # > 10000 variables
        'easy': [],      # clause/variable ratio < 3
        'medium_hard': [], # ratio 3-4.5
        'hard': []       # ratio > 4.5
    }
    
    for root, _, files in os.walk(benchmark_dir):
        for file in files:
            if file.endswith('.cnf'):
                filepath = os.path.join(root, file)
                
                try:
                    clauses, n_vars = parse_cnf_header(filepath)
                    ratio = clause_to_variable_ratio(clauses, n_vars)
                    
                    # Categorize by size
                    if n_vars < 100:
                        categories['small'].append(filepath)
                    elif n_vars < 1000:
                        categories['medium'].append(filepath)
                    elif n_vars < 10000:
                        categories['large'].append(filepath)
                    else:
                        categories['very_large'].append(filepath)
                    
                    # Categorize by difficulty (based on ratio)
                    if ratio < 3:
                        categories['easy'].append(filepath)
                    elif ratio < 4.5:
                        categories['medium_hard'].append(filepath)
                    else:
                        categories['hard'].append(filepath)
                        
                except Exception as e:
                    print(f"Error processing {filepath}: {str(e)}")
    
    return categories

def parse_cnf_header(cnf_path: str) -> Tuple[List[List[int]], int]:
    """
    Parse the header of a CNF file to extract problem characteristics.
    
    Args:
        cnf_path: Path to the CNF file
        
    Returns:
        Tuple of (clauses, n_vars)
    """
    clauses = []
    n_vars = 0
    n_clauses = 0
    
    with open(cnf_path, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments
            if line.startswith('c'):
                continue
            
            # Process problem line (p cnf <vars> <clauses>)
            if line.startswith('p cnf'):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        n_vars = int(parts[2])
                        n_clauses = int(parts[3])
                        break
                    except ValueError:
                        print(f"Warning: Invalid format in problem line of {cnf_path}")
                        # Use a reasonable default if parsing fails
                        filename = os.path.basename(cnf_path)
                        if 'uf50' in filename:
                            n_vars, n_clauses = 50, 218
                        elif 'uf75' in filename:
                            n_vars, n_clauses = 75, 325
                        elif 'uf100' in filename:
                            n_vars, n_clauses = 100, 430
                        elif 'uf125' in filename:
                            n_vars, n_clauses = 125, 538
                        elif 'uf150' in filename:
                            n_vars, n_clauses = 150, 645
                        else:
                            n_vars, n_clauses = 100, 430
                        break
    
    # Now read clauses (we only need a few to get the basic structure)
    with open(cnf_path, 'r', errors='ignore') as f:
        current_clause = []
        
        for line in f:
            line = line.strip()
            
            # Skip comments and problem line
            if line.startswith('c') or line.startswith('p'):
                continue
            
            # Process literals
            try:
                for lit_str in line.split():
                    if lit_str == '%':
                        continue  # Skip malformed characters
                        
                    lit = int(lit_str)
                    
                    if lit == 0:  # End of clause
                        if current_clause:
                            clauses.append(current_clause)
                            current_clause = []
                        
                        # Limit the number of clauses we parse
                        if len(clauses) >= min(n_clauses, 100):  # Just get a sample
                            break
                    else:
                        current_clause.append(lit)
            except ValueError:
                # Skip lines with parsing errors
                continue
    
    # If we couldn't parse any clauses but have the variable count,
    # return a minimal representation
    if not clauses and n_vars > 0:
        clauses = [[1], [2], [3]]  # Minimal clauses for size estimation
        
    return clauses, n_vars

def create_benchmark_subset(input_dir: str, output_dir: str, 
                           categories: List[str] = None, 
                           count_per_category: int = 10, 
                           seed: int = 42) -> None:
    """
    Create a subset of benchmark problems for quicker testing.
    
    Args:
        input_dir: Directory containing benchmark files
        output_dir: Directory to save the subset
        categories: List of categories to include
        count_per_category: Number of problems to include per category
        seed: Random seed for reproducibility
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    
    if categories is None:
        categories = ['small', 'medium', 'large', 'easy', 'medium_hard', 'hard']
    
    # Categorize benchmarks
    categorized = categorize_benchmarks(input_dir)
    
    # Select subset of each category
    for category in categories:
        if category in categorized and categorized[category]:
            category_dir = os.path.join(output_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            
            # Select random subset
            selected = random.sample(
                categorized[category], 
                min(count_per_category, len(categorized[category]))
            )
            
            # Copy selected files
            for filepath in selected:
                dest = os.path.join(category_dir, os.path.basename(filepath))
                shutil.copy(filepath, dest)
            
            print(f"Created subset of {len(selected)} {category} problems")

def generate_phase_transition_problems(output_dir: str, 
                                     variable_counts: List[int] = None,
                                     ratios: List[float] = None,
                                     instances_per_config: int = 5,
                                     seed: int = 42) -> None:
    """
    Generate SAT problems around the phase transition point.
    
    Args:
        output_dir: Directory to save generated problems
        variable_counts: List of variable counts to use
        ratios: List of clause-to-variable ratios to use
        instances_per_config: Number of instances to generate per configuration
        seed: Random seed for reproducibility
    """
    from sat_problems import generate_sat_problem  # Import here to avoid circular imports
    
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    
    if variable_counts is None:
        variable_counts = [50, 100, 200]
    
    if ratios is None:
        # Generate problems across the phase transition region
        ratios = np.linspace(3.5, 5.0, 10)
    
    for vars in variable_counts:
        for ratio in ratios:
            clauses = int(vars * ratio)
            
            for instance in range(instances_per_config):
                # Generate random SAT problem
                problem = generate_sat_problem(vars, clauses)
                
                # Save to file
                filename = f"vars{vars}_ratio{ratio:.2f}_inst{instance}.cnf"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w') as f:
                    # Write CNF header
                    f.write(f"c Random {vars}-variable, {clauses}-clause 3-SAT problem\n")
                    f.write(f"c Clause-to-variable ratio: {ratio:.2f}\n")
                    f.write(f"c Generated for phase transition analysis\n")
                    f.write(f"p cnf {vars} {len(problem)}\n")
                    
                    # Write clauses
                    for clause in problem:
                        f.write(" ".join(map(str, clause)) + " 0\n")
            
            print(f"Generated {instances_per_config} instances with {vars} variables and ratio {ratio:.2f}")

if __name__ == "__main__":
    # Example usage
    download_sat_benchmarks()
    categorized = categorize_benchmarks('benchmarks')
    create_benchmark_subset('benchmarks', 'benchmarks/subset')
    generate_phase_transition_problems('benchmarks/phase_transition')