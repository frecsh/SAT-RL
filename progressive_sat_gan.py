#!/usr/bin/env python3
"""
Progressive GAN training for SAT problems.
This implementation starts with simpler problems and gradually increases complexity
to improve convergence and solution quality.
"""

import os
import torch
import numpy as np
import random
import time
from typing import List, Tuple, Dict, Any, Optional, Union
import copy

from sat_problems import generate_sat_problem, is_satisfied, count_satisfied_clauses
from sat_gan import SATGAN


class ProgressiveSATGAN:
    """
    Progressive GAN for SAT problems that gradually increases problem complexity.
    Starts with a small subset of clauses and grows to the full problem.
    """
    
    def __init__(self, n_vars: int, clauses: List[List[int]], 
                 latent_dim: int = 100,
                 clause_weight: float = 1.0,
                 diversity_weight: float = 0.1,
                 learning_rate: float = 0.0002):
        """
        Initialize the Progressive SATGAN model.
        
        Args:
            n_vars: Number of variables in the SAT problem
            clauses: List of clauses, where each clause is a list of literals
            latent_dim: Dimension of the latent space
            clause_weight: Weight of clause satisfaction loss
            diversity_weight: Weight of diversity loss
            learning_rate: Learning rate for optimizers
        """
        self.n_vars = n_vars
        self.full_clauses = clauses
        self.latent_dim = latent_dim
        self.clause_weight = clause_weight
        self.diversity_weight = diversity_weight
        self.learning_rate = learning_rate
        
        # Stage parameters
        self.num_stages = 5  # Default number of stages
        self.current_stage = 0
        self.clauses_per_stage = []
        self.gan_models = []
        
        # Compute clause usage frequency for prioritizing important clauses
        self._compute_clause_importance()
        
        # Prepare progressive training stages
        self._prepare_stages()
        
        # Create first stage GAN
        self._create_stage_gan(0)
    
    def _compute_clause_importance(self) -> None:
        """
        Compute the importance of each clause based on variable overlap.
        This helps prioritize clauses that share variables with others.
        """
        n_clauses = len(self.full_clauses)
        self.clause_importance = np.zeros(n_clauses)
        
        # Create variable to clause mapping
        var_to_clauses = {}
        for i, clause in enumerate(self.full_clauses):
            for lit in clause:
                var = abs(lit)
                if var not in var_to_clauses:
                    var_to_clauses[var] = []
                var_to_clauses[var].append(i)
        
        # Calculate importance based on variable overlap
        for i, clause in enumerate(self.full_clauses):
            # Count how many other clauses share variables with this clause
            connected_clauses = set()
            for lit in clause:
                var = abs(lit)
                connected_clauses.update(var_to_clauses[var])
            
            # Remove self-connection
            if i in connected_clauses:
                connected_clauses.remove(i)
            
            # Importance is proportional to the number of connected clauses
            self.clause_importance[i] = len(connected_clauses)
        
        # Normalize importance to [0.1, 1.0]
        if max(self.clause_importance) > min(self.clause_importance):
            self.clause_importance = 0.1 + 0.9 * (self.clause_importance - min(self.clause_importance)) / (max(self.clause_importance) - min(self.clause_importance))
        else:
            self.clause_importance = np.ones_like(self.clause_importance)
    
    def _prepare_stages(self) -> None:
        """
        Prepare the progressive training stages.
        Computes which clauses to include at each stage.
        """
        n_clauses = len(self.full_clauses)
        
        # Calculate clauses per stage using exponential growth
        if n_clauses < 50:
            # For small problems, fewer stages
            self.num_stages = min(3, n_clauses)
        else:
            self.num_stages = min(5, n_clauses // 20 + 1)
            
        # Calculate number of clauses for each stage
        if self.num_stages > 1:
            # Use exponential growth
            ratio = (n_clauses / 10) ** (1 / (self.num_stages - 1))
            self.clauses_per_stage = [
                min(n_clauses, int(10 * ratio ** i))
                for i in range(self.num_stages)
            ]
            
            # Ensure the last stage has all clauses
            self.clauses_per_stage[-1] = n_clauses
        else:
            # Only one stage, use all clauses
            self.clauses_per_stage = [n_clauses]
            
        print(f"Progressive training with {self.num_stages} stages:")
        for i, num_clauses in enumerate(self.clauses_per_stage):
            print(f"  Stage {i+1}: {num_clauses} clauses ({num_clauses/n_clauses:.1%} of total)")
    
    def _select_stage_clauses(self, stage: int) -> List[List[int]]:
        """
        Select clauses for a specific training stage.
        
        Args:
            stage: Stage index
            
        Returns:
            List of clauses for this stage
        """
        n_clauses = len(self.full_clauses)
        num_clauses = self.clauses_per_stage[stage]
        
        if num_clauses >= n_clauses:
            return self.full_clauses
        
        # Select clauses using importance-weighted sampling
        indices = list(range(n_clauses))
        weights = self.clause_importance.copy()
        
        # Sample without replacement
        stage_indices = np.random.choice(
            indices, 
            size=num_clauses, 
            replace=False, 
            p=weights/weights.sum()
        )
        
        return [self.full_clauses[i] for i in stage_indices]
    
    def _create_stage_gan(self, stage: int) -> None:
        """
        Create a GAN model for the specified training stage.
        
        Args:
            stage: Stage index
        """
        # Select clauses for this stage
        stage_clauses = self._select_stage_clauses(stage)
        
        # Create GAN model
        gan = SATGAN(
            self.n_vars,
            stage_clauses,
            latent_dim=self.latent_dim,
            clause_weight=self.clause_weight,
            diversity_weight=self.diversity_weight,
            learning_rate=self.learning_rate
        )
        
        # If not the first stage, transfer weights from previous stage
        if stage > 0 and stage - 1 < len(self.gan_models):
            prev_gan = self.gan_models[stage - 1]
            gan.generator.load_state_dict(prev_gan.generator.state_dict())
            gan.discriminator.load_state_dict(prev_gan.discriminator.state_dict())
        
        # Store the model
        if stage < len(self.gan_models):
            self.gan_models[stage] = gan
        else:
            self.gan_models.append(gan)
    
    def _train_stage(self, stage_idx: int, solutions: List[List[int]]) -> None:
        """
        Train the GAN for a specific stage.
        
        Args:
            stage_idx: Index of the stage to train
            solutions: Solutions to train on for this stage
        """
        print(f"Training GAN for stage {stage_idx+1}/{len(self.clauses_per_stage)}")
        
        # Get the GAN for this stage
        gan = self.gan_models[stage_idx]
        
        # Apply stabilization measures for better numerical stability
        gan.stabilize_training()
        
        # Use smaller batch size for more stable training (reduce from original value)
        batch_size = min(8, len(solutions) // 2) if len(solutions) > 4 else 2
        
        # Train the GAN
        gan.train(
            solutions=solutions,
            batch_size=batch_size,  # Smaller batch size for stability
            epochs=self.epochs_per_stage,
            eval_interval=max(1, self.epochs_per_stage // 5)
        )
    
    def train(self, solutions: List[List[int]] = None, 
              epochs_per_stage: int = 100,
              batch_size: int = 32,
              min_solutions_per_stage: int = 50) -> None:
        """
        Train the progressive GAN through all stages.
        
        Args:
            solutions: Initial solution set (optional)
            epochs_per_stage: Number of epochs to train each stage
            batch_size: Batch size for training
            min_solutions_per_stage: Minimum number of solutions needed per stage
        """
        start_time = time.time()
        print(f"Starting progressive SATGAN training with {self.num_stages} stages")
        
        # Train each stage sequentially
        for stage in range(self.num_stages):
            self.current_stage = stage
            stage_clauses = self._select_stage_clauses(stage)
            stage_name = f"Stage {stage+1}/{self.num_stages} ({len(stage_clauses)} clauses)"
            
            print(f"\n{'-'*60}\nTraining {stage_name}\n{'-'*60}")
            
            # Create or get the GAN for this stage
            if stage >= len(self.gan_models):
                self._create_stage_gan(stage)
                
            gan = self.gan_models[stage]
            
            # Filter or generate training solutions for this stage
            stage_solutions = self._prepare_stage_solutions(
                stage, solutions, min_solutions_per_stage)
            
            print(f"Training with {len(stage_solutions)} solutions")
            
            # Train the stage GAN
            self._train_stage(stage, stage_solutions)
            
            # If this is the last stage, stop
            if stage == self.num_stages - 1:
                break
            
            # Generate solutions for the next stage using the current model
            if stage < self.num_stages - 1:
                print(f"Generating solutions for next stage...")
                next_stage_solutions = self._generate_next_stage_solutions(
                    stage, 2*min_solutions_per_stage)
                
                # Combine with existing solutions if available
                if solutions:
                    all_solutions = solutions + next_stage_solutions
                else:
                    all_solutions = next_stage_solutions
                    
                solutions = all_solutions
        
        elapsed = time.time() - start_time
        print(f"\nProgressive SATGAN training completed in {elapsed:.1f} seconds")
    
    def _prepare_stage_solutions(self, stage: int, 
                                solutions: List[List[int]], 
                                min_solutions: int) -> List[List[int]]:
        """
        Prepare solutions for a specific training stage.
        Filters existing solutions and generates new ones if needed.
        
        Args:
            stage: Stage index
            solutions: Existing solutions (may be None)
            min_solutions: Minimum number of solutions needed
            
        Returns:
            Solutions for this stage
        """
        stage_clauses = self._select_stage_clauses(stage)
        
        # If no solutions provided, generate random ones
        if not solutions:
            return self._generate_random_solutions(stage_clauses, min_solutions)
        
        # Filter existing solutions based on satisfaction ratio for this stage
        good_solutions = []
        min_satisfaction = 0.7  # Accept solutions satisfying at least 70% of clauses
        
        for sol in solutions:
            # Make sure solutions use 1/-1 format instead of potentially having other values
            normalized_sol = [1 if lit > 0 else -1 for lit in sol]
            
            satisfied = count_satisfied_clauses(stage_clauses, normalized_sol)
            ratio = satisfied / len(stage_clauses)
            
            if ratio >= min_satisfaction:
                good_solutions.append(normalized_sol)
        
        # If we have enough good solutions, use them
        if len(good_solutions) >= min_solutions:
            return good_solutions
        
        # Otherwise, generate additional solutions
        print(f"Found only {len(good_solutions)} good solutions, generating more...")
        random_solutions = self._generate_random_solutions(
            stage_clauses, min_solutions - len(good_solutions))
        
        return good_solutions + random_solutions
    
    def _generate_random_solutions(self, clauses: List[List[int]], 
                                 count: int) -> List[List[int]]:
        """
        Generate random solutions with local search improvement.
        
        Args:
            clauses: Clauses to satisfy
            count: Number of solutions to generate
            
        Returns:
            List of generated solutions
        """
        solutions = []
        
        for _ in range(count):
            # Generate random assignment (using 1/-1 format)
            assignment = [1 if random.random() < 0.5 else -1 
                         for i in range(1, self.n_vars + 1)]
            
            # Improve with local search
            for _ in range(100):  # 100 local search steps
                # Pick a random variable
                var_idx = random.randrange(self.n_vars)
                
                # Try flipping it
                old_lit = assignment[var_idx]
                new_lit = -old_lit
                
                satisfied_old = count_satisfied_clauses(clauses, assignment)
                
                # Try the flip
                assignment[var_idx] = new_lit
                satisfied_new = count_satisfied_clauses(clauses, assignment)
                
                # Keep the flip only if it's better
                if satisfied_new < satisfied_old:
                    assignment[var_idx] = old_lit
            
            solutions.append(assignment)
        
        return solutions
    
    def _generate_next_stage_solutions(self, current_stage: int, 
                                    count: int) -> List[List[int]]:
        """
        Generate solutions for the next stage using the current stage's GAN.
        
        Args:
            current_stage: Current stage index
            count: Number of solutions to generate
            
        Returns:
            Generated solutions
        """
        # Get the current GAN
        gan = self.gan_models[current_stage]
        
        # Get the clauses for the next stage
        next_stage = min(current_stage + 1, self.num_stages - 1)
        next_stage_clauses = self._select_stage_clauses(next_stage)
        
        # Generate more solutions than needed to allow filtering
        oversampling = 3
        candidates = gan.generate(num_samples=count * oversampling, temperature=1.0)
        
        # Score candidates on next stage clauses
        scored_candidates = []
        for sol in candidates:
            satisfied = count_satisfied_clauses(next_stage_clauses, sol)
            ratio = satisfied / len(next_stage_clauses)
            scored_candidates.append((sol, ratio))
        
        # Sort by satisfaction ratio
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Take the best ones
        return [sol for sol, _ in scored_candidates[:count]]
    
    def generate(self, num_samples: int = 10, 
               temperature: float = 1.0) -> List[List[int]]:
        """
        Generate variable assignments using the final stage GAN.
        
        Args:
            num_samples: Number of samples to generate
            temperature: Temperature for sampling
            
        Returns:
            Generated variable assignments
        """
        # Use the final stage GAN
        if not self.gan_models or self.current_stage >= len(self.gan_models):
            raise ValueError("Model has not been trained yet")
            
        final_gan = self.gan_models[self.current_stage]
        return final_gan.generate(num_samples, temperature)
    
    def save(self, path: str) -> None:
        """
        Save the progressive GAN model.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model data
        data = {
            'n_vars': self.n_vars,
            'latent_dim': self.latent_dim,
            'clause_weight': self.clause_weight,
            'diversity_weight': self.diversity_weight,
            'num_stages': self.num_stages,
            'current_stage': self.current_stage,
            'clauses_per_stage': self.clauses_per_stage,
            'clause_importance': self.clause_importance,
            'final_generator_state': self.gan_models[-1].generator.state_dict()
            if self.gan_models else None
        }
        
        torch.save(data, path)
        print(f"Progressive SATGAN model saved to {path}")
    
    @classmethod
    def load(cls, path: str, clauses: List[List[int]]) -> "ProgressiveSATGAN":
        """
        Load a progressive GAN model.
        
        Args:
            path: Path to the saved model
            clauses: Full set of clauses
            
        Returns:
            Loaded progressive GAN model
        """
        # Load model data
        data = torch.load(path, map_location=torch.device('cpu'))
        
        # Create model
        model = cls(
            data['n_vars'],
            clauses,
            latent_dim=data['latent_dim'],
            clause_weight=data['clause_weight'],
            diversity_weight=data['diversity_weight']
        )
        
        # Restore stage info
        model.num_stages = data['num_stages']
        model.current_stage = data['current_stage']
        model.clauses_per_stage = data['clauses_per_stage']
        model.clause_importance = data['clause_importance']
        
        # Create final stage GAN
        model._create_stage_gan(model.current_stage)
        
        # Load generator state
        if data['final_generator_state'] is not None:
            model.gan_models[-1].generator.load_state_dict(data['final_generator_state'])
        
        return model


# Example usage
if __name__ == "__main__":
    from sat_problems import generate_sat_problem
    
    # Generate a moderately challenging SAT problem
    n_vars = 50
    n_clauses = 200
    print(f"Generating SAT problem with {n_vars} variables and {n_clauses} clauses...")
    clauses = generate_sat_problem(n_vars, n_clauses)
    
    # Create and train the progressive GAN
    print("Creating progressive GAN...")
    prog_gan = ProgressiveSATGAN(n_vars, clauses, latent_dim=50)
    
    # Train the model
    print("Training progressive GAN...")
    prog_gan.train(epochs_per_stage=50, batch_size=32)
    
    # Generate solutions
    print("\nGenerating solutions...")
    solutions = prog_gan.generate(num_samples=20)
    
    # Evaluate solutions
    satisfaction_scores = []
    for i, sol in enumerate(solutions):
        satisfied = count_satisfied_clauses(clauses, sol)
        ratio = satisfied / n_clauses
        satisfaction_scores.append(ratio)
        
        if i < 5:  # Print first 5 solutions
            print(f"Solution {i+1}: {ratio:.2%} satisfied")
    
    # Print overall statistics
    print(f"\nAverage satisfaction: {sum(satisfaction_scores)/len(satisfaction_scores):.2%}")
    print(f"Max satisfaction: {max(satisfaction_scores):.2%}")
    print(f"Solutions that fully satisfy the problem: {sum(1 for s in satisfaction_scores if s == 1.0)}")
    
    # Save the model
    prog_gan.save("models/progressive_sat_gan.pth")