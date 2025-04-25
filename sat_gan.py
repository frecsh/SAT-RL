#!/usr/bin/env python3
"""
GAN-based generator for SAT variable assignments.
This module implements a Generative Adversarial Network that learns from
successful SAT solutions to generate promising variable assignments,
replacing random exploration with learned patterns.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # Added missing import
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
import random
import time
import copy
from sat_problems import is_satisfied, count_satisfied_clauses

class SATSolutionDataset(Dataset):
    """
    Dataset class for SAT solutions used in SATGAN training.
    Transforms SAT variable assignments (literals) into tensor format.
    """
    
    def __init__(self, n_vars: int, solutions: List[List[int]] = None):
        """
        Initialize the dataset with SAT solutions.
        
        Args:
            n_vars: Number of variables in the SAT problem
            solutions: List of variable assignments (each a list of literals)
        """
        self.n_vars = n_vars
        self.data = []
        
        if solutions:
            self.add_solutions(solutions)
    
    def __len__(self) -> int:
        """Return the number of solutions in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a solution by index.
        
        Args:
            idx: Index of the solution
            
        Returns:
            Tensor representation of the solution
        """
        return self.data[idx]
    
    def add_solution(self, solution: List[int]) -> None:
        """
        Add a single solution to the dataset.
        
        Args:
            solution: Variable assignment as a list of literals
        """
        tensor_solution = torch.zeros(self.n_vars, dtype=torch.float32)
        
        for literal in solution:
            var_idx = abs(literal) - 1  # Convert to 0-indexed
            if var_idx < self.n_vars:  # Ensure variable is within range
                tensor_solution[var_idx] = 1.0 if literal > 0 else -1.0
        
        self.data.append(tensor_solution)
    
    def add_solutions(self, solutions: List[List[int]]) -> None:
        """
        Add multiple solutions to the dataset.
        
        Args:
            solutions: List of variable assignments
        """
        for solution in solutions:
            self.add_solution(solution)

class SATGenerator(nn.Module):
    """
    Generator network for SATGAN.
    Takes random noise as input and produces variable assignments.
    
    This version is designed for improved numerical stability during training.
    """
    
    def __init__(self, latent_dim: int, n_vars: int, hidden_dims: List[int] = [128, 256, 128]):
        """
        Initialize the generator network.
        
        Args:
            latent_dim: Dimension of the latent space
            n_vars: Number of variables in the SAT problem
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        # Store dimensions
        self.latent_dim = latent_dim
        self.n_vars = n_vars
        self.hidden_dims = hidden_dims
        
        # Build model layers separately for more control
        self.input_layer = nn.Linear(latent_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], n_vars)
        
        # Initialize weights carefully
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better numerical stability"""
        for layer in [self.input_layer, self.output_layer] + list(self.hidden_layers):
            # Use Xavier/Glorot initialization for weights
            nn.init.xavier_uniform_(layer.weight, gain=1.0)
            # Initialize biases to zeros
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.
        
        Args:
            z: Latent vector (batch_size, latent_dim)
            
        Returns:
            Generated variable assignments in [-1, 1] range (batch_size, n_vars)
        """
        if z.dtype != torch.float32:
            z = z.float()
            
        # Input normalization
        z = torch.clamp(z, -2.0, 2.0)
        
        # Apply input layer with LeakyReLU
        x = F.leaky_relu(self.input_layer(z), negative_slope=0.2)
        
        # Apply hidden layers
        for i, layer in enumerate(self.hidden_layers):
            x = F.leaky_relu(layer(x), negative_slope=0.2)
            # Add layer normalization for stability
            if x.size(0) > 1:  # Only apply if batch size > 1
                x = F.layer_norm(x, x.size()[1:])
        
        # Output layer with tanh activation
        x = self.output_layer(x)
        x = torch.tanh(x)
        
        return x


class SATDiscriminator(nn.Module):
    """
    Discriminator network for SATGAN.
    Takes variable assignments as input and outputs a probability
    that the assignment is real (from training data) vs fake (from generator).
    
    This version is specifically designed for maximum numerical stability
    to prevent NaN issues during training.
    """
    
    def __init__(self, n_vars: int, hidden_dims: List[int] = [256, 128, 64]):
        """
        Initialize the discriminator network.
        
        Args:
            n_vars: Number of variables in the SAT problem
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        # Reduced complexity for stability
        self.hidden_dims = hidden_dims
        self.n_vars = n_vars
        
        # Build model layers separately for more control
        self.input_layer = nn.Linear(n_vars, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            
        # Output layer with no activation (BCEWithLogitsLoss will apply sigmoid)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        # Initialize weights carefully
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better numerical stability"""
        for layer in [self.input_layer, self.output_layer] + list(self.hidden_layers):
            # Use Xavier/Glorot initialization for weights
            nn.init.xavier_uniform_(layer.weight, gain=0.5)  # gain < 1 for more stability
            # Initialize biases to small negative values to prevent saturation
            if layer.bias is not None:
                nn.init.constant_(layer.bias, -0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.
        
        Args:
            x: Variable assignment tensor (batch_size, n_vars)
            
        Returns:
            Raw logits for real/fake classification (batch_size, 1)
        """
        if x.dtype != torch.float32:
            x = x.float()
        
        # Strict input normalization
        x = torch.clamp(x, -1.0, 1.0)
        
        # Apply input layer with LeakyReLU
        x = F.leaky_relu(self.input_layer(x), negative_slope=0.1)
        
        # Apply dropout for regularization
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Apply hidden layers with regularization
        for layer in self.hidden_layers:
            x = F.leaky_relu(layer(x), negative_slope=0.1)
            x = F.dropout(x, p=0.2, training=self.training)
            
            # Add layer normalization for stability
            x = F.layer_norm(x, x.size()[1:])
            
        # Output layer with no activation
        # We'll keep outputs in a controlled range
        x = self.output_layer(x)
        x = torch.clamp(x, -10.0, 10.0)  # Prevent extreme output values
        
        return x


class ClauseSatisfactionLoss(nn.Module):
    """
    Custom loss function that penalizes the generator for producing
    variable assignments that don't satisfy many clauses.
    Optimized for speed with large SAT instances.
    """
    
    def __init__(self, clauses: List[List[int]], weight: float = 1.0):
        """
        Initialize the clause satisfaction loss.
        
        Args:
            clauses: List of clauses, where each clause is a list of literals
            weight: Weight of the clause satisfaction loss relative to GAN loss
        """
        super().__init__()
        self.weight = weight
        
        # Find max variable index for tensor representation
        max_var = 0
        for clause in clauses:
            for lit in clause:
                max_var = max(max_var, abs(lit))
        
        # Create sparse tensor representations for faster computation
        clause_indices = []
        clause_values = []
        
        for clause_idx, clause in enumerate(clauses):
            for lit in clause:
                var_idx = abs(lit) - 1  # Convert to 0-indexed
                sign = 1 if lit > 0 else -1
                clause_indices.append((clause_idx, var_idx))
                clause_values.append(sign)
        
        # Store clause info as sparse tensor
        if clause_indices:
            indices = torch.tensor(clause_indices, dtype=torch.long).t()
            values = torch.tensor(clause_values, dtype=torch.float)
            size = (len(clauses), max_var)
            self.clause_tensor = torch.sparse.FloatTensor(indices, values, size)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.clause_tensor = self.clause_tensor.cuda()
        else:
            self.clause_tensor = None
        
        # Cache constants
        self.n_clauses = len(clauses)
    
    def forward(self, assignments: torch.Tensor) -> torch.Tensor:
        """
        Calculate the clause satisfaction loss for a batch of assignments.
        
        Args:
            assignments: Variable assignments tensor (batch_size, n_vars)
            
        Returns:
            Loss value penalizing unsatisfied clauses
        """
        if self.clause_tensor is None:
            return torch.tensor(0.0, device=assignments.device)
        
        batch_size = assignments.size(0)
        
        # Compute satisfaction for all clauses in parallel using sparse operations
        # For each batch item, we need to compute which clauses are satisfied
        
        # Create a list to store batch results
        batch_satisfaction = []
        
        for i in range(batch_size):
            # Get the current assignment
            assignment = assignments[i]
            
            # Multiply assignment by clause tensor to get literal satisfaction values
            # This sets positive values for satisfied literals
            lit_values = torch.sparse.mm(self.clause_tensor, assignment.unsqueeze(1)).squeeze()
            
            # A clause is satisfied if any literal is satisfied (value > 0)
            clause_satisfied = (lit_values > 0).float()
            
            # Compute the ratio of satisfied clauses
            satisfaction_ratio = clause_satisfied.sum() / self.n_clauses
            batch_satisfaction.append(satisfaction_ratio)
        
        # Average satisfaction ratio across the batch
        avg_satisfaction = torch.stack(batch_satisfaction).mean()
        
        # Loss is inversely proportional to satisfaction
        loss = self.weight * (1.0 - avg_satisfaction)
        
        return loss


class DiversityLoss(nn.Module):
    """
    Loss function that encourages diversity in generated samples.
    Uses mini-batch discrimination to penalize lack of diversity.
    """
    
    def __init__(self, weight: float = 0.1):
        """
        Initialize the diversity loss.
        
        Args:
            weight: Weight of the diversity loss
        """
        super().__init__()
        self.weight = weight
        
    def forward(self, assignments: torch.Tensor) -> torch.Tensor:
        """
        Calculate the diversity loss for a batch of assignments.
        Penalizes samples that are too similar to each other.
        
        Args:
            assignments: Variable assignments tensor (batch_size, n_vars)
            
        Returns:
            Loss value penalizing lack of diversity
        """
        batch_size = assignments.size(0)
        
        # If batch size is too small, return zero loss
        if batch_size < 2:
            return torch.tensor(0.0, device=assignments.device)
        
        # Calculate pairwise distances between samples
        # We use L1 distance as a measure of similarity
        pairwise_distances = torch.cdist(assignments, assignments, p=1)
        
        # Set diagonal to infinity to exclude self-comparisons
        mask = torch.eye(batch_size, device=assignments.device) * float('inf')
        masked_distances = pairwise_distances + mask
        
        # Find the minimum distance for each sample to any other sample
        min_distances = torch.min(masked_distances, dim=1)[0]
        
        # The diversity loss is inversely proportional to the minimum distances
        # We want to maximize the minimum distance between samples
        # Using a softplus to ensure numerical stability
        diversity_loss = torch.mean(torch.exp(-min_distances / assignments.size(1)))
        
        return self.weight * diversity_loss


class SATGAN:
    """
    Generative Adversarial Network for generating SAT variable assignments.
    """
    
    def __init__(self, n_vars: int, clauses: List[List[int]], 
                 latent_dim: int = 100, 
                 gen_hidden_dims: List[int] = [128, 256, 512],
                 disc_hidden_dims: List[int] = [512, 256, 128],
                 clause_weight: float = 1.0,
                 diversity_weight: float = 0.1,
                 learning_rate: float = 0.0001,
                 gp_weight: float = 10.0,  # Weight for gradient penalty
                 betas: Tuple[float, float] = (0.5, 0.999)):
        """
        Initialize the SATGAN model.
        
        Args:
            n_vars: Number of variables in the SAT problem
            clauses: List of clauses, where each clause is a list of literals
            latent_dim: Dimension of the latent space
            gen_hidden_dims: List of hidden layer dimensions for generator
            disc_hidden_dims: List of hidden layer dimensions for discriminator
            clause_weight: Weight of the clause satisfaction loss
            diversity_weight: Weight of the diversity loss
            learning_rate: Learning rate for Adam optimizer
            gp_weight: Weight of the gradient penalty for Wasserstein GAN with gradient penalty
            betas: Betas for Adam optimizer
        """
        self.n_vars = n_vars
        self.clauses = clauses
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight  # Store gradient penalty weight
        
        # Store the max variable index from clauses for tensor validation
        self.max_var_in_clauses = 0
        for clause in clauses:
            for lit in clause:
                self.max_var_in_clauses = max(self.max_var_in_clauses, abs(lit))
        
        # Ensure n_vars is at least as large as the max variable in clauses
        if self.max_var_in_clauses > n_vars:
            print(f"Warning: n_vars ({n_vars}) is less than the maximum variable in clauses ({self.max_var_in_clauses})")
            print(f"Adjusting n_vars to {self.max_var_in_clauses}")
            self.n_vars = self.max_var_in_clauses
        
        # Create generator and discriminator networks
        self.generator = SATGenerator(latent_dim, self.n_vars, gen_hidden_dims)
        self.discriminator = SATDiscriminator(self.n_vars, disc_hidden_dims)
        
        # Initialize weights for better training stability
        self._init_weights(self.generator)
        self._init_weights(self.discriminator)
        
        # Create clause satisfaction loss
        self.clause_loss = ClauseSatisfactionLoss(clauses, clause_weight)
        
        # Create diversity loss
        self.diversity_loss = DiversityLoss(diversity_weight)
        
        # Move to GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Using MSE loss instead of BCE for better numerical stability
        self.criterion = torch.nn.MSELoss()
        
        # Optimizers with reduced learning rate for stability
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=betas)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=betas)
        
        # For gradient clipping to prevent NaNs
        self.max_grad_norm = 0.5
        
        # Training statistics
        self.stats = {
            'g_loss': [],
            'd_loss': [],
            'clause_sat': []
        }
    
    def _init_weights(self, model):
        """Initialize network weights properly for better training stability"""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Kaiming initialization for better stability of deep networks
                nn.init.kaiming_normal_(module.weight.data, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0.0)
    
    def compute_gradient_penalty(self, real_data, fake_data):
        """
        Compute gradient penalty for improved WGAN training stability.
        This penalizes the discriminator if its gradients are too large,
        which helps prevent the NaN issues during training.
        
        Args:
            real_data: Real data samples
            fake_data: Generated fake data samples
            
        Returns:
            Gradient penalty loss term
        """
        batch_size = real_data.size(0)
        
        # Create random interpolation between real and fake data
        alpha = torch.rand(batch_size, 1, device=self.device)
        alpha = alpha.expand(batch_size, real_data.size(1))
        
        # Create interpolated samples
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated = interpolated.detach()
        interpolated.requires_grad_(True)
        
        # Calculate discriminator output for interpolated samples
        disc_interpolates = self.discriminator(interpolated)
        
        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolated,
            grad_outputs=torch.ones_like(disc_interpolates, device=self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Calculate gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
    
    def stabilize_training(self):
        """
        Reset and stabilize the training process when numerical issues are detected.
        Call this method when encountering NaN values or unstable training.
        """
        print("Applying GAN stabilization measures...")
        
        # 1. Re-initialize model weights with more conservative values
        for module in self.discriminator.modules():
            if isinstance(module, nn.Linear):
                # Very conservative initialization
                nn.init.normal_(module.weight.data, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias.data)
        
        # 2. Reduce learning rates
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] *= 0.1
            print(f"Reduced discriminator learning rate to {param_group['lr']}")
            
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] *= 0.5
            print(f"Reduced generator learning rate to {param_group['lr']}")
        
        # 3. Increase gradient clipping threshold for smoother updates
        self.max_grad_norm *= 0.5
        print(f"Reduced gradient clipping threshold to {self.max_grad_norm}")
        
        return self
    
    def _generate_initial_data(self, num_samples: int = 100) -> List[List[int]]:
        """
        Generate initial data when no solutions are provided.
        Uses random assignments with a bias toward satisfying clauses.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of generated variable assignments
        """
        print(f"Generating {num_samples} initial solutions randomly")
        solutions = []
        
        for _ in range(num_samples):
            # Generate a random solution
            assignment = []
            for var_idx in range(1, self.n_vars + 1):
                # 50% chance of positive or negative
                if random.random() < 0.5:
                    assignment.append(var_idx)
                else:
                    assignment.append(-var_idx)
            solutions.append(assignment)
                
        return solutions
    
    def _preprocess_solutions(self, solutions: List[List[int]]) -> torch.Tensor:
        """
        Preprocess solutions into a tensor format suitable for training.
        
        Args:
            solutions: List of variable assignments as lists of literals
            
        Returns:
            Tensor representation of the solutions
        """
        tensor_data = []
        
        for solution in solutions:
            # Initialize tensor with zeros for all variables
            tensor_solution = np.zeros(self.n_vars, dtype=np.float32)
            
            for literal in solution:
                var_idx = abs(literal) - 1  # Convert to 0-indexed
                if var_idx < self.n_vars:  # Ensure variable is within range
                    tensor_solution[var_idx] = 1.0 if literal > 0 else -1.0
            
            tensor_data.append(tensor_solution)
        
        # Debug output
        print(f"Preprocessing {len(solutions)} solutions to tensor with {self.n_vars} variables")
        
        # Convert to tensor
        tensor = torch.tensor(tensor_data, dtype=torch.float32).to(self.device)
        
        # Verify tensor dimensions are correct
        if tensor.size(1) != self.n_vars:
            print(f"Warning: Tensor dimension mismatch. Expected {self.n_vars} variables, got {tensor.size(1)}")
            # Pad or truncate tensor to match expected dimensions
            if tensor.size(1) < self.n_vars:
                padding = torch.zeros(tensor.size(0), self.n_vars - tensor.size(1), device=self.device)
                tensor = torch.cat([tensor, padding], dim=1)
            else:
                tensor = tensor[:, :self.n_vars]
            print(f"Adjusted tensor dimensions to {tensor.shape}")
        
        return tensor
    
    def train(self, solutions=None, batch_size=16, epochs=50, eval_interval=10, early_stopping_patience=None):
        """
        Train the GAN model on SAT solutions
        
        Args:
            solutions: List of SAT solutions to train on
            batch_size: Size of batches for training
            epochs: Number of epochs to train for
            eval_interval: How often to evaluate the model during training (every N epochs)
            early_stopping_patience: Stop training if no improvement for this many evaluations
        """
        # Set models to training mode
        self.generator.train()
        self.discriminator.train()
        
        # Handle case where no solutions are provided
        if solutions is None or len(solutions) == 0:
            solutions = self._generate_initial_data(batch_size * 2)
        
        # Preprocess solutions for training with explicit float conversion
        train_data = self._preprocess_solutions(solutions)
        
        # Add debug print to verify tensor dtype
        print(f"Train data dtype before verification: {train_data.dtype}")
        
        # Ensure the data is float type
        if train_data.dtype != torch.float32:
            print(f"Converting train_data from {train_data.dtype} to float32")
            train_data = train_data.float()
        
        # Create data loader
        dataset = TensorDataset(train_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Labels for real and fake data with label smoothing for stability
        # Use 0.9 instead of 1.0 and 0.1 instead of 0.0
        real_label_value = 0.9  # Label smoothing for stability
        fake_label_value = 0.1  # Label smoothing for stability
        
        # Training history
        g_losses = []
        d_losses = []
        sat_scores = []
        
        # Early stopping variables
        best_sat_score = 0
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            running_d_loss = 0.0
            running_g_loss = 0.0
            
            # Initialize epoch loss values to handle case where all batches are skipped
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            
            # For stability monitoring
            loss_values = []
            nan_batches = 0
            total_batches = 0
            
            # Iterate over batches
            for batch_idx, (real_data,) in enumerate(dataloader):
                total_batches += 1
                
                # Move data to device and ensure float type
                real_data = real_data.to(self.device)
                if real_data.dtype != torch.float32:
                    real_data = real_data.float()
                    
                batch_size = real_data.size(0)
                
                # -----------------
                # Train Discriminator
                # -----------------
                self.discriminator.zero_grad()
                
                # Generate fake data
                noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_data = self.generator(noise)
                
                # Ensure fake data is float
                if fake_data.dtype != torch.float32:
                    fake_data = fake_data.float()
                
                # Real data
                output_real = self.discriminator(real_data)
                label_real = torch.full((batch_size, 1), real_label_value, dtype=torch.float32, device=self.device)
                loss_real = self.criterion(output_real, label_real)
                
                # Fake data
                output_fake = self.discriminator(fake_data.detach())
                label_fake = torch.full((batch_size, 1), fake_label_value, dtype=torch.float32, device=self.device)
                loss_fake = self.criterion(output_fake, label_fake)
                
                try:
                    # Gradient penalty for stability
                    gradient_penalty = self.compute_gradient_penalty(real_data, fake_data.detach())
                    
                    # Combined loss with gradient penalty
                    d_loss = loss_real + loss_fake + self.gp_weight * gradient_penalty
                    
                    # Check for very large loss values before backprop
                    if d_loss.item() > 100 or torch.isnan(d_loss).item():
                        print(f"Warning: Discriminator loss issue detected: {d_loss.item()}")
                        nan_batches += 1
                        # Reset discriminator weights if loss is problematic
                        if nan_batches > 3:  # If multiple consecutive NaN batches
                            print("Reinitializing discriminator weights due to instability")
                            self._init_weights(self.discriminator)
                        continue
                    
                    d_loss.backward()
                    
                    # Apply gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
                    
                    self.optimizer_D.step()
                        
                    running_d_loss += d_loss.item()
                    loss_values.append(d_loss.item())
                except Exception as e:
                    print(f"Error in discriminator training: {str(e)}")
                    nan_batches += 1
                    continue
                
                # -----------------
                # Train Generator less frequently for stability
                # -----------------
                # Only train generator every other batch
                if batch_idx % 2 == 0:
                    try:
                        self.generator.zero_grad()
                        
                        # Generate fake data again
                        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                        fake_data = self.generator(noise)
                        # Ensure fake data is float
                        if fake_data.dtype != torch.float32:
                            fake_data = fake_data.float()
                        
                        # Try to fool discriminator
                        output = self.discriminator(fake_data)
                        label = torch.full((batch_size, 1), real_label_value, dtype=torch.float32, device=self.device)
                        
                        # GAN loss
                        g_loss = self.criterion(output, label)
                        
                        # Add clause satisfaction loss if applicable
                        clause_loss = self.clause_loss(fake_data)
                        g_loss += clause_loss
                        
                        # Add diversity loss if applicable
                        diversity_loss = self.diversity_loss(fake_data)
                        g_loss += diversity_loss
                        
                        # Check for very large loss values before backprop
                        if g_loss.item() > 100 or torch.isnan(g_loss).item():
                            print(f"Warning: Generator loss issue detected: {g_loss.item()}")
                            continue
                        
                        # Backprop
                        g_loss.backward()
                        
                        # Apply gradient clipping to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.max_grad_norm)
                        
                        self.optimizer_G.step()
                            
                        running_g_loss += g_loss.item()
                        loss_values.append(g_loss.item())
                    except Exception as e:
                        print(f"Error in generator training: {str(e)}")
                        continue
            
            # Check for numerical stability issues
            if nan_batches > 0:
                print(f"Warning: {nan_batches}/{total_batches} batches had issues in epoch {epoch}")
            
            if loss_values:
                avg_loss = sum(loss_values) / len(loss_values)
                max_loss = max(loss_values) if loss_values else 0
                print(f"Epoch {epoch} loss stats: avg={avg_loss:.4f}, max={max_loss:.4f}")
            
            # Save epoch losses (only if we had valid batches)
            if total_batches > nan_batches and batch_idx >= 0:
                valid_batches = total_batches - nan_batches
                epoch_d_loss = running_d_loss / valid_batches if valid_batches > 0 else 0
                epoch_g_loss = running_g_loss / valid_batches if valid_batches > 0 else 0
            
            g_losses.append(epoch_g_loss)
            d_losses.append(epoch_d_loss)
            
            # Evaluate model every eval_interval epochs
            if epoch % eval_interval == 0 or epoch == epochs - 1:
                try:
                    # Generate samples and evaluate
                    num_eval_samples = min(100, batch_size * 5)
                    gen_solutions = self.generate(num_eval_samples)
                    
                    # Calculate clause satisfaction
                    sat_ratios = []
                    for sol in gen_solutions:
                        satisfied_count = count_satisfied_clauses(self.clauses, sol)
                        sat_ratio = satisfied_count / len(self.clauses)
                        sat_ratios.append(sat_ratio)
                    
                    avg_sat_ratio = sum(sat_ratios) / len(sat_ratios) if sat_ratios else 0
                    sat_scores.append(avg_sat_ratio)
                except Exception as e:
                    print(f"Error during evaluation: {str(e)}")
                    avg_sat_ratio = 0.0
                
                # Print progress
                print(f"Epoch {epoch}/{epochs} | "
                      f"D Loss: {epoch_d_loss:.4f} | "
                      f"G Loss: {epoch_g_loss:.4f} | "
                      f"Avg Sat: {avg_sat_ratio:.2%}")
    
    def generate(self, num_samples: int = 100, batch_size: int = 32, threshold: float = 0.0, temperature: float = 1.0) -> List[List[int]]:
        """
        Generate SAT variable assignments using the trained generator.
        
        Args:
            num_samples: Number of samples to generate
            batch_size: Batch size for generation
            threshold: Threshold for converting continuous values to binary (-1/1)
                       If 0, use sign of the value; if > 0, use threshold
            temperature: Controls randomness in generation (higher = more random)
                       Values close to 0 make outputs more deterministic
        
        Returns:
            List of generated variable assignments as lists of literals
        """
        # Set model to evaluation mode
        self.generator.eval()
        
        # Generate samples
        samples = []
        remaining = num_samples
        
        with torch.no_grad():
            while remaining > 0:
                # Generate a batch
                curr_batch_size = min(batch_size, remaining)
                noise = torch.randn(curr_batch_size, self.latent_dim, device=self.device)
                
                # Apply temperature scaling to noise (higher temp = more randomness)
                if temperature != 1.0:
                    noise = noise * temperature
                    
                fake_data = self.generator(noise)
                
                # Convert continuous values to binary assignments
                if threshold > 0:
                    # Use explicit threshold
                    pos_indices = fake_data > threshold
                    neg_indices = fake_data < -threshold
                    
                    # Handle values in between thresholds (uncertain)
                    uncertain = (~pos_indices) & (~neg_indices)
                    if uncertain.any():
                        # For uncertain values, use the sign
                        pos_indices = pos_indices | (uncertain & (fake_data > 0))
                        neg_indices = neg_indices | (uncertain & (fake_data <= 0))
                else:
                    # Simply use the sign
                    pos_indices = fake_data > 0
                    neg_indices = fake_data <= 0
                
                # Convert to literals
                for i in range(curr_batch_size):
                    assignment = []
                    for j in range(self.n_vars):
                        if pos_indices[i, j]:
                            assignment.append(j + 1)  # 1-indexed
                        elif neg_indices[i, j]:
                            assignment.append(-(j + 1))  # 1-indexed
                    samples.append(assignment)
                
                remaining -= curr_batch_size
        
        # Set model back to training mode
        self.generator.train()
        
        return samples