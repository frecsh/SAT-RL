"""
Improved GAN-based approach for SAT problems.
This module implements a GAN with experience replay to generate promising solutions.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import random
import time
import matplotlib.pyplot as plt


class ImprovedSATGAN:
    def __init__(self, n_vars, clauses, latent_dim=32, epochs=100, initial_solutions=None):
        """
        Initialize an Improved SAT GAN with experience replay.
        
        Args:
            n_vars: Number of variables in the SAT problem
            clauses: List of clauses in CNF form
            latent_dim: Dimension of latent space for generator input
            epochs: Number of training epochs
            initial_solutions: Optional list of known promising solutions to seed the experience buffer
        """
        self.n_vars = n_vars
        self.clauses = clauses
        self.latent_dim = latent_dim
        self.epochs = epochs
        
        # Create experience replay buffer
        self.experience_buffer = []
        self.max_buffer_size = 200
        
        # Add initial solutions to buffer if provided
        if initial_solutions:
            for solution in initial_solutions:
                if solution is not None:
                    self.add_to_experience_buffer(solution)
        
        # Build and compile generator and discriminator
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.gan = self._build_gan()
        
        # Track best solution
        self.best_solution = None
        self.best_satisfied = 0
        
        # Performance tracking
        self.g_loss_history = []
        self.d_loss_history = []
        self.satisfaction_history = []
    
    def _build_generator(self):
        """Build generator model that transforms noise into candidate solutions"""
        noise_input = Input(shape=(self.latent_dim,))
        
        x = Dense(64, activation=LeakyReLU(alpha=0.2))(noise_input)
        x = Dense(128, activation=LeakyReLU(alpha=0.2))(x)
        x = Dropout(0.2)(x)
        x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
        x = Dropout(0.2)(x)
        # Output layer uses sigmoid to get values between 0 and 1
        output = Dense(self.n_vars, activation='sigmoid')(x)
        
        model = Model(noise_input, output)
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
        return model
    
    def _build_discriminator(self):
        """Build discriminator model that distinguishes valid from invalid solutions"""
        solution_input = Input(shape=(self.n_vars,))
        
        x = Dense(128, activation=LeakyReLU(alpha=0.2))(solution_input)
        x = Dropout(0.3)(x)
        x = Dense(64, activation=LeakyReLU(alpha=0.2))(x)
        x = Dropout(0.3)(x)
        # Output probability of being a valid solution
        validity = Dense(1, activation='sigmoid')(x)
        
        model = Model(solution_input, validity)
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
        return model
    
    def _build_gan(self):
        """Build combined GAN model"""
        # For the combined model, we only train the generator
        self.discriminator.trainable = False
        
        # Input is noise, output is validity after passing through generator and discriminator
        noise_input = Input(shape=(self.latent_dim,))
        generated_solution = self.generator(noise_input)
        validity = self.discriminator(generated_solution)
        
        # Combined model trains generator to fool discriminator
        model = Model(noise_input, validity)
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
        return model
    
    def evaluate_solution(self, solution):
        """
        Evaluate how many clauses are satisfied by the solution.
        Returns: tuple (satisfied_count, total_clauses)
        """
        binary_solution = np.round(solution).astype(int)
        satisfied = 0
        
        for clause in self.clauses:
            clause_satisfied = False
            for literal in clause:
                var_idx = abs(literal) - 1  # Convert to 0-indexed
                var_value = binary_solution[var_idx]
                
                # Check if literal is satisfied
                if (literal > 0 and var_value == 1) or (literal < 0 and var_value == 0):
                    clause_satisfied = True
                    break
            
            if clause_satisfied:
                satisfied += 1
        
        # Update best solution if better
        if satisfied > self.best_satisfied:
            self.best_satisfied = satisfied
            self.best_solution = binary_solution.copy()
            
            # Add to experience buffer if significantly better
            if satisfied > 0.8 * len(self.clauses):
                self.add_to_experience_buffer(binary_solution)
        
        return satisfied, len(self.clauses)
    
    def add_to_experience_buffer(self, solution):
        """Add a solution to the experience buffer if not already present"""
        # Convert to binary
        binary_solution = np.round(solution).astype(int)
        
        # Check if solution is already in buffer
        for existing in self.experience_buffer:
            if np.array_equal(binary_solution, existing):
                return False
        
        # Add to buffer
        self.experience_buffer.append(binary_solution.copy())
        
        # Keep buffer at max size
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
        
        return True
    
    def generate_training_data(self, batch_size):
        """Generate training data for discriminator"""
        # Create random noise for generator
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        
        # Generate fake solutions
        gen_solutions = self.generator.predict(noise, verbose=0)
        fake_solutions = np.round(gen_solutions).astype(int)
        fake_labels = np.zeros((batch_size, 1))
        
        # Create real solutions - mix random and from experience buffer
        real_solutions = []
        
        # Add solutions from experience buffer
        if self.experience_buffer:
            for _ in range(min(batch_size // 2, len(self.experience_buffer))):
                real_solutions.append(random.choice(self.experience_buffer))
        
        # Fill the rest with random solutions but with bias towards satisfying clauses
        while len(real_solutions) < batch_size:
            solution = np.random.randint(0, 2, size=self.n_vars)
            satisfied, total = self.evaluate_solution(solution)
            
            # Accept with probability proportional to satisfaction
            if satisfied / total > 0.5 or random.random() < 0.3:
                real_solutions.append(solution)
        
        real_solutions = np.array(real_solutions)
        real_labels = np.ones((batch_size, 1))
        
        return real_solutions, fake_solutions, real_labels, fake_labels
    
    def train_with_experience_replay(self, batch_size=32):
        """Train the GAN with experience replay"""
        print("Training SAT-GAN with experience replay...")
        start_time = time.time()
        
        # Reset histories
        self.g_loss_history = []
        self.d_loss_history = []
        self.satisfaction_history = []
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Train discriminator
            real_solutions, fake_solutions, real_labels, fake_labels = self.generate_training_data(batch_size)
            
            d_loss_real = self.discriminator.train_on_batch(real_solutions, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_solutions, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, np.ones((batch_size, 1)))
            
            # Every 10 epochs, generate solutions and evaluate
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                noise = np.random.normal(0, 1, (10, self.latent_dim))
                gen_solutions = self.generator.predict(noise, verbose=0)
                
                avg_satisfaction = 0
                for sol in gen_solutions:
                    binary_sol = np.round(sol).astype(int)
                    satisfied, total = self.evaluate_solution(binary_sol)
                    avg_satisfaction += satisfied / total
                
                avg_satisfaction /= 10
                self.satisfaction_history.append(avg_satisfaction)
                
                print(f"Epoch {epoch}/{self.epochs} | "
                     f"D loss: {d_loss:.4f} | G loss: {g_loss:.4f} | "
                     f"Satisfaction: {avg_satisfaction:.4f} | "
                     f"Best: {self.best_satisfied}/{len(self.clauses)} | "
                     f"Time: {time.time() - epoch_start:.2f}s")
                
                # Add promising solutions to experience buffer
                for sol in gen_solutions:
                    binary_sol = np.round(sol).astype(int)
                    satisfied, total = self.evaluate_solution(binary_sol)
                    if satisfied / total > 0.7:
                        self.add_to_experience_buffer(binary_sol)
            
            self.g_loss_history.append(g_loss)
            self.d_loss_history.append(d_loss)
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        print(f"Best solution satisfies {self.best_satisfied}/{len(self.clauses)} clauses")
        
        # Visualize training history
        self.visualize_training_history()
        
        return self.best_solution
    
    def solve(self, max_generations=100):
        """Generate and evaluate solutions to find the best one"""
        if self.best_solution is not None and self.best_satisfied == len(self.clauses):
            return self.best_solution
        
        print(f"Searching for solution with GAN over {max_generations} generations...")
        start_time = time.time()
        
        for i in range(max_generations):
            # Generate a batch of solutions
            batch_size = 32
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            generated_solutions = self.generator.predict(noise, verbose=0)
            
            # Evaluate each solution
            for sol_idx, solution in enumerate(generated_solutions):
                binary_solution = np.round(solution).astype(int)
                satisfied, total = self.evaluate_solution(binary_solution)
                
                # Print progress every 10 generations
                if i % 10 == 0 and sol_idx == 0:
                    print(f"Generation {i}/{max_generations} | "
                         f"Best: {self.best_satisfied}/{len(self.clauses)} | "
                         f"Time: {time.time() - start_time:.2f}s")
                
                # If found complete solution, return it
                if satisfied == total:
                    print(f"Solution found at generation {i}!")
                    return self.best_solution
        
        print(f"Search completed. Best solution satisfies {self.best_satisfied}/{len(self.clauses)} clauses")
        return self.best_solution
    
    def visualize_training_history(self):
        """Visualize GAN training history"""
        plt.figure(figsize=(15, 5))
        
        # Plot loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.g_loss_history, label='Generator Loss')
        plt.plot(self.d_loss_history, label='Discriminator Loss')
        plt.title('GAN Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot satisfaction rate
        plt.subplot(1, 2, 2)
        epochs = range(0, self.epochs, 10)
        if len(epochs) != len(self.satisfaction_history):
            epochs = range(0, len(self.satisfaction_history) * 10, 10)
        plt.plot(epochs, self.satisfaction_history)
        plt.title('Average Clause Satisfaction Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Satisfaction Rate')
        plt.ylim([0, 1])
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('satgan_training_history.png')
        plt.close()
        print("GAN training history visualization saved to satgan_training_history.png")