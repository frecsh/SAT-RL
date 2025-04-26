"""
Improved GAN-based approach for SAT solving using experience replay.
This enhances solution generation by maintaining a buffer of promising solutions.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import random
from collections import deque


class ImprovedSATGAN:
    def __init__(self, n_vars, clauses, latent_dim=32, batch_size=64, epochs=100):
        """
        Initialize an improved GAN for SAT problem solving.
        
        Args:
            n_vars: Number of variables in the SAT problem
            clauses: List of clauses in CNF form
            latent_dim: Dimension of the latent space
            batch_size: Batch size for training
            epochs: Number of training epochs
        """
        self.n_vars = n_vars
        self.clauses = clauses
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=1000)
        
        # Initialize GAN components
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.gan = self._build_gan()
        
        # Track best solutions
        self.best_solution = None
        self.best_satisfied = 0
    
    def _build_generator(self):
        """Build generator network using functional API"""
        inputs = Input(shape=(self.latent_dim,))
        x = Dense(64)(inputs)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Dense(128)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        outputs = Dense(self.n_vars, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
        return model
    
    def _build_discriminator(self):
        """Build discriminator network using functional API"""
        inputs = Input(shape=(self.n_vars,))
        x = Dense(128)(inputs)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        x = Dense(64)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
        return model
    
    def _build_gan(self):
        """Build combined GAN model (generator + discriminator)"""
        # Freeze discriminator weights during generator training
        self.discriminator.trainable = False
        
        # GAN input (noise) and output (generated samples)
        gan_input = Input(shape=(self.latent_dim,))
        generated_sample = self.generator(gan_input)
        gan_output = self.discriminator(generated_sample)
        
        # Combined model
        model = Model(gan_input, gan_output)
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
        return model
    
    def count_satisfied_clauses(self, assignment):
        """Count number of clauses satisfied by an assignment"""
        satisfied = 0
        assignment_binary = np.round(assignment).astype(int)  # Convert probabilities to binary
        
        for clause in self.clauses:
            for literal in clause:
                var = abs(literal) - 1  # Convert to 0-indexed
                val = assignment_binary[var]
                if (literal > 0 and val == 1) or (literal < 0 and val == 0):
                    satisfied += 1
                    break
        return satisfied
    
    def _evaluate_solution(self, solution):
        """Evaluate a solution and update best if needed"""
        satisfied = self.count_satisfied_clauses(solution)
        
        # Update best solution if current is better
        if satisfied > self.best_satisfied:
            self.best_satisfied = satisfied
            self.best_solution = solution.copy()
            
        return satisfied / len(self.clauses)  # Return normalized satisfaction rate
    
    def generate_samples(self, n_samples):
        """Generate samples from the generator"""
        # Generate random noise
        noise = np.random.normal(0, 1, size=(n_samples, self.latent_dim))
        
        # Generate samples
        generated_samples = self.generator.predict(noise, verbose=0)
        
        return generated_samples
    
    def train_with_experience_replay(self, initial_solutions=None, buffer_size=1000):
        """Train the GAN using experience replay"""
        # Initialize experience buffer
        self.experience_buffer = deque(maxlen=buffer_size)
        
        # Add initial solutions to buffer if provided
        if initial_solutions is not None:
            for solution in initial_solutions:
                self.experience_buffer.append(solution)
                self._evaluate_solution(solution)
        
        # Generate some random solutions if buffer is empty
        if len(self.experience_buffer) == 0:
            random_solutions = np.random.randint(0, 2, size=(100, self.n_vars))
            for solution in random_solutions:
                satisfaction = self._evaluate_solution(solution)
                if satisfaction > 0.5:  # Only add promising solutions
                    self.experience_buffer.append(solution)
        
        # Training loop
        for epoch in range(self.epochs):
            # Sample from experience buffer for discriminator training
            if len(self.experience_buffer) > self.batch_size:
                # Real samples from experience buffer
                idx = np.random.randint(0, len(self.experience_buffer), self.batch_size)
                real_samples = np.array([self.experience_buffer[i] for i in idx])
            else:
                real_samples = np.array(list(self.experience_buffer))
                if len(real_samples) < 2:  # Need at least some samples
                    real_samples = np.random.randint(0, 2, size=(10, self.n_vars))
            
            # Generate fake samples
            noise = np.random.normal(0, 1, size=(len(real_samples), self.latent_dim))
            fake_samples = self.generator.predict(noise, verbose=0)
            
            # Train discriminator
            self.discriminator.trainable = True
            d_loss_real = self.discriminator.train_on_batch(real_samples, np.ones((len(real_samples), 1)))
            d_loss_fake = self.discriminator.train_on_batch(fake_samples, np.zeros((len(fake_samples), 1)))
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            
            # Train generator
            self.discriminator.trainable = False
            noise = np.random.normal(0, 1, size=(self.batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, np.ones((self.batch_size, 1)))
            
            # Generate new samples and add promising ones to experience buffer
            if epoch % 5 == 0:
                new_samples = self.generate_samples(10)
                for sample in new_samples:
                    binary_sample = np.round(sample).astype(int)
                    satisfaction = self._evaluate_solution(binary_sample)
                    if satisfaction > 0.8:  # Only add highly promising solutions
                        self.experience_buffer.append(binary_sample)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.epochs}, D loss: {d_loss:.4f}, G loss: {g_loss:.4f}, "
                      f"Best satisfied: {self.best_satisfied}/{len(self.clauses)}, "
                      f"Buffer size: {len(self.experience_buffer)}")
                
            # If solution found, terminate early
            if self.best_satisfied == len(self.clauses):
                print(f"Solution found at epoch {epoch}!")
                break
    
    def solve(self, max_generations=100, population_size=50):
        """Solve SAT problem using the GAN"""
        # First train the GAN with experience replay
        self.train_with_experience_replay()
        
        # If we haven't found a solution yet, try to generate more samples
        if self.best_satisfied < len(self.clauses):
            for gen in range(max_generations):
                samples = self.generate_samples(population_size)
                binary_samples = np.round(samples).astype(int)
                
                # Evaluate all samples
                for sample in binary_samples:
                    self._evaluate_solution(sample)
                
                print(f"Generation {gen}, Best satisfied: {self.best_satisfied}/{len(self.clauses)}")
                
                # If solution found, terminate
                if self.best_satisfied == len(self.clauses):
                    print(f"Solution found at generation {gen}!")
                    break
        
        # Return best solution found
        return np.round(self.best_solution).astype(int)