import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class SATGAN:
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, latent_dim, device, clause_weight=0):
        """
        Initialize the SATGAN model
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            g_optimizer: Optimizer for generator
            d_optimizer: Optimizer for discriminator
            latent_dim: Dimension of latent space
            device: Device to run the model on (e.g., 'cpu' or 'cuda')
            clause_weight: Weight for clause satisfaction loss
        """
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.latent_dim = latent_dim
        self.device = device
        self.clause_weight = clause_weight

    def evaluate_clause_satisfaction(self, samples):
        """
        Evaluate clause satisfaction for generated samples
        
        Args:
            samples: Generated samples
        
        Returns:
            Clause satisfaction loss
        """
        # Placeholder for clause satisfaction evaluation
        return torch.tensor(0.0, device=self.device)

    def evaluate(self, n_samples=5):
        """
        Evaluate the model by generating samples
        
        Args:
            n_samples: Number of samples to generate
        """
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
            samples = self.generator(z)
            print("Generated samples:", samples)

    def train(self, solutions=None, training_data=None, epochs=30, batch_size=64, eval_interval=10):
        """
        Train the GAN model
        
        Args:
            solutions: List of SAT solutions (alias for training_data)
            training_data: List of SAT solutions
            epochs: Number of training epochs
            batch_size: Batch size for training
            eval_interval: How often to evaluate model (every N epochs)
        """
        # Handle solutions/training_data parameter name compatibility
        if solutions is not None and training_data is None:
            training_data = solutions
        
        # Set model to training mode
        self.generator.train()
        self.discriminator.train()
        
        # Convert training data to tensors
        training_tensor = torch.tensor(training_data, dtype=torch.float32).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(training_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(epochs):
            g_losses = []
            d_losses = []
            
            for batch in dataloader:
                real_samples = batch[0]
                
                # Train Discriminator
                self.d_optimizer.zero_grad()
                
                # Real samples
                real_pred = self.discriminator(real_samples)
                real_loss = F.binary_cross_entropy(
                    real_pred, 
                    torch.ones(real_samples.size(0), 1).to(self.device)
                )
                
                # Generate fake samples
                z = torch.randn(real_samples.size(0), self.latent_dim).to(self.device)
                fake_samples = self.generator(z)
                fake_pred = self.discriminator(fake_samples.detach())
                fake_loss = F.binary_cross_entropy(
                    fake_pred, 
                    torch.zeros(fake_samples.size(0), 1).to(self.device)
                )
                
                # Total discriminator loss
                d_loss = real_loss + fake_loss
                d_loss.backward()
                self.d_optimizer.step()
                d_losses.append(d_loss.item())
                
                # Train Generator
                self.g_optimizer.zero_grad()
                
                # Generate fake samples again (since we detached earlier)
                z = torch.randn(real_samples.size(0), self.latent_dim).to(self.device)
                fake_samples = self.generator(z)
                fake_pred = self.discriminator(fake_samples)
                
                # Generator loss
                g_loss = F.binary_cross_entropy(
                    fake_pred, 
                    torch.ones(fake_samples.size(0), 1).to(self.device)
                )
                
                # Add clause satisfaction loss if weight > 0
                if self.clause_weight > 0:
                    clause_loss = self.evaluate_clause_satisfaction(fake_samples)
                    g_loss += self.clause_weight * clause_loss
                
                g_loss.backward()
                self.g_optimizer.step()
                g_losses.append(g_loss.item())
            
            # Print progress
            if epoch % eval_interval == 0 or epoch == epochs - 1:
                avg_g_loss = sum(g_losses) / len(g_losses)
                avg_d_loss = sum(d_losses) / len(d_losses)
                print(f"Epoch {epoch+1}/{epochs} | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f}")
                
                # Evaluate model on sample
                self.evaluate(n_samples=5)
        
        print("Training complete!")