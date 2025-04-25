class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        
        # Ensure dimensions are consistent
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Model architecture
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, output_dim),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, z):
        # Validate input dimensions
        if z.size(1) != self.latent_dim:
            raise ValueError(f"Expected latent dimension {self.latent_dim}, got {z.size(1)}")
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        
        # Store input dimension for validation
        self.input_dim = input_dim
        
        # Model architecture
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )
    
    def forward(self, x):
        # Validate input dimensions
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.size(1)}")
        return self.model(x)