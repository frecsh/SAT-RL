def prepare_training_data(dataset_path, num_vars):
    """
    Load and prepare the dataset for training.
    
    Args:
        dataset_path: Path to dataset
        num_vars: Number of variables in the SAT problem
    
    Returns:
        training_data, validation_data
    """
    # Load solutions
    with open(dataset_path, 'r') as f:
        solutions = json.load(f)
    
    # Validate solution dimensions
    for solution in solutions:
        if len(solution) != num_vars:
            raise ValueError(f"Expected solution dimension {num_vars}, got {len(solution)}")
    
    # Split into training and validation
    train_size = int(0.8 * len(solutions))
    training_data = solutions[:train_size]
    validation_data = solutions[train_size:]
    
    print(f"Using {len(training_data)} solutions for training and {len(validation_data)} for validation")
    
    return training_data, validation_data

def train_satgan(training_data, num_vars, num_epochs=30, batch_size=64):
    """
    Train the SATGAN model
    
    Args:
        training_data: List of solutions
        num_vars: Number of variables in the SAT problem
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Trained SATGAN model
    """
    print(f"Starting SATGAN training for {num_epochs} epochs...")
    
    # Initialize the GAN model with explicit dimension parameters
    latent_dim = 100  # Typical value for GANs
    output_dim = num_vars  # Must match the number of variables in solutions
    
    # Check dimensions for sanity
    example_solution = training_data[0]
    if len(example_solution) != output_dim:
        raise ValueError(f"Solution dimension mismatch. Expected {output_dim}, got {len(example_solution)}")
    
    # Initialize model with correct dimensions
    try:
        model = SATGAN(
            num_vars=output_dim,
            latent_dim=latent_dim,
            clause_weight=0.1,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Train the model
        model.train(training_data, epochs=num_epochs, batch_size=batch_size)
        return model
        
    except Exception as e:
        print(f"Error during GAN training: {str(e)}")
        # Re-raise if needed for debugging
        raise
        
def main():
    parser = argparse.ArgumentParser(description="Train a SATGAN model")
    parser.add_argument("--stage", type=int, default=1, help="Training stage (1-5)")
    parser.add_argument("--clauses", type=int, default=10, help="Number of clauses")
    parser.add_argument("--vars", type=int, default=20, help="Number of variables")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    
    args = parser.parse_args()
    
    print(f"------------------------------------------------------------")
    print(f"Training Stage {args.stage}/5 ({args.clauses} clauses)")
    print(f"------------------------------------------------------------")
    
    # Generate or load dataset
    dataset_path = f"data/solutions_stage{args.stage}.json"
    if not os.path.exists(dataset_path):
        # Generate dataset if it doesn't exist
        generate_dataset(args.vars, args.clauses, dataset_path)
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        solutions = json.load(f)
    
    print(f"Training with {len(solutions)} solutions")
    
    # Prepare data
    train_data, val_data = prepare_training_data(dataset_path, args.vars)
    
    # Train model with explicit error handling
    try:
        model = train_satgan(
            training_data=train_data,
            num_vars=args.vars,
            num_epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Save trained model
        model_path = f"models/satgan_stage{args.stage}.pth"
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
    except Exception as e:
        print(f"Error during GAN training: {str(e)}")
        traceback.print_exc()