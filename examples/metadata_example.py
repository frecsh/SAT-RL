"""
Example demonstrating how to use metadata enrichment for experience storage.

This example shows:
1. How to create a storage backend with metadata tracking
2. How to add experiment tags and custom metadata
3. How to track per-episode random seeds for reproducibility
4. How to search for experiments by metadata
5. How to reproduce an experiment using saved metadata
"""

import os
import random
import numpy as np
from typing import Dict, Any, List

# Import our metadata system
from utils.metadata import (
    ExperimentMetadata, 
    create_experiment_metadata,
    set_random_seeds
)

# Import our type system
from utils.type_system import TypeSpecification, ExperienceSchema, DType

# Import our storage system with metadata integration
from utils.storage.metadata_integration import (
    create_storage_with_metadata,
    MetadataSearchEngine
)


def create_dummy_experience(episode_id: int, step: int) -> Dict[str, Any]:
    """Create a sample experience for demonstration."""
    return {
        "observation": np.random.rand(4).astype(np.float32),
        "action": random.randint(0, 1),
        "reward": random.random(),
        "next_observation": np.random.rand(4).astype(np.float32),
        "done": False,
        "episode_id": episode_id,
        "step": step,
        "metadata": {}  # Will be populated by the storage adapter
    }


def main():
    # Create a schema for our experiences
    schema = ExperienceSchema(
        field_specs={
            "observation": TypeSpecification(DType.FLOAT32, shape=(4,)),
            "action": TypeSpecification(DType.INT32),
            "reward": TypeSpecification(DType.FLOAT32),
            "next_observation": TypeSpecification(DType.FLOAT32, shape=(4,)),
            "done": TypeSpecification(DType.BOOL),
            "episode_id": TypeSpecification(DType.INT32),
            "step": TypeSpecification(DType.INT32),
            "metadata": TypeSpecification(nested_spec={
                "experiment_id": TypeSpecification(DType.STRING),
                "episode_seed": TypeSpecification(DType.INT32),
                # Could add more metadata fields here
            })
        },
        version="1.0"
    )
    
    # Directories for storing data and metadata
    os.makedirs("data", exist_ok=True)
    os.makedirs("metadata", exist_ok=True)
    
    # Create a storage backend with metadata tracking
    storage = create_storage_with_metadata(
        backend_type="hdf5",  # Could also use "arrow"
        schema=schema,
        path="data/experiment_data.h5",
        experiment_name="MetadataEnrichmentDemo",
        description="Example demonstrating metadata enrichment features",
        tags=["demo", "metadata", "reproducibility"],
        custom_metadata={
            "author": "John Madison",
            "purpose": "Demonstrating metadata enrichment",
            "version": "0.1"
        },
        # If seed is None, a random seed will be generated
        seed=None,
        metadata_dir="metadata"
    )
    
    print(f"Created experiment with ID: {storage.metadata.experiment_id}")
    print(f"Base random seed: {storage.metadata.base_random_seed}")
    
    # Simulate multiple episodes
    for episode_id in range(3):
        # Get a deterministic seed for this episode
        # This ensures reproducibility of episodes
        episode_seed = storage.start_episode()
        print(f"Episode {episode_id} seed: {episode_seed}")
        
        # Use the episode seed to ensure deterministic behavior
        set_random_seeds(episode_seed)
        
        # Create some experiences for this episode
        experiences = []
        for step in range(5):
            experiences.append(create_dummy_experience(episode_id, step))
        
        # Write batch of experiences
        storage.write_batch(experiences)
    
    # Add more tags later in the experiment
    storage.add_tags(["completed", "successful"])
    
    # Add more custom metadata
    storage.add_custom_metadata({
        "final_reward": 10.5,
        "completion_time": "2023-09-15T10:30:00"
    })
    
    # Update metadata with latest git info and timestamp
    storage.update_metadata()
    
    # Close the storage when done
    storage.close()
    
    print("\n" + "=" * 50 + "\n")
    print("Now demonstrating search capabilities:")
    
    # Create a search engine to find experiments by metadata
    search_engine = MetadataSearchEngine(
        metadata_dir="metadata",
        storage_dirs=["data"]
    )
    
    # List all experiments
    experiment_ids = search_engine.list_experiments()
    print(f"Found {len(experiment_ids)} experiments: {experiment_ids}")
    
    # Search for experiments with specific tags
    experiments_with_tags = search_engine.search_experiments(
        tags=["demo", "completed"],
        match_all_tags=True
    )
    
    print(f"Found {len(experiments_with_tags)} experiments with all required tags")
    for metadata in experiments_with_tags:
        print(f"  - {metadata.name} (ID: {metadata.experiment_id})")
        print(f"    Description: {metadata.description}")
        print(f"    Tags: {metadata.tags}")
        print(f"    Git commit: {metadata.git_info.commit_hash}")
        print(f"    Hardware: {metadata.hardware_info.platform} - {metadata.hardware_info.cpu_model}")
        print(f"    GPU: {metadata.hardware_info.gpu_name or 'None'}")
        print(f"    Created: {metadata.created_at}")
        print(f"    Custom metadata: {metadata.custom_metadata}")
        print()
    
    # Demonstrate how to reproduce an episode
    if experiment_ids:
        experiment_id = experiment_ids[0]
        episode_id = 1  # Reproduce the second episode
        
        result = search_engine.reproduce_episode(experiment_id, episode_id)
        if result:
            seed, experiences = result
            print(f"To reproduce episode {episode_id}, use seed: {seed}")
            print(f"Found {len(experiences)} experiences for this episode")
            
            # Here you would set the random seed and replay the episode
            set_random_seeds(seed)
            
            # The experiences could be used for analysis or replay visualization
            if experiences:
                print(f"First experience in episode: {experiences[0]}")
        else:
            print(f"Could not find information to reproduce episode {episode_id}")


if __name__ == "__main__":
    main()