"""
Metadata enrichment utilities for experience storage.
Provides tools for tracking git repository information, hardware details,
random seeds, and experiment tags for perfect reproducibility.
"""

import os
import json
import uuid
import time
import socket
import platform
import subprocess
import hashlib
import random
import logging
import dataclasses
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime
import importlib.metadata  # Use importlib.metadata instead of pkg_resources

# Set up logging
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    logger.warning("GitPython not found. Git repository tracking will be limited.")
    GIT_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not found. GPU information will be limited.")
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not found. Some hardware information will be limited.")
    TENSORFLOW_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    logger.warning("GPUtil not found. Detailed GPU information will be limited.")
    GPUTIL_AVAILABLE = False


@dataclasses.dataclass
class GitInfo:
    """Information about the git repository."""
    commit_hash: str = ""
    branch: str = ""
    is_dirty: bool = False
    commit_timestamp: str = ""
    commit_message: str = ""
    remote_url: str = ""


@dataclasses.dataclass
class HardwareInfo:
    """Information about the hardware environment."""
    hostname: str = ""
    platform: str = ""
    platform_version: str = ""
    python_version: str = ""
    cpu_count: int = 0
    cpu_model: str = ""
    memory_gb: float = 0.0
    gpu_name: str = ""
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    cuda_version: str = ""
    cudnn_version: str = ""


@dataclasses.dataclass
class SoftwareInfo:
    """Information about the software environment."""
    python_version: str = ""
    packages: Dict[str, str] = dataclasses.field(default_factory=dict)
    torch_version: str = ""
    tensorflow_version: str = ""
    numpy_version: str = ""
    environment_vars: Dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ExperimentMetadata:
    """Complete metadata for an experiment."""
    # Unique identifier for this experiment
    experiment_id: str = ""
    # User-provided name and description
    name: str = ""
    description: str = ""
    # Time tracking
    created_at: str = ""
    updated_at: str = ""
    # Git information
    git_info: GitInfo = dataclasses.field(default_factory=GitInfo)
    # Hardware information
    hardware_info: HardwareInfo = dataclasses.field(default_factory=HardwareInfo)
    # Software environment
    software_info: SoftwareInfo = dataclasses.field(default_factory=SoftwareInfo)
    # User-defined tags for searching and filtering
    tags: Set[str] = dataclasses.field(default_factory=set)
    # Additional custom metadata
    custom_metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # Base random seed
    base_random_seed: int = 0
    # Episode random seeds (only track a few to save space)
    episode_seeds: Dict[int, int] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = dataclasses.asdict(self)
        # Convert set to list for JSON serialization
        result['tags'] = list(result['tags'])
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentMetadata':
        """Create an ExperimentMetadata instance from a dictionary."""
        # Convert list back to set
        if 'tags' in data and isinstance(data['tags'], list):
            data['tags'] = set(data['tags'])
            
        # Handle nested dataclasses
        if 'git_info' in data and isinstance(data['git_info'], dict):
            data['git_info'] = GitInfo(**data['git_info'])
        
        if 'hardware_info' in data and isinstance(data['hardware_info'], dict):
            data['hardware_info'] = HardwareInfo(**data['hardware_info'])
            
        if 'software_info' in data and isinstance(data['software_info'], dict):
            data['software_info'] = SoftwareInfo(**data['software_info'])
            
        return cls(**data)
    
    def to_json(self, indent=2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ExperimentMetadata':
        """Create an ExperimentMetadata instance from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


def get_git_info(repo_path: Optional[str] = None) -> GitInfo:
    """
    Get information about the git repository.
    
    Args:
        repo_path: Path to the git repository (default: current directory)
        
    Returns:
        GitInfo object with repository details
    """
    git_info = GitInfo()
    
    try:
        if not repo_path:
            repo_path = os.getcwd()
            
        if GIT_AVAILABLE:
            # Use GitPython for detailed information
            try:
                repo = git.Repo(repo_path, search_parent_directories=True)
                
                git_info.commit_hash = repo.head.commit.hexsha
                git_info.branch = repo.active_branch.name
                git_info.is_dirty = repo.is_dirty()
                git_info.commit_timestamp = datetime.fromtimestamp(
                    repo.head.commit.committed_date).isoformat()
                git_info.commit_message = repo.head.commit.message.strip()
                
                # Get remote URL if available
                try:
                    if repo.remotes:
                        git_info.remote_url = repo.remotes.origin.url
                except Exception as e:
                    logger.debug(f"Could not get remote URL: {e}")
            except Exception as e:
                logger.warning(f"Error getting git information: {e}")
        else:
            # Fallback to subprocess calls
            try:
                # Get commit hash
                result = subprocess.run(
                    ['git', 'rev-parse', 'HEAD'],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
                git_info.commit_hash = result.stdout.strip()
                
                # Get branch name
                result = subprocess.run(
                    ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
                git_info.branch = result.stdout.strip()
                
                # Check if repository is dirty
                result = subprocess.run(
                    ['git', 'status', '--porcelain'],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
                git_info.is_dirty = bool(result.stdout.strip())
                
                # Get commit timestamp
                result = subprocess.run(
                    ['git', 'show', '-s', '--format=%ci', 'HEAD'],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
                timestamp = result.stdout.strip()
                git_info.commit_timestamp = timestamp
                
                # Get commit message
                result = subprocess.run(
                    ['git', 'show', '-s', '--format=%s', 'HEAD'],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
                git_info.commit_message = result.stdout.strip()
                
                # Get remote URL
                result = subprocess.run(
                    ['git', 'config', '--get', 'remote.origin.url'],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=False  # Don't raise error if no remote
                )
                if result.returncode == 0:
                    git_info.remote_url = result.stdout.strip()
            except subprocess.SubprocessError as e:
                logger.warning(f"Error in git subprocess: {e}")
            except Exception as e:
                logger.warning(f"Error getting git information: {e}")
    except Exception as e:
        logger.warning(f"Failed to get git information: {e}")
        
    return git_info


def get_hardware_info() -> HardwareInfo:
    """
    Get information about the hardware environment.
    
    Returns:
        HardwareInfo object with hardware details
    """
    hw_info = HardwareInfo()
    
    try:
        # Basic system information
        hw_info.hostname = socket.gethostname()
        hw_info.platform = platform.system()
        hw_info.platform_version = platform.version()
        hw_info.python_version = platform.python_version()
        
        # CPU information
        hw_info.cpu_count = os.cpu_count() or 0
        
        # Try to get CPU model name
        cpu_model = "Unknown"
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_model = line.split(":")[1].strip()
                            break
            except Exception:
                pass
        elif platform.system() == "Darwin":  # macOS
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                cpu_model = result.stdout.strip()
            except Exception:
                pass
        elif platform.system() == "Windows":
            try:
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    cpu_model = lines[1].strip()
            except Exception:
                pass
                
        hw_info.cpu_model = cpu_model
        
        # Memory information
        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if "MemTotal" in line:
                            mem_kb = int(line.split()[1])
                            hw_info.memory_gb = mem_kb / 1024 / 1024
                            break
            except Exception:
                pass
        elif platform.system() == "Darwin":  # macOS
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                mem_bytes = int(result.stdout.strip())
                hw_info.memory_gb = mem_bytes / 1024 / 1024 / 1024
            except Exception:
                pass
        elif platform.system() == "Windows":
            try:
                result = subprocess.run(
                    ["wmic", "computersystem", "get", "totalphysicalmemory"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    mem_bytes = int(lines[1].strip())
                    hw_info.memory_gb = mem_bytes / 1024 / 1024 / 1024
            except Exception:
                pass
        
        # GPU information
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                hw_info.gpu_count = len(gpus)
                if gpus:
                    hw_info.gpu_name = gpus[0].name
                    hw_info.gpu_memory_gb = gpus[0].memoryTotal / 1024  # Convert from MB to GB
            except Exception as e:
                logger.debug(f"Error getting GPU info from GPUtil: {e}")
        
        # Try PyTorch for CUDA information
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                hw_info.gpu_count = torch.cuda.device_count()
                if hw_info.gpu_count > 0:
                    hw_info.gpu_name = torch.cuda.get_device_name(0)
                    hw_info.cuda_version = torch.version.cuda
                    # Get cuDNN version if available
                    if hasattr(torch.backends, 'cudnn') and torch.backends.cudnn.is_available():
                        hw_info.cudnn_version = str(torch.backends.cudnn.version())
            except Exception as e:
                logger.debug(f"Error getting CUDA info from PyTorch: {e}")
        
        # Try TensorFlow for CUDA information
        if TENSORFLOW_AVAILABLE:
            try:
                if tf.test.is_gpu_available():
                    # Get GPU device count from TensorFlow
                    gpus = tf.config.list_physical_devices('GPU')
                    hw_info.gpu_count = hw_info.gpu_count or len(gpus)
                    
                    # Get CUDA version from TensorFlow
                    if hw_info.cuda_version == "" and hasattr(tf, 'sysconfig'):
                        cuda_version = tf.sysconfig.get_build_info().get('cuda_version', '')
                        if cuda_version:
                            hw_info.cuda_version = cuda_version
            except Exception as e:
                logger.debug(f"Error getting GPU info from TensorFlow: {e}")
    except Exception as e:
        logger.warning(f"Error collecting hardware information: {e}")
    
    return hw_info


def get_software_info() -> SoftwareInfo:
    """
    Get information about the software environment.
    
    Returns:
        SoftwareInfo object with software details
    """
    sw_info = SoftwareInfo()
    
    try:
        # Python version
        sw_info.python_version = platform.python_version()
        
        # Installed packages
        for package in importlib.metadata.distributions():
            sw_info.packages[package.metadata['Name'].lower()] = package.version
        
        # PyTorch version
        if TORCH_AVAILABLE:
            sw_info.torch_version = torch.__version__
        
        # TensorFlow version
        if TENSORFLOW_AVAILABLE:
            sw_info.tensorflow_version = tf.__version__
        
        # NumPy version (often used in ML projects)
        try:
            import numpy
            sw_info.numpy_version = numpy.__version__
        except ImportError:
            pass
        
        # Relevant environment variables
        relevant_vars = [
            "PYTHONPATH", "PATH", "LD_LIBRARY_PATH", "CUDA_HOME",
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTHONHASHSEED"
        ]
        for var in relevant_vars:
            if var in os.environ:
                sw_info.environment_vars[var] = os.environ[var]
    except Exception as e:
        logger.warning(f"Error collecting software information: {e}")
    
    return sw_info


def generate_experiment_id() -> str:
    """
    Generate a unique experiment ID.
    
    Returns:
        Unique experiment ID string
    """
    # Combine UUID with timestamp for uniqueness
    timestamp = int(time.time())
    random_part = uuid.uuid4().hex[:8]
    return f"{timestamp}-{random_part}"


def generate_random_seed() -> int:
    """
    Generate a random seed for reproducibility.
    
    Returns:
        Random seed integer
    """
    # Use system entropy to generate seed
    # This is more reliable than time-based seeds
    return random.randrange(2**32)


def set_random_seeds(seed: int) -> None:
    """
    Set random seeds for all relevant libraries to ensure reproducibility.
    
    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    if TENSORFLOW_AVAILABLE:
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass
    
    # Set Python hash seed for even more determinism
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_experiment_metadata(
    name: str,
    description: str = "",
    tags: Optional[List[str]] = None,
    custom_metadata: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None
) -> ExperimentMetadata:
    """
    Create a complete metadata object for an experiment.
    
    Args:
        name: Experiment name
        description: Experiment description
        tags: List of searchable tags
        custom_metadata: Additional custom metadata
        seed: Random seed (if None, one will be generated)
        
    Returns:
        ExperimentMetadata object with complete metadata
    """
    timestamp = datetime.now().isoformat()
    
    # Generate seed if not provided
    if seed is None:
        seed = generate_random_seed()
    
    # Set seeds for reproducibility
    set_random_seeds(seed)
    
    metadata = ExperimentMetadata(
        experiment_id=generate_experiment_id(),
        name=name,
        description=description,
        created_at=timestamp,
        updated_at=timestamp,
        git_info=get_git_info(),
        hardware_info=get_hardware_info(),
        software_info=get_software_info(),
        tags=set(tags or []),
        custom_metadata=custom_metadata or {},
        base_random_seed=seed,
        episode_seeds={}
    )
    
    return metadata


def generate_episode_seed(metadata: ExperimentMetadata, episode_id: int) -> int:
    """
    Generate a deterministic seed for a specific episode based on the base seed.
    
    Args:
        metadata: Experiment metadata with base seed
        episode_id: Episode identifier
        
    Returns:
        Deterministic seed for the episode
    """
    # Combine the base seed and episode ID to create a deterministic but unique seed
    combined = f"{metadata.base_random_seed}-{episode_id}"
    hash_value = int(hashlib.md5(combined.encode()).hexdigest(), 16)
    episode_seed = hash_value % (2**32)  # Ensure it fits in a 32-bit integer
    
    # Store the seed for reproducibility (only store a limited number)
    # This helps keep metadata size reasonable while still enabling reproduction of key episodes
    if len(metadata.episode_seeds) < 100 or episode_id % 100 == 0:
        metadata.episode_seeds[episode_id] = episode_seed
        
    return episode_seed


def update_experiment_metadata(metadata: ExperimentMetadata) -> ExperimentMetadata:
    """
    Update dynamic fields in experiment metadata.
    
    Args:
        metadata: Existing experiment metadata
        
    Returns:
        Updated ExperimentMetadata object
    """
    # Update timestamp
    metadata.updated_at = datetime.now().isoformat()
    
    # Update git info (may have changed during experiment)
    metadata.git_info = get_git_info()
    
    return metadata


def save_metadata_to_file(metadata: ExperimentMetadata, filepath: str) -> bool:
    """
    Save experiment metadata to a JSON file.
    
    Args:
        metadata: Experiment metadata
        filepath: Path to save the JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save metadata to file
        with open(filepath, 'w') as f:
            f.write(metadata.to_json())
        
        return True
    except Exception as e:
        logger.error(f"Error saving metadata to file: {e}")
        return False


def load_metadata_from_file(filepath: str) -> Optional[ExperimentMetadata]:
    """
    Load experiment metadata from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        ExperimentMetadata object if successful, None otherwise
    """
    try:
        with open(filepath, 'r') as f:
            metadata = ExperimentMetadata.from_json(f.read())
        
        return metadata
    except Exception as e:
        logger.error(f"Error loading metadata from file: {e}")
        return None


def search_metadata_by_tags(metadata_list: List[ExperimentMetadata], 
                          tags: List[str],
                          match_all: bool = False) -> List[ExperimentMetadata]:
    """
    Search for experiments with specific tags.
    
    Args:
        metadata_list: List of metadata objects to search
        tags: List of tags to search for
        match_all: If True, all tags must match; if False, any tag may match
        
    Returns:
        List of matching metadata objects
    """
    tag_set = set(tags)
    results = []
    
    for metadata in metadata_list:
        if match_all:
            # All tags must be present
            if tag_set.issubset(metadata.tags):
                results.append(metadata)
        else:
            # Any tag may match
            if tag_set.intersection(metadata.tags):
                results.append(metadata)
    
    return results


class MetadataManager:
    """Manager class for handling experiment metadata."""
    
    def __init__(self, storage_dir: str = "metadata"):
        """
        Initialize the metadata manager.
        
        Args:
            storage_dir: Directory to store metadata files
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.current_metadata = None
    
    def create_experiment(self, name: str, description: str = "", 
                        tags: Optional[List[str]] = None,
                        custom_metadata: Optional[Dict[str, Any]] = None,
                        seed: Optional[int] = None) -> ExperimentMetadata:
        """
        Create a new experiment with metadata.
        
        Args:
            name: Experiment name
            description: Experiment description
            tags: List of tags
            custom_metadata: Additional metadata
            seed: Random seed
            
        Returns:
            ExperimentMetadata for the new experiment
        """
        self.current_metadata = create_experiment_metadata(
            name, description, tags, custom_metadata, seed
        )
        
        # Save the metadata
        self._save_current_metadata()
        
        return self.current_metadata
    
    def get_episode_seed(self, episode_id: int) -> int:
        """
        Get a deterministic seed for an episode.
        
        Args:
            episode_id: Episode identifier
            
        Returns:
            Seed for the episode
        """
        if self.current_metadata is None:
            raise ValueError("No active experiment. Call create_experiment first.")
        
        return generate_episode_seed(self.current_metadata, episode_id)
    
    def update_metadata(self, tags: Optional[List[str]] = None,
                      custom_data: Optional[Dict[str, Any]] = None) -> ExperimentMetadata:
        """
        Update the current experiment metadata.
        
        Args:
            tags: New tags to add
            custom_data: New custom data to add
            
        Returns:
            Updated metadata
        """
        if self.current_metadata is None:
            raise ValueError("No active experiment. Call create_experiment first.")
        
        # Add new tags
        if tags:
            self.current_metadata.tags.update(tags)
        
        # Update custom metadata
        if custom_data:
            self.current_metadata.custom_metadata.update(custom_data)
        
        # Update timestamp and git info
        self.current_metadata = update_experiment_metadata(self.current_metadata)
        
        # Save updated metadata
        self._save_current_metadata()
        
        return self.current_metadata
    
    def _save_current_metadata(self) -> None:
        """Save the current metadata to file."""
        if self.current_metadata:
            filepath = os.path.join(
                self.storage_dir, 
                f"{self.current_metadata.experiment_id}.json"
            )
            save_metadata_to_file(self.current_metadata, filepath)
    
    def load_experiment(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """
        Load an experiment by ID.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            ExperimentMetadata if found, None otherwise
        """
        filepath = os.path.join(self.storage_dir, f"{experiment_id}.json")
        metadata = load_metadata_from_file(filepath)
        
        if metadata:
            self.current_metadata = metadata
            
        return metadata
    
    def list_experiments(self) -> List[str]:
        """
        List all available experiment IDs.
        
        Returns:
            List of experiment IDs
        """
        experiment_ids = []
        
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json"):
                experiment_id = filename.replace(".json", "")
                experiment_ids.append(experiment_id)
                
        return experiment_ids
    
    def search_experiments(self, tags: Optional[List[str]] = None,
                         match_all_tags: bool = False,
                         name_contains: Optional[str] = None,
                         git_hash: Optional[str] = None) -> List[ExperimentMetadata]:
        """
        Search for experiments matching criteria.
        
        Args:
            tags: List of tags to search for
            match_all_tags: If True, all tags must match
            name_contains: String that must be in the experiment name
            git_hash: Git commit hash to match
            
        Returns:
            List of matching experiment metadata
        """
        results = []
        
        for experiment_id in self.list_experiments():
            metadata = self.load_experiment(experiment_id)
            
            if not metadata:
                continue
                
            # Check tags
            if tags:
                if match_all_tags:
                    if not set(tags).issubset(metadata.tags):
                        continue
                else:
                    if not set(tags).intersection(metadata.tags):
                        continue
            
            # Check name
            if name_contains and name_contains.lower() not in metadata.name.lower():
                continue
                
            # Check git hash
            if git_hash and metadata.git_info.commit_hash != git_hash:
                continue
                
            results.append(metadata)
            
        return results


# Global instance for convenience
metadata_manager = MetadataManager()