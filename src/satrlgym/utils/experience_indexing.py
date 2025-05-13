"""
Indexing utilities for experience data.
"""

from typing import Any

import numpy as np


class ExperienceIndex:
    """
    Index for fast lookups of experience data.
    """

    def __init__(self):
        self.index = {}

    def add(self, key, value):
        """Add an entry to the index"""
        self.index[key] = value

    def get(self, key):
        """Get a value from the index"""
        return self.index.get(key)


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer that stores and samples experiences based on priority.
    Uses sum-tree structure for efficient sampling with priorities.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ):
        """
        Initialize the prioritized replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            alpha: Priority exponent (0 = uniform sampling, higher = more prioritization)
            beta: Importance sampling correction exponent (0 = no correction, 1 = full correction)
            beta_increment: Amount to increase beta each time we sample
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        # Experience storage
        self.experiences = [None] * capacity
        self.next_idx = 0
        self.size = 0

        # Sum tree for efficient priority-based sampling
        self.sum_tree = SumTree(capacity)

        # Tracking the maximum priority
        self.max_priority = 1.0

    def add(self, experience: dict[str, Any], priority: float | None = None) -> None:
        """
        Add an experience to the buffer with a given priority.

        Args:
            experience: Experience dictionary
            priority: Priority value (if None, use max_priority)
        """
        if priority is None:
            priority = self.max_priority

        # Apply priority exponentiation
        priority = priority**self.alpha

        # Store in sum tree and experience buffer
        self.sum_tree.add(priority)

        idx = self.next_idx
        self.experiences[idx] = experience

        # Update index
        self.next_idx = (self.next_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[list[dict[str, Any]], list[int], np.ndarray]:
        """
        Sample a batch of experiences based on priorities.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (experiences, indices, importance_sampling_weights)
        """
        # Increase beta for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Sample indices based on priorities
        indices = []
        weights = np.zeros(batch_size, dtype=np.float32)
        segment_size = self.sum_tree.total() / batch_size

        for i in range(batch_size):
            # Sample from each segment
            low = segment_size * i
            high = segment_size * (i + 1)
            value = np.random.uniform(low, high)

            # Get experience index from sum tree
            idx, priority = self.sum_tree.get_leaf(value)
            indices.append(idx)

            # Calculate importance sampling weight
            # (N * P(i))^(-beta) / max_weight
            sample_prob = priority / self.sum_tree.total()
            weights[i] = (self.size * sample_prob) ** (-self.beta)

        # Normalize weights
        weights /= weights.max()

        # Get experiences from indices
        sampled_experiences = [self.experiences[idx] for idx in indices]

        return sampled_experiences, indices, weights

    def update_priorities(self, indices: list[int], priorities: list[float]) -> None:
        """
        Update priorities for specific indices.

        Args:
            indices: List of indices to update
            priorities: List of new priority values
        """
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            # Ensure positive priority
            priority = max(1e-8, priority)

            # Update max priority
            self.max_priority = max(self.max_priority, priority)

            # Apply priority exponentiation
            priority = priority**self.alpha

            # Update sum tree
            self.sum_tree.update(idx, priority)

    def __len__(self) -> int:
        """Get current size of buffer."""
        return self.size


class SumTree:
    """
    Sum tree data structure for efficient sampling with priorities.
    A binary tree where the parent's value is the sum of its children.
    Leaf nodes contain priorities, and the root contains the sum of all priorities.
    """

    def __init__(self, capacity: int):
        """
        Initialize the sum tree.

        Args:
            capacity: Number of leaf nodes (experiences)
        """
        # Number of leaf nodes
        self.capacity = capacity

        # Tree structure: tree[0] is the root, tree[1] and tree[2] are the children, etc.
        # Total size = 2*capacity - 1 (internal nodes + leaf nodes)
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

        # Data indices for the leaf nodes
        self.data_idx = 0

    def total(self) -> float:
        """Get sum of all priorities."""
        return self.tree[0]

    def add(self, priority: float) -> None:
        """
        Add a new priority value in the tree.

        Args:
            priority: Priority value
        """
        # Get the leaf node index
        self.data_idx + self.capacity - 1

        # Update tree with new priority
        self.update(self.data_idx, priority)

        # Update data index
        self.data_idx = (self.data_idx + 1) % self.capacity

    def update(self, data_idx: int, priority: float) -> None:
        """
        Update a priority value at a specific index.

        Args:
            data_idx: Index of the data
            priority: New priority value
        """
        # Get the leaf node index
        idx = data_idx + self.capacity - 1

        # Update leaf node
        delta = priority - self.tree[idx]
        self.tree[idx] = priority

        # Update parent nodes
        self._propagate(idx, delta)

    def _propagate(self, idx: int, delta: float) -> None:
        """
        Propagate changes up the tree.

        Args:
            idx: Index of the node
            delta: Change in value
        """
        # Get parent index
        parent = (idx - 1) // 2

        # Update parent
        self.tree[parent] += delta

        # Continue propagation if not at root
        if parent > 0:
            self._propagate(parent, delta)

    def get_leaf(self, value: float) -> tuple[int, float]:
        """
        Find a leaf node based on a value.

        Args:
            value: Value to search for (0 <= value <= total_priority)

        Returns:
            Tuple of (data_idx, priority)
        """
        # Start from root
        idx = 0

        # Traverse the tree
        while idx < self.capacity - 1:  # Not a leaf node
            left = 2 * idx + 1
            right = left + 1

            # Go left or right based on comparison with value
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right

        # Convert tree index to data index
        data_idx = idx - (self.capacity - 1)

        return data_idx, self.tree[idx]


class RingBuffer:
    """
    Ring buffer implementation for efficient fixed-size FIFO queue.
    """

    def __init__(self, capacity: int):
        """
        Initialize the ring buffer.

        Args:
            capacity: Maximum number of elements
        """
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.next_idx = 0
        self.size = 0

    def add(self, item: Any) -> None:
        """
        Add an item to the buffer.

        Args:
            item: Item to add
        """
        self.buffer[self.next_idx] = item
        self.next_idx = (self.next_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> list[Any]:
        """
        Sample random items from the buffer.

        Args:
            batch_size: Number of items to sample

        Returns:
            List of sampled items
        """
        batch_size = min(batch_size, self.size)
        indices = np.random.choice(self.size, batch_size, replace=False)

        # Map indices to actual buffer indices
        buffer_indices = [(self.next_idx - 1 - i) % self.capacity for i in indices]
        return [self.buffer[i] for i in buffer_indices]

    def __len__(self) -> int:
        """Get current size of buffer."""
        return self.size


def index_experience_file(file_path, use_filelock=True):
    """
    Creates an index for a single experience file.

    Args:
        file_path: Path to the experience file
        use_filelock: Whether to use filelock to prevent concurrent access

    Returns:
        Dictionary with index information
    """
    import json
    import os
    from pathlib import Path

    if use_filelock:
        try:
            from filelock import FileLock
        except ImportError:
            print(
                "Warning: filelock package not found. Concurrent access to files may cause issues."
            )
            use_filelock = False

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Experience file {file_path} not found")

    # Create lock file path
    lock_file = file_path.parent / f"{file_path.name}.lock"

    # Use lock if enabled
    if use_filelock:
        lock = FileLock(lock_file)
        lock.acquire()

    try:
        # Try to load existing index if it exists
        index_file = file_path.parent / f"{file_path.name}.idx"
        if index_file.exists():
            with open(index_file) as f:
                return json.load(f)

        # Index doesn't exist, create it by scanning the file
        from .storage import ExperienceReader

        index = {
            "episode_count": 0,
            "transition_count": 0,
            "episodes": [],
            "metadata": {},
        }

        with ExperienceReader(file_path) as reader:
            # Get file metadata
            index["metadata"] = reader.get_metadata()

            # Scan batches and index episodes
            current_episode = {"start": 0, "length": 0, "reward": 0}
            transition_idx = 0

            for batch in reader.iter_batches(1000):
                for transition in batch:
                    # If this is a terminal state, end the episode
                    if transition.get("done", False) and current_episode["length"] > 0:
                        current_episode["end"] = transition_idx
                        current_episode["length"] = (
                            current_episode["end"] - current_episode["start"] + 1
                        )
                        index["episodes"].append(current_episode)
                        index["episode_count"] += 1

                        # Start a new episode
                        current_episode = {
                            "start": transition_idx + 1,
                            "length": 0,
                            "reward": 0,
                        }
                    else:
                        # Add to current episode
                        current_episode["length"] += 1
                        current_episode["reward"] += transition.get("reward", 0)

                    transition_idx += 1

            # Add the last episode if it's not empty
            if current_episode["length"] > 0:
                current_episode["end"] = transition_idx - 1
                index["episodes"].append(current_episode)
                index["episode_count"] += 1

            # Set total transition count
            index["transition_count"] = transition_idx

        # Save the index
        with open(index_file, "w") as f:
            json.dump(index, f)

        return index

    finally:
        # Release the lock if used
        if use_filelock:
            lock.release()
            # Clean up the lock file if it exists
            try:
                if lock_file.exists():
                    os.unlink(lock_file)
            except BaseException:
                pass


def create_index_for_directory(directory_path, output_path=None, recursive=False):
    """
    Creates an index for all experience files in a directory.

    Args:
        directory_path: Path to directory containing experience files
        output_path: Path to save the index to. If None, will be saved in the directory
        recursive: Whether to search recursively through subdirectories

    Returns:
        Path to the created index file
    """
    import json
    from pathlib import Path

    directory_path = Path(directory_path)
    if not directory_path.exists() or not directory_path.is_dir():
        raise ValueError(
            f"Directory {directory_path} does not exist or is not a directory"
        )

    if output_path is None:
        output_path = directory_path / "experience_index.json"
    else:
        output_path = Path(output_path)

    index = {}

    pattern = "**/*.exp" if recursive else "*.exp"
    for exp_file in directory_path.glob(pattern):
        try:
            file_index = index_experience_file(str(exp_file))
            index[str(exp_file.relative_to(directory_path))] = file_index
        except Exception as e:
            print(f"Error indexing {exp_file}: {e}")

    with open(output_path, "w") as f:
        json.dump(index, f)

    return output_path
