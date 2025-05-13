"""
Format conversion utilities for importing and exporting experience data between
different RL framework formats.

Supports:
- RLDS TFRecord format
- D4RL HDF format
- OpenAI Gym recordings
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FormatConverter(ABC):
    """
    Base class for format converters that handle import/export between different
    RL data formats and our internal representation.
    """

    @abstractmethod
    def import_data(
        self, source_path: str | Path, target_path: str | Path, **kwargs
    ) -> Path:
        """
        Import data from an external format to our internal format.

        Args:
            source_path: Path to the source data file/directory
            target_path: Path where to save the converted data
            **kwargs: Format-specific options

        Returns:
            Path to the converted data
        """

    @abstractmethod
    def export_data(
        self, source_path: str | Path, target_path: str | Path, **kwargs
    ) -> Path:
        """
        Export data from our internal format to an external format.

        Args:
            source_path: Path to our internal format data
            target_path: Path where to save the exported data
            **kwargs: Format-specific options

        Returns:
            Path to the exported data
        """

    @staticmethod
    def _ensure_directory(path: str | Path) -> Path:
        """Ensure the directory exists."""
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)
        return path


class RLDSConverter(FormatConverter):
    """
    Converter for RLDS (Reverb-based RL Datasets) TFRecord format.

    RLDS is a common format used by TF-Agents and other TensorFlow-based RL frameworks.
    """

    def import_data(
        self, source_path: str | Path, target_path: str | Path, **kwargs
    ) -> Path:
        """
        Import RLDS TFRecord data to our internal format.

        Args:
            source_path: Path to the RLDS data directory or file
            target_path: Path to save the converted data
            **kwargs: Additional options:
                - batch_size: Batch size for conversion (default: 1000)
                - compression: Compression format to use (default: 'zstd')

        Returns:
            Path to the converted data
        """
        try:
            import tensorflow as tf
        except ImportError:
            logger.error("TensorFlow is not installed. Cannot convert RLDS format.")
            raise

        batch_size = kwargs.get("batch_size", 1000)
        kwargs.get("compression", "zstd")

        target_path = self._ensure_directory(target_path)

        logger.info(f"Importing RLDS data from {source_path} to {target_path}")

        # Load RLDS dataset
        rlds_dataset = tf.data.TFRecordDataset(
            [str(source_path)], compression_type=kwargs.get("rlds_compression", "")
        )

        # Define schema mapping between RLDS and our format
        schema_mapping = self._get_schema_mapping(kwargs.get("schema_mapping", {}))

        # Process in batches to avoid loading everything into memory
        from .storage import ExperienceWriter  # Import here to avoid circular imports

        with ExperienceWriter(target_path) as writer:
            for i, batch in enumerate(rlds_dataset.batch(batch_size)):
                # Convert batch to our format
                converted_batch = self._convert_rlds_batch(batch, schema_mapping)
                writer.write_batch(converted_batch)

                if i % 10 == 0:
                    logger.info(f"Processed {i * batch_size} transitions")

        logger.info(f"RLDS import complete. Data saved to {target_path}")
        return target_path

    def export_data(
        self, source_path: str | Path, target_path: str | Path, **kwargs
    ) -> Path:
        """
        Export our internal format to RLDS TFRecord format.

        Args:
            source_path: Path to our internal format data
            target_path: Path to save the RLDS data
            **kwargs: Additional options:
                - batch_size: Batch size for conversion (default: 1000)
                - rlds_compression: Compression for RLDS (default: None)

        Returns:
            Path to the exported RLDS data
        """
        try:
            import tensorflow as tf
        except ImportError:
            logger.error("TensorFlow is not installed. Cannot convert to RLDS format.")
            raise

        batch_size = kwargs.get("batch_size", 1000)
        target_path = self._ensure_directory(target_path)

        logger.info(
            f"Exporting data from {source_path} to RLDS format at {target_path}"
        )

        # Create TFRecord writer
        options = tf.io.TFRecordOptions(
            compression_type=kwargs.get("rlds_compression", "")
        )
        writer = tf.io.TFRecordWriter(str(target_path), options=options)

        # Define schema mapping between our format and RLDS
        schema_mapping = self._get_schema_mapping(kwargs.get("schema_mapping", {}))

        # Process in batches
        from .storage import ExperienceReader  # Import here to avoid circular imports

        with ExperienceReader(source_path) as reader:
            batch_count = 0
            for batch in reader.iter_batches(batch_size):
                # Convert batch to RLDS format
                tf_examples = self._convert_to_rlds_batch(batch, schema_mapping)

                # Write to TFRecord
                for example in tf_examples:
                    writer.write(example.SerializeToString())

                batch_count += 1
                if batch_count % 10 == 0:
                    logger.info(f"Processed {batch_count * batch_size} transitions")

        writer.close()
        logger.info(f"Export to RLDS format complete. Data saved to {target_path}")
        return target_path

    def _get_schema_mapping(self, user_mapping: dict) -> dict:
        """
        Get schema mapping between RLDS and our format.
        Default mappings can be overridden by providing user_mapping.

        Returns:
            Dictionary mapping our fields to RLDS fields
        """
        # Default mapping
        default_mapping = {
            "observation": "observation",
            "action": "action",
            "reward": "reward",
            "next_observation": "next_observation",
            "done": "done",
            "info": "info",
        }

        # Override with user mapping
        default_mapping.update(user_mapping)
        return default_mapping

    def _convert_rlds_batch(self, batch, schema_mapping: dict) -> list[dict[str, Any]]:
        """Convert RLDS batch to our format."""
        # Implementation depends on exact RLDS structure
        # This is a placeholder implementation
        result = []
        for i in range(len(batch)):
            entry = {}
            for our_field, rlds_field in schema_mapping.items():
                if rlds_field in batch:
                    entry[our_field] = batch[rlds_field][i].numpy()
            result.append(entry)
        return result

    def _convert_to_rlds_batch(self, batch, schema_mapping: dict) -> list:
        """Convert our batch to RLDS format TF Examples."""
        try:
            import tensorflow as tf
        except ImportError:
            logger.error("TensorFlow is not installed. Cannot convert to RLDS format.")
            raise

        examples = []
        for entry in batch:
            feature_dict = {}

            # Map each field according to schema
            for our_field, rlds_field in schema_mapping.items():
                if our_field in entry:
                    # Convert data to TF Feature based on type
                    feature = self._to_tf_feature(entry[our_field])
                    if feature is not None:
                        feature_dict[rlds_field] = feature

            # Create TF Example
            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            examples.append(example)

        return examples

    def _to_tf_feature(self, data: Any) -> Any:
        """Convert data to TensorFlow Feature."""
        try:
            import tensorflow as tf
        except ImportError:
            logger.error("TensorFlow is not installed. Cannot convert to TF Feature.")
            raise

        # Handle different data types
        if isinstance(
            data,
            (
                int,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[data]))

        elif isinstance(data, (float, np.float32, np.float64)):
            return tf.train.Feature(float_list=tf.train.FloatList(value=[data]))

        elif isinstance(data, (str, bytes)):
            if isinstance(data, str):
                data = data.encode("utf-8")
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))

        elif isinstance(data, np.ndarray):
            if data.dtype in (np.float32, np.float64):
                return tf.train.Feature(
                    float_list=tf.train.FloatList(value=data.flatten())
                )
            elif data.dtype in (
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ):
                return tf.train.Feature(
                    int64_list=tf.train.Int64List(value=data.flatten())
                )
            else:
                logger.warning(f"Unsupported numpy dtype: {data.dtype}")
                return None

        elif isinstance(data, (list, tuple)):
            if all(
                isinstance(
                    x,
                    (
                        int,
                        np.int8,
                        np.int16,
                        np.int32,
                        np.int64,
                        np.uint8,
                        np.uint16,
                        np.uint32,
                        np.uint64,
                    ),
                )
                for x in data
            ):
                return tf.train.Feature(int64_list=tf.train.Int64List(value=data))
            elif all(isinstance(x, (float, np.float32, np.float64)) for x in data):
                return tf.train.Feature(float_list=tf.train.FloatList(value=data))
            elif all(isinstance(x, (str, bytes)) for x in data):
                bytes_data = [
                    x.encode("utf-8") if isinstance(x, str) else x for x in data
                ]
                return tf.train.Feature(bytes_list=tf.train.BytesList(value=bytes_data))
            else:
                logger.warning(f"Mixed types in list/tuple not supported")
                return None

        else:
            logger.warning(f"Unsupported data type: {type(data)}")
            return None


class D4RLConverter(FormatConverter):
    """
    Converter for D4RL HDF5 format.

    D4RL is a common benchmark format for offline RL algorithms.
    """

    def import_data(
        self, source_path: str | Path, target_path: str | Path, **kwargs
    ) -> Path:
        """
        Import D4RL HDF5 data to our internal format.

        Args:
            source_path: Path to the D4RL HDF5 file
            target_path: Path to save the converted data
            **kwargs: Additional options:
                - batch_size: Batch size for conversion (default: 1000)
                - compression: Compression format to use (default: 'zstd')

        Returns:
            Path to the converted data
        """
        try:
            import h5py
        except ImportError:
            logger.error("h5py is not installed. Cannot convert D4RL format.")
            raise

        batch_size = kwargs.get("batch_size", 1000)
        kwargs.get("compression", "zstd")

        target_path = self._ensure_directory(target_path)

        logger.info(f"Importing D4RL data from {source_path} to {target_path}")

        # Load D4RL dataset
        with h5py.File(source_path, "r") as f:
            # Extract dataset info
            n_transitions = len(f["observations"])

            # Define schema mapping between D4RL and our format
            schema_mapping = self._get_schema_mapping(kwargs.get("schema_mapping", {}))

            # Process in batches
            from .storage import (
                ExperienceWriter,  # Import here to avoid circular imports
            )

            with ExperienceWriter(target_path) as writer:
                for i in range(0, n_transitions, batch_size):
                    end_idx = min(i + batch_size, n_transitions)

                    # Extract batch data from D4RL
                    batch_data = {}
                    for d4rl_field in schema_mapping.values():
                        if d4rl_field in f and i < len(f[d4rl_field]):
                            batch_data[d4rl_field] = f[d4rl_field][i:end_idx]

                    # Convert to our format
                    converted_batch = self._convert_d4rl_batch(
                        batch_data, schema_mapping
                    )
                    writer.write_batch(converted_batch)

                    if i % (batch_size * 10) == 0:
                        logger.info(f"Processed {i}/{n_transitions} transitions")

        logger.info(f"D4RL import complete. Data saved to {target_path}")
        return target_path

    def export_data(
        self, source_path: str | Path, target_path: str | Path, **kwargs
    ) -> Path:
        """
        Export our internal format to D4RL HDF5 format.

        Args:
            source_path: Path to our internal format data
            target_path: Path to save the D4RL data
            **kwargs: Additional options:
                - batch_size: Batch size for conversion (default: 1000)

        Returns:
            Path to the exported D4RL data
        """
        try:
            import h5py
        except ImportError:
            logger.error("h5py is not installed. Cannot convert to D4RL format.")
            raise

        batch_size = kwargs.get("batch_size", 1000)
        target_path = self._ensure_directory(target_path)

        logger.info(
            f"Exporting data from {source_path} to D4RL format at {target_path}"
        )

        # Create HDF5 file
        with h5py.File(target_path, "w") as f:
            # Define schema mapping between our format and D4RL
            schema_mapping = self._get_schema_mapping(kwargs.get("schema_mapping", {}))

            # Process in batches
            from .storage import (
                ExperienceReader,  # Import here to avoid circular imports
            )

            # First pass to determine dataset sizes and types
            field_dtypes = {}
            field_shapes = {}
            total_transitions = 0

            with ExperienceReader(source_path) as reader:
                # Sample first batch to determine shapes and types
                first_batch = next(reader.iter_batches(1))
                if first_batch:
                    first_entry = first_batch[0]
                    for our_field, d4rl_field in schema_mapping.items():
                        if our_field in first_entry:
                            data = first_entry[our_field]
                            if isinstance(data, np.ndarray):
                                field_dtypes[d4rl_field] = data.dtype
                                field_shapes[d4rl_field] = (None,) + data.shape
                            else:
                                # Convert Python types to numpy types
                                if isinstance(data, int):
                                    field_dtypes[d4rl_field] = np.int64
                                elif isinstance(data, float):
                                    field_dtypes[d4rl_field] = np.float32
                                elif isinstance(data, bool):
                                    field_dtypes[d4rl_field] = np.bool_
                                else:
                                    field_dtypes[d4rl_field] = np.dtype("O")
                                field_shapes[d4rl_field] = (None,)

                # Count total transitions
                total_transitions = reader.get_total_transitions()

            # Create datasets in the HDF5 file
            datasets = {}
            for field, dtype in field_dtypes.items():
                shape = list(field_shapes[field])
                shape[0] = total_transitions  # Set correct first dimension

                # Create dataset with chunking for efficiency
                chunk_size = min(batch_size, total_transitions)
                chunk_shape = list(shape)
                chunk_shape[0] = chunk_size

                datasets[field] = f.create_dataset(
                    field, shape=shape, dtype=dtype, chunks=tuple(chunk_shape)
                )

            # Second pass to fill datasets
            with ExperienceReader(source_path) as reader:
                offset = 0
                for batch in reader.iter_batches(batch_size):
                    # Convert batch to D4RL format and write to HDF5
                    batch_size_actual = len(batch)

                    # Process each field separately
                    for our_field, d4rl_field in schema_mapping.items():
                        if d4rl_field in datasets:
                            field_data = []
                            for entry in batch:
                                if our_field in entry:
                                    field_data.append(entry[our_field])

                            if field_data:
                                # Convert to numpy array with correct dtype
                                field_array = np.array(
                                    field_data, dtype=field_dtypes[d4rl_field]
                                )
                                # Write to HDF5
                                datasets[d4rl_field][
                                    offset : offset + batch_size_actual
                                ] = field_array

                    offset += batch_size_actual
                    if offset % (batch_size * 10) == 0:
                        logger.info(
                            f"Processed {offset}/{total_transitions} transitions"
                        )

            # Add dataset metadata
            f.attrs["total_transitions"] = total_transitions
            # Add any additional metadata
            metadata = kwargs.get("metadata", {})
            for key, value in metadata.items():
                f.attrs[key] = value

        logger.info(f"Export to D4RL format complete. Data saved to {target_path}")
        return target_path

    def _get_schema_mapping(self, user_mapping: dict) -> dict:
        """
        Get schema mapping between our format and D4RL.
        Default mappings can be overridden by providing user_mapping.

        Returns:
            Dictionary mapping our fields to D4RL fields
        """
        # Default mapping
        default_mapping = {
            "observation": "observations",
            "action": "actions",
            "reward": "rewards",
            "next_observation": "next_observations",
            "done": "terminals",
            "info": "infos",
        }

        # Override with user mapping
        default_mapping.update(user_mapping)
        return default_mapping

    def _convert_d4rl_batch(
        self, batch_data: dict[str, np.ndarray], schema_mapping: dict
    ) -> list[dict[str, Any]]:
        """Convert D4RL batch to our format."""
        # Reverse the mapping to go from D4RL fields to our fields
        reverse_mapping = {
            d4rl_field: our_field for our_field, d4rl_field in schema_mapping.items()
        }

        result = []
        batch_size = 0

        # Find the batch size from any field
        for field, data in batch_data.items():
            if len(data) > 0:
                batch_size = len(data)
                break

        # Create entries for each transition
        for i in range(batch_size):
            entry = {}
            for d4rl_field, data in batch_data.items():
                if d4rl_field in reverse_mapping and i < len(data):
                    our_field = reverse_mapping[d4rl_field]
                    entry[our_field] = data[i]
            result.append(entry)

        return result


class GymRecordingConverter(FormatConverter):
    """
    Converter for OpenAI Gym recordings.

    Handles the standard format used by gym.wrappers.RecordVideo and similar wrappers.
    """

    def import_data(
        self, source_path: str | Path, target_path: str | Path, **kwargs
    ) -> Path:
        """
        Import Gym recordings to our internal format.

        Args:
            source_path: Path to the Gym recording directory or file
            target_path: Path to save the converted data
            **kwargs: Additional options:
                - video: Whether to import video data (default: False)
                - compression: Compression format to use (default: 'zstd')

        Returns:
            Path to the converted data
        """
        import json

        source_path = Path(source_path)
        target_path = self._ensure_directory(target_path)

        logger.info(f"Importing Gym recordings from {source_path} to {target_path}")

        # Check if we're importing a directory or a single file
        if source_path.is_dir():
            # Find all JSON files in the directory
            json_files = list(source_path.glob("*.json"))
            if not json_files:
                raise ValueError(f"No JSON files found in {source_path}")
        else:
            if source_path.suffix != ".json":
                raise ValueError(f"Expected a JSON file, got {source_path}")
            json_files = [source_path]

        # Import each JSON file
        from .storage import ExperienceWriter  # Import here to avoid circular imports

        with ExperienceWriter(target_path) as writer:
            for json_file in json_files:
                with open(json_file) as f:
                    episode_data = json.load(f)

                # Convert episode data to our format
                converted_episode = self._convert_gym_episode(episode_data)
                writer.write_batch(converted_episode)

        logger.info(f"Gym recordings import complete. Data saved to {target_path}")
        return target_path

    def export_data(
        self, source_path: str | Path, target_path: str | Path, **kwargs
    ) -> Path:
        """
        Export our internal format to Gym recording format.

        Args:
            source_path: Path to our internal format data
            target_path: Path where to save the Gym recording
            **kwargs: Additional options:
                - episode_per_file: Whether to save each episode in a separate file (default: True)

        Returns:
            Path to the exported Gym recordings
        """
        import json

        target_path = self._ensure_directory(target_path)
        os.makedirs(target_path, exist_ok=True)

        logger.info(f"Exporting data from {source_path} to Gym format at {target_path}")

        # Process in episodes
        from .storage import ExperienceReader  # Import here to avoid circular imports

        episode_per_file = kwargs.get("episode_per_file", True)

        with ExperienceReader(source_path) as reader:
            # Group transitions by episode
            episodes = {}
            current_episode_id = None
            current_episode = []

            for batch in reader.iter_batches(
                1000
            ):  # Process in larger batches for efficiency
                for transition in batch:
                    # Detect episode boundaries
                    episode_id = transition.get("episode_id", 0)
                    if current_episode_id is None:
                        current_episode_id = episode_id

                    if episode_id != current_episode_id:
                        # Save completed episode
                        episodes[current_episode_id] = current_episode
                        current_episode = []
                        current_episode_id = episode_id

                    current_episode.append(transition)

            # Save last episode if any
            if current_episode:
                episodes[current_episode_id] = current_episode

            # Convert and save episodes
            for episode_id, episode_data in episodes.items():
                # Convert to Gym format
                gym_episode = self._convert_to_gym_episode(episode_data)

                if episode_per_file:
                    # Save each episode to a separate file
                    filename = f"episode_{episode_id}.json"
                    with open(target_path / filename, "w") as f:
                        json.dump(gym_episode, f, cls=NumpyEncoder)
                else:
                    # Append to a single file
                    filename = "episodes.json"
                    with open(target_path / filename, "a") as f:
                        f.write(json.dumps(gym_episode, cls=NumpyEncoder) + "\n")

        logger.info(f"Export to Gym format complete. Data saved to {target_path}")
        return target_path

    def _convert_gym_episode(self, episode_data: dict) -> list[dict[str, Any]]:
        """Convert Gym episode data to our format."""
        # Extract transitions from the episode
        transitions = []

        # Handle different Gym recording formats
        if "steps" in episode_data:
            # Standard Gym format
            steps = episode_data["steps"]
            episode_id = episode_data.get("episode_id", 0)

            for i, step in enumerate(steps):
                # Extract data from step
                observation = step.get("observation")
                action = step.get("action")
                reward = step.get("reward", 0.0)
                done = step.get("done", False)
                info = step.get("info", {})

                # Create transition
                transition = {
                    "observation": observation,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "episode_id": episode_id,
                    "step_id": i,
                }

                # Add next observation if available
                if i < len(steps) - 1:
                    transition["next_observation"] = steps[i + 1].get("observation")
                else:
                    # For the last step, next_observation might be in the episode data
                    transition["next_observation"] = episode_data.get(
                        "final_observation", observation
                    )

                transitions.append(transition)

        # Handle other formats as needed

        return transitions

    def _convert_to_gym_episode(self, episode_data: list[dict[str, Any]]) -> dict:
        """Convert our episode data to Gym format."""
        # Create Gym episode structure
        gym_episode = {
            "steps": [],
            "episode_id": episode_data[0].get("episode_id", 0) if episode_data else 0,
            "return": sum(step.get("reward", 0) for step in episode_data),
            "length": len(episode_data),
        }

        # Convert each transition to a step
        for transition in episode_data:
            step = {
                "observation": transition.get("observation"),
                "action": transition.get("action"),
                "reward": transition.get("reward", 0.0),
                "done": transition.get("done", False),
                "info": transition.get("info", {}),
            }
            gym_episode["steps"].append(step)

        # Add final observation
        if episode_data:
            gym_episode["final_observation"] = episode_data[-1].get("next_observation")

        return gym_episode


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy arrays and datatypes."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def get_converter(format_name: str) -> FormatConverter:
    """
    Get a converter for the specified format.

    Args:
        format_name: Name of the format ('rlds', 'd4rl', 'gym')

    Returns:
        A FormatConverter instance
    """
    converters = {
        "rlds": RLDSConverter(),
        "d4rl": D4RLConverter(),
        "gym": GymRecordingConverter(),
    }

    if format_name.lower() not in converters:
        raise ValueError(
            f"Unknown format: {format_name}. Supported formats: {list(converters.keys())}"
        )

    return converters[format_name.lower()]


"""
Utilities for converting between different formats.
"""


def convert_dimacs_to_cnf(dimacs_string):
    """
    Convert DIMACS format to CNF representation.

    Args:
        dimacs_string: String containing DIMACS format

    Returns:
        CNF representation
    """


def convert_cnf_to_dimacs(cnf):
    """
    Convert CNF representation to DIMACS format.

    Args:
        cnf: CNF representation

    Returns:
        String in DIMACS format
    """
