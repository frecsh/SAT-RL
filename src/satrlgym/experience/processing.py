"""
Processing utilities for experience data.

This module provides functions for processing, filtering, and transforming
experience data collected from SAT environments.
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


class ExperienceProcessor:
    """Base class for experience data processors."""

    def __init__(self):
        """Initialize the processor."""

    def process(self, experience_batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Process a batch of experience data.

        Args:
            experience_batch: List of experience dictionaries

        Returns:
            Processed list of experience dictionaries
        """
        return experience_batch


class ExperienceTransform:
    """Base class for transformations applied to experience data."""

    def __init__(self):
        """Initialize the transform with an empty list of transforms."""
        self.transforms = []

    def __call__(self, experience_batch):
        """Apply the transformation to experience data"""
        return self.transform(experience_batch)

    def transform(self, experience_batch):
        """Transform experience data by applying all transforms in sequence"""
        result = experience_batch
        for transform_fn in self.transforms:
            result = transform_fn(result)
        return result

    def add_standardize(self, fields):
        """
        Add standardization transform for specified fields.

        Args:
            fields: List of field names to standardize
        """

        def standardize_transform(batch):
            if not batch:
                return batch

            # Collect data for each field
            field_data = {field: [] for field in fields}
            for exp in batch:
                for field in fields:
                    if field in exp and isinstance(
                        exp[field], (np.ndarray, list, float, int)
                    ):
                        field_data[field].append(exp[field])

            # Compute statistics and standardize each field
            stats = {}
            for field in fields:
                if not field_data[field]:
                    continue

                # Convert to numpy array
                data = np.array(field_data[field])

                # Standardize the data
                standardized, mean, std = standardize_data(data)
                stats[field] = {"mean": mean, "std": std}

                # Update experiences with standardized values
                idx = 0
                for exp in batch:
                    if field in exp and isinstance(
                        exp[field], (np.ndarray, list, float, int)
                    ):
                        exp[field] = standardized[idx]
                        idx += 1

            return batch

        self.transforms.append(standardize_transform)
        return self

    def add_n_step_returns(
        self,
        reward_field="rewards",
        done_field="dones",
        output_field="returns",
        n_steps=1,
        gamma=0.99,
        value_field=None,
    ):
        """
        Add n-step return calculation transform.

        Args:
            reward_field: Field name for rewards
            done_field: Field name for done flags
            output_field: Field name to store calculated returns
            n_steps: Number of steps to look ahead
            gamma: Discount factor
            value_field: Optional field name for value estimates

        Returns:
            Self for method chaining
        """

        def n_step_returns_transform(batch):
            if not batch:
                return batch

            # Extract rewards and dones from batch
            rewards_list = [exp.get(reward_field, 0.0) for exp in batch]
            dones_list = [exp.get(done_field, False) for exp in batch]

            # Get values if provided, otherwise use zeros
            if value_field:
                values_list = [exp.get(value_field, 0.0) for exp in batch]
            else:
                values_list = [0.0 for _ in batch]

            # Convert to numpy arrays
            rewards = np.array(rewards_list)
            dones = np.array(dones_list, dtype=bool)
            values = np.array(values_list)

            # Reshape for calculate_n_step_returns
            batch_size = 1
            time_steps = len(rewards)
            rewards_reshaped = rewards.reshape(batch_size, time_steps)
            dones_reshaped = dones.reshape(batch_size, time_steps)
            values_reshaped = values.reshape(batch_size, time_steps)

            # Calculate n-step returns
            returns = calculate_n_step_returns(
                rewards_reshaped, values_reshaped, dones_reshaped, gamma, n_steps
            )

            # Add returns to each experience
            for i, exp in enumerate(batch):
                exp[output_field] = returns[0, i]  # Extract from batch dimension

            return batch

        self.transforms.append(n_step_returns_transform)
        return self

    def add_reward_scaling(
        self,
        reward_field="reward",
        scale=1.0,
        shift=0.0,
        scaling_method="linear",
        clip_min=None,
        clip_max=None,
    ):
        """
        Add reward scaling transform.

        Args:
            reward_field: Field name for rewards
            scale: Scaling factor
            shift: Value to add after scaling
            scaling_method: Method to use for scaling ('linear', 'clip', 'tanh', 'sign', 'scale', 'standardize')
            clip_min: Minimum value for clipping
            clip_max: Maximum value for clipping

        Returns:
            Self for method chaining
        """

        def reward_scaling_transform(batch):
            if not batch:
                return batch

            # Extract rewards
            rewards = np.array([exp.get(reward_field, 0.0) for exp in batch])

            # Apply scaling
            scaled_rewards = scale_rewards(
                rewards, scaling_method, scale, shift, clip_min, clip_max
            )

            # Update batch with scaled rewards
            for i, exp in enumerate(batch):
                if reward_field in exp:
                    exp[reward_field] = scaled_rewards[i]

            return batch

        self.transforms.append(reward_scaling_transform)
        return self


class RewardNormalizer(ExperienceProcessor):
    """Normalizes rewards in experience data to have zero mean and unit variance."""

    def __init__(self, clip_range: tuple[float, float] | None = None):
        """
        Initialize the reward normalizer.

        Args:
            clip_range: Optional range to clip normalized rewards, e.g. (-5, 5)
        """
        super().__init__()
        self.scaler = StandardScaler()
        self.clip_range = clip_range
        self.is_fitted = False

    def process(self, experience_batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Normalize rewards in the experience batch.

        Args:
            experience_batch: List of experience dictionaries

        Returns:
            Batch with normalized rewards
        """
        if not experience_batch:
            return experience_batch

        # Extract rewards
        rewards = np.array(
            [exp.get("reward", 0.0) for exp in experience_batch]
        ).reshape(-1, 1)

        # Fit scaler if not already fitted
        if not self.is_fitted:
            self.scaler.fit(rewards)
            self.is_fitted = True

        # Transform rewards
        normalized_rewards = self.scaler.transform(rewards).flatten()

        # Clip if specified
        if self.clip_range:
            normalized_rewards = np.clip(
                normalized_rewards, self.clip_range[0], self.clip_range[1]
            )

        # Update experience with normalized rewards
        result = []
        for i, exp in enumerate(experience_batch):
            exp_copy = exp.copy()
            exp_copy["reward"] = normalized_rewards[i]
            exp_copy["original_reward"] = exp.get(
                "reward", 0.0
            )  # Store original for reference
            result.append(exp_copy)

        return result

    def reset(self):
        """Reset the internal state of the normalizer."""
        self.scaler = StandardScaler()
        self.is_fitted = False


class ObservationNormalizer(ExperienceProcessor):
    """Normalizes observations in experience data."""

    def __init__(self, observation_keys: list[str] = None, use_min_max: bool = False):
        """
        Initialize the observation normalizer.

        Args:
            observation_keys: Keys of observation fields to normalize
                             (None means normalize all numeric arrays)
            use_min_max: If True, use min-max scaling instead of standardization
        """
        super().__init__()
        self.observation_keys = observation_keys
        self.scalers = {}
        self.is_fitted = False
        self.scaler_class = MinMaxScaler if use_min_max else StandardScaler

    def process(self, experience_batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Normalize observations in the experience batch.

        Args:
            experience_batch: List of experience dictionaries

        Returns:
            Batch with normalized observations
        """
        if not experience_batch:
            return experience_batch

        # Determine keys to normalize if not specified
        if not self.observation_keys:
            # Look at the first experience entry to find numeric array fields
            sample = experience_batch[0].get("observation", {})
            if isinstance(sample, dict):
                self.observation_keys = [
                    key
                    for key, value in sample.items()
                    if isinstance(value, np.ndarray) and value.dtype.kind in "iuf"
                ]
            else:
                # If observation is not a dictionary, normalize the whole array
                self.observation_keys = ["observation"]

        result = []
        for exp in experience_batch:
            exp_copy = exp.copy()

            # Handle nested observation dictionary
            if "observation" in exp and isinstance(exp["observation"], dict):
                obs_copy = exp["observation"].copy()

                for key in self.observation_keys:
                    if key in obs_copy and isinstance(obs_copy[key], np.ndarray):
                        # Get or create scaler for this key
                        if key not in self.scalers:
                            self.scalers[key] = self.scaler_class()

                        # Flatten for scaling
                        original_shape = obs_copy[key].shape
                        flattened = obs_copy[key].reshape(-1, 1)

                        # Fit scaler if not already fitted
                        if not self.is_fitted:
                            self.scalers[key].fit(flattened)

                        # Transform and reshape back
                        obs_copy[key] = (
                            self.scalers[key]
                            .transform(flattened)
                            .reshape(original_shape)
                        )

                exp_copy["observation"] = obs_copy

            # Handle flat observation array
            elif "observation" in exp and isinstance(exp["observation"], np.ndarray):
                key = "observation"
                if key not in self.scalers:
                    self.scalers[key] = self.scaler_class()

                original_shape = exp["observation"].shape
                flattened = exp["observation"].reshape(-1, 1)

                if not self.is_fitted:
                    self.scalers[key].fit(flattened)

                exp_copy["observation"] = (
                    self.scalers[key].transform(flattened).reshape(original_shape)
                )

            result.append(exp_copy)

        # Mark as fitted after first batch
        self.is_fitted = True
        return result

    def reset(self):
        """Reset the internal state of the normalizer."""
        self.scalers = {}
        self.is_fitted = False


class ExperienceFilter(ExperienceProcessor):
    """Filters experience data based on specified criteria."""

    def __init__(self, filter_fn: Callable[[dict[str, Any]], bool]):
        """
        Initialize the filter.

        Args:
            filter_fn: Function that takes an experience entry and returns
                      True if it should be kept, False if it should be filtered out
        """
        super().__init__()
        self.filter_fn = filter_fn

    def process(self, experience_batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Filter the experience batch.

        Args:
            experience_batch: List of experience dictionaries

        Returns:
            Filtered list of experience dictionaries
        """
        return [exp for exp in experience_batch if self.filter_fn(exp)]


class CompositeProcessor(ExperienceProcessor):
    """Applies multiple processors in sequence."""

    def __init__(self, processors: list[ExperienceProcessor]):
        """
        Initialize the composite processor.

        Args:
            processors: List of processors to apply in sequence
        """
        super().__init__()
        self.processors = processors

    def process(self, experience_batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Process the experience batch through all processors.

        Args:
            experience_batch: List of experience dictionaries

        Returns:
            Processed list of experience dictionaries
        """
        result = experience_batch
        for processor in self.processors:
            result = processor.process(result)
        return result

    def reset(self):
        """Reset all processors."""
        for processor in self.processors:
            if hasattr(processor, "reset"):
                processor.reset()


def filter_by_reward_threshold(threshold: float, higher_is_better: bool = True):
    """
    Create a filter function that keeps experiences with rewards above/below a threshold.

    Args:
        threshold: Reward threshold
        higher_is_better: If True, keep rewards >= threshold; otherwise keep rewards <= threshold

    Returns:
        Filter function
    """
    if higher_is_better:
        return lambda exp: exp.get("reward", 0.0) >= threshold
    else:
        return lambda exp: exp.get("reward", 0.0) <= threshold


def filter_by_done(keep_done_only: bool = True):
    """
    Create a filter function that keeps experiences based on their done flag.

    Args:
        keep_done_only: If True, keep only experiences where done=True;
                       if False, keep only experiences where done=False

    Returns:
        Filter function
    """
    return lambda exp: exp.get("done", False) == keep_done_only


def process_experience(
    experience_data: list[dict[str, Any]], processors: list[ExperienceProcessor] = None
) -> list[dict[str, Any]]:
    """
    Process experience data using specified processors.

    Args:
        experience_data: Raw experience data to process
        processors: List of processors to apply (if None, apply default processors)

    Returns:
        Processed experience data
    """
    if processors is None:
        # Default processing: normalize rewards and observations
        processors = [RewardNormalizer(clip_range=(-10, 10)), ObservationNormalizer()]

    processor = CompositeProcessor(processors)
    return processor.process(experience_data)


def calculate_n_step_returns(rewards, next_values, dones, gamma=0.99, n_steps=1):
    """
    Calculate n-step returns for a batch of experiences.

    Args:
        rewards: List or array of rewards [batch_size, time_steps]
        next_values: List or array of value estimates for next states [batch_size, time_steps]
        dones: List or array of done flags [batch_size, time_steps]
        gamma: Discount factor
        n_steps: Number of steps to look ahead

    Returns:
        N-step returns of shape [batch_size, time_steps]
    """
    import numpy as np

    # Convert inputs to numpy arrays if they're not already
    rewards = np.array(rewards)
    next_values = np.array(next_values)
    dones = np.array(dones, dtype=np.bool_)

    batch_size, time_steps = rewards.shape
    returns = np.zeros_like(rewards)

    for t in range(time_steps):
        n_step_return = np.zeros(batch_size)

        # Initialize with immediate reward
        n_step_return = rewards[:, t].copy()

        # For each future step within n_steps
        future_val = 0
        for step in range(1, n_steps + 1):
            if t + step < time_steps:
                # Apply discount and add the reward if not done
                discount = gamma**step
                mask = ~dones[:, t + step - 1]  # Use previous step's done flag
                n_step_return[mask] += discount * rewards[:, t + step][mask]

                # For the last step, add the discounted next state value
                if step == n_steps:
                    future_mask = mask & ~dones[:, t + step]
                    future_val = discount * gamma * next_values[:, t + step]
                    n_step_return[future_mask] += future_val[future_mask]

        returns[:, t] = n_step_return

    return returns


def normalize_data(
    data: np.ndarray,
    fields_to_normalize: list[str] = None,
    min_val: float = 0.0,
    max_val: float = 1.0,
    per_feature: bool = True,
    eps: float = 1e-8,
    clip_range: tuple[float, float] | None = None,
    stats: dict[str, dict[str, np.ndarray]] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """
    Normalize data to specified range, typically [0,1] or [-1,1].

    Args:
        data: Input dictionary with arrays to normalize or a single array
        fields_to_normalize: List of field names to normalize (if data is a dict)
        min_val: Target minimum value after normalization
        max_val: Target maximum value after normalization
        per_feature: Whether to normalize each feature independently
        eps: Small constant to avoid division by zero
        clip_range: Optional range to clip values post-normalization
        stats: Pre-computed statistics from previous normalization to reuse

    Returns:
        Tuple of (normalized_data, stats_dict)
    """
    # Handle single array case
    if isinstance(data, np.ndarray):
        if stats is not None and "min" in stats and "max" in stats:
            return _normalize_array_with_stats(
                data, stats["min"], stats["max"], min_val, max_val, clip_range
            )
        else:
            return _normalize_array(
                data, min_val, max_val, per_feature, eps, clip_range
            )

    # Handle dictionary case
    if not fields_to_normalize:
        fields_to_normalize = list(data.keys())

    normalized_data = {}
    result_stats = {}

    for field in fields_to_normalize:
        if field in data and isinstance(data[field], np.ndarray):
            # Use pre-computed stats if available
            if stats is not None and field in stats:
                field_stats = stats[field]
                normalized_array, field_min, field_max = _normalize_array_with_stats(
                    data[field],
                    field_stats["min"],
                    field_stats["max"],
                    min_val,
                    max_val,
                    clip_range,
                )
            else:
                normalized_array, field_min, field_max = _normalize_array(
                    data[field], min_val, max_val, per_feature, eps, clip_range
                )

            normalized_data[field] = normalized_array
            result_stats[field] = {"min": field_min, "max": field_max}
        else:
            # Keep non-array fields unchanged
            if field in data:
                normalized_data[field] = data[field]

    # Copy any fields not being normalized
    for field in data:
        if field not in normalized_data:
            normalized_data[field] = data[field]

    return normalized_data, result_stats


def _normalize_array(
    array: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 1.0,
    per_feature: bool = True,
    eps: float = 1e-8,
    clip_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Internal helper to normalize a single array."""
    if not isinstance(array, np.ndarray):
        array = np.array(array, dtype=np.float32)

    # Save original shape and ensure 2D for proper axis operations
    original_shape = array.shape
    if array.ndim == 1:
        array = array.reshape(-1, 1)

    # Compute min and max
    if per_feature:
        data_min = np.min(array, axis=0)
        data_max = np.max(array, axis=0)
    else:
        data_min = np.min(array)
        data_max = np.max(array)

    # Handle constant values
    range_diff = data_max - data_min
    if per_feature:
        range_diff = np.maximum(range_diff, eps)
    else:
        range_diff = max(range_diff, eps)

    # Normalize to [0, 1]
    normalized = (array - data_min) / range_diff

    # Scale to [min_val, max_val]
    normalized = normalized * (max_val - min_val) + min_val

    # Clip if range provided
    if clip_range is not None:
        normalized = np.clip(normalized, clip_range[0], clip_range[1])

    # Restore original shape
    if len(original_shape) == 1:
        normalized = normalized.flatten()
    else:
        normalized = normalized.reshape(original_shape)

    return normalized, data_min, data_max


def _normalize_array_with_stats(
    array: np.ndarray,
    data_min: np.ndarray,
    data_max: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 1.0,
    clip_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize array using pre-computed min/max statistics.

    Args:
        array: Input array to normalize
        data_min: Pre-computed minimum values
        data_max: Pre-computed maximum values
        min_val: Target minimum value after normalization
        max_val: Target maximum value after normalization
        clip_range: Optional range to clip values post-normalization

    Returns:
        Tuple of (normalized_array, data_min, data_max)
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array, dtype=np.float32)

    # Save original shape
    original_shape = array.shape
    if array.ndim == 1 and data_min.ndim > 0:
        array = array.reshape(-1, 1)

    # Ensure data_min and data_max have compatible shapes
    if np.isscalar(data_min) and np.isscalar(data_max):
        # Use scalar min/max for the whole array
        range_diff = data_max - data_min
        normalized = (array - data_min) / range_diff
    else:
        # Use per-feature min/max
        range_diff = data_max - data_min
        normalized = (array - data_min) / range_diff

    # Scale to target range
    normalized = normalized * (max_val - min_val) + min_val

    # Clip if range provided
    if clip_range is not None:
        normalized = np.clip(normalized, clip_range[0], clip_range[1])

    # Restore original shape
    if len(original_shape) == 1:
        normalized = normalized.flatten()
    else:
        normalized = normalized.reshape(original_shape)

    return normalized, data_min, data_max


def standardize_data(
    data,
    fields_to_standardize: list[str] = None,
    per_feature: bool = True,
    epsilon: float = 1e-8,
    clip_range: tuple[float, float] | None = None,
    stats: dict[str, dict[str, np.ndarray]] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """
    Standardize data to zero mean and unit variance.

    Args:
        data: Input dictionary with arrays to standardize or a single array
        fields_to_standardize: List of field names to standardize (if data is a dict)
        per_feature: Whether to standardize each feature independently
        epsilon: Small constant to prevent division by zero
        clip_range: Optional tuple of (min, max) to clip the standardized values
        stats: Pre-computed statistics from previous standardization to reuse

    Returns:
        Tuple of (standardized_data, stats_dict)
    """
    # Handle single array case
    if isinstance(data, np.ndarray):
        if stats is not None and "mean" in stats and "std" in stats:
            return _standardize_array_with_stats(
                data, stats["mean"], stats["std"], epsilon, clip_range
            )
        else:
            return _standardize_array(data, per_feature, epsilon, clip_range)

    # Handle dictionary case
    if not fields_to_standardize:
        fields_to_standardize = list(data.keys())

    standardized_data = {}
    result_stats = {}

    for field in fields_to_standardize:
        if field in data and isinstance(data[field], np.ndarray):
            # Use pre-computed stats if available
            if stats is not None and field in stats:
                field_stats = stats[field]
                (
                    standardized_array,
                    field_mean,
                    field_std,
                ) = _standardize_array_with_stats(
                    data[field],
                    field_stats["mean"],
                    field_stats["std"],
                    epsilon,
                    clip_range,
                )
            else:
                standardized_array, field_mean, field_std = _standardize_array(
                    data[field], per_feature, epsilon, clip_range
                )

            standardized_data[field] = standardized_array
            result_stats[field] = {"mean": field_mean, "std": field_std}
        else:
            # Keep non-array fields unchanged
            if field in data:
                standardized_data[field] = data[field]

    # Copy any fields not being standardized
    for field in data:
        if field not in standardized_data:
            standardized_data[field] = data[field]

    return standardized_data, result_stats


def _standardize_array(
    array: np.ndarray,
    per_feature: bool = True,
    epsilon: float = 1e-8,
    clip_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Internal helper to standardize a single array."""
    if not isinstance(array, np.ndarray):
        array = np.array(array, dtype=np.float32)

    # Save original shape and ensure 2D for proper axis operations
    original_shape = array.shape
    if array.ndim == 1:
        array = array.reshape(-1, 1)

    # Compute mean and std
    if per_feature:
        mean = np.mean(array, axis=0)
        std = np.std(array, axis=0)
    else:
        mean = np.mean(array)
        std = np.std(array)

    # Avoid division by zero
    std = np.maximum(std, epsilon)

    # Standardize data
    standardized = (array - mean) / std

    # Clip values if requested
    if clip_range is not None:
        standardized = np.clip(standardized, clip_range[0], clip_range[1])

    # Restore original shape
    if len(original_shape) == 1:
        standardized = standardized.flatten()
    else:
        standardized = standardized.reshape(original_shape)

    return standardized, mean, std


def _standardize_array_with_stats(
    array: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    epsilon: float = 1e-8,
    clip_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize array using pre-computed mean and std statistics.

    Args:
        array: Input array to standardize
        mean: Pre-computed mean values
        std: Pre-computed standard deviation values
        epsilon: Small constant to prevent division by zero
        clip_range: Optional tuple of (min, max) to clip the standardized values

    Returns:
        Tuple of (standardized_array, mean, std)
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array, dtype=np.float32)

    # Save original shape
    original_shape = array.shape
    if array.ndim == 1 and mean.ndim > 0:
        array = array.reshape(-1, 1)

    # Avoid division by zero
    safe_std = np.maximum(std, epsilon)

    # Standardize data
    standardized = (array - mean) / safe_std

    # Clip values if requested
    if clip_range is not None:
        standardized = np.clip(standardized, clip_range[0], clip_range[1])

    # Restore original shape
    if len(original_shape) == 1:
        standardized = standardized.flatten()
    else:
        standardized = standardized.reshape(original_shape)

    return standardized, mean, safe_std


def scale_rewards(
    rewards: np.ndarray,
    scaling_method: str = "linear",
    scale: float = 1.0,
    shift: float = 0.0,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> np.ndarray:
    """
    Scale rewards using various methods.

    Args:
        rewards: Input rewards to scale
        scaling_method: Method to use: 'linear', 'clip', 'tanh', 'sign', 'scale', or 'standardize'
        scale: Scaling factor to multiply rewards by (for 'linear' and 'scale')
        shift: Value to add to rewards after scaling (for 'linear')
        clip_min: Minimum value for clipping (for 'clip' method)
        clip_max: Maximum value for clipping (for 'clip' method)

    Returns:
        Scaled rewards
    """
    if not isinstance(rewards, np.ndarray):
        rewards = np.array(rewards, dtype=np.float32)

    # Copy rewards to avoid modifying the original array
    scaled = rewards.copy()

    if scaling_method == "linear":
        # Apply scaling and shifting
        scaled = scaled * scale + shift

    elif scaling_method == "scale":
        # Simple scaling without shift
        scaled = scaled * scale

    elif scaling_method == "clip":
        # Clip rewards to specified range
        if clip_min is not None and clip_max is not None:
            scaled = np.clip(scaled, clip_min, clip_max)
        elif clip_min is not None:
            scaled = np.maximum(scaled, clip_min)
        elif clip_max is not None:
            scaled = np.minimum(scaled, clip_max)

    elif scaling_method == "tanh":
        # Scale rewards using hyperbolic tangent
        scaled = np.tanh(scaled * scale)
        if shift != 0:
            scaled += shift

    elif scaling_method == "sign":
        # Replace rewards with their signs (-1, 0, 1)
        scaled = np.sign(scaled)
        if scale != 1.0:
            scaled *= scale
        if shift != 0:
            scaled += shift

    elif scaling_method == "standardize":
        # Standardize rewards to have zero mean and unit variance
        if len(scaled) > 1:
            mean = np.mean(scaled)
            std = np.std(scaled)
            if std < 1e-8:
                std = 1.0  # Avoid division by zero
            scaled = (scaled - mean) / std

            # Apply optional scale and shift after standardization
            if scale != 1.0:
                scaled *= scale
            if shift != 0:
                scaled += shift

        # Apply clipping if specified
        if clip_min is not None and clip_max is not None:
            scaled = np.clip(scaled, clip_min, clip_max)

    else:
        raise ValueError(f"Unknown scaling method: {scaling_method}")

    return scaled
