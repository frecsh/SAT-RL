"""
Dataset Quality Assurance utilities.

This module provides tools for validating dataset integrity, generating statistics,
and detecting anomalies in experience replay datasets.
"""

import logging
import os
from typing import Any

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validates experience replay datasets for integrity and consistency."""

    def __init__(self, required_fields: set[str] | None = None):
        """
        Initialize the validator with configuration.

        Args:
            required_fields: Set of field names that must be present in all trsitions.
                             Defaults to standard RL fields if None.
        """
        self.required_fields = required_fields or {
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "dones",
        }

    def validate_dataset(
        self, data_path: str, sample_size: int | None = None
    ) -> dict[str, Any]:
        """
        Validate dataset integrity.

        Args:
            data_path: Path to the dataset file or directory
            sample_size: Number of transitions to sample for validation (None for all)

        Returns:
            Dict containing validation results
        """
        from experience.storage import create_storage

        results = {"is_valid": True, "errors": [], "warnings": [], "metadata": {}}

        # Determine file type and create appropriate storage reader
        file_ext = os.path.splitext(data_path)[1].lower()

        try:
            storage = create_storage(file_ext.replace(".", ""), data_path)

            # Check if file exists and is readable
            if not os.path.exists(data_path):
                results["is_valid"] = False
                results["errors"].append(f"File not found: {data_path}")
                return results

            # Read batch (all or sample)
            try:
                batch = storage.read_batch()
                results["metadata"]["total_transitions"] = len(
                    batch.get("observations", [])
                )
            except Exception as e:
                results["is_valid"] = False
                results["errors"].append(f"Failed to read data: {str(e)}")
                return results

            # Check required fields
            for field in self.required_fields:
                if field not in batch:
                    results["is_valid"] = False
                    results["errors"].append(f"Required field missing: {field}")

            # Check consistent lengths across fields
            lengths = {
                field: len(values)
                for field, values in batch.items()
                if isinstance(values, (list, np.ndarray))
            }

            if len(set(lengths.values())) > 1:
                results["is_valid"] = False
                results["errors"].append(f"Inconsistent field lengths: {lengths}")

            # Check for NaN values
            for field, values in batch.items():
                if isinstance(values, np.ndarray) and values.dtype.kind == "f":
                    nan_count = np.isnan(values).sum()
                    if nan_count > 0:
                        results["warnings"].append(
                            f"Field '{field}' contains {nan_count} NaN values"
                        )

            # If metadata is available, include it
            if hasattr(storage, "get_metadata"):
                try:
                    results["metadata"]["file_metadata"] = storage.get_metadata()
                except BaseException:
                    results["warnings"].append("Failed to read metadata")

            return results

        except Exception as e:
            results["is_valid"] = False
            results["errors"].append(f"Validation failed: {str(e)}")
            return results


class StatisticsGenerator:
    """Generates statistics for experience replay datasets."""

    def __init__(self, numeric_fields: list[str] | None = None):
        """
        Initialize the statistics generator.

        Args:
            numeric_fields: List of field names to calculate statistics for.
                           If None, will attempt to calculate for all numeric fields.
        """
        self.numeric_fields = numeric_fields

    def generate_statistics(
        self, data_path: str, sample_size: int | None = None
    ) -> dict[str, dict[str, float]]:
        """
        Calculate statistics for dataset fields.

        Args:
            data_path: Path to the dataset file
            sample_size: Number of transitions to sample for stats (None for all)

        Returns:
            Dict mapping field names to their statistics
        """
        from experience.storage import create_storage

        # Determine file type and create appropriate storage reader
        file_ext = os.path.splitext(data_path)[1].lower()
        storage = create_storage(file_ext.replace(".", ""), data_path)

        # Read batch (all or sample)
        batch = storage.read_batch()

        statistics = {}

        # Determine fields to process
        fields_to_process = self.numeric_fields if self.numeric_fields else batch.keys()

        for field in fields_to_process:
            if field not in batch:
                continue

            values = batch[field]

            # Skip non-numeric fields
            if not isinstance(values, np.ndarray) or values.dtype.kind not in "iuf":
                continue

            # Calculate statistics
            try:
                field_stats = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                    "count": len(values),
                    "nan_count": (
                        int(np.isnan(values).sum()) if values.dtype.kind == "f" else 0
                    ),
                }
                statistics[field] = field_stats
            except Exception as e:
                logger.warning(
                    f"Failed to calculate statistics for field {field}: {str(e)}"
                )

        return statistics


class AnomalyDetector:
    """Detects anomalies in experience replay datasets."""

    def __init__(
        self,
        threshold_std_multiplier: float = 5.0,
        min_reward: float | None = None,
        max_reward: float | None = None,
    ):
        """
        Initialize the anomaly detector.

        Args:
            threshold_std_multiplier: Number of standard deviations from mean to flag as anomaly
            min_reward: Minimum valid reward value (None for no limit)
            max_reward: Maximum valid reward value (None for no limit)
        """
        self.threshold_std_multiplier = threshold_std_multiplier
        self.min_reward = min_reward
        self.max_reward = max_reward

    def detect_anomalies(
        self, data_path: str, fields_to_check: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Detect anomalies in the dataset.

        Args:
            data_path: Path to the dataset file
            fields_to_check: List of fields to check for anomalies (None for all numeric)

        Returns:
            Dict with anomaly detection results
        """
        from experience.storage import create_storage

        results = {
            "anomalies_detected": False,
            "anomaly_count": 0,
            "field_anomalies": {},
            "corrupt_transitions": [],
        }

        # Determine file type and create appropriate storage reader
        file_ext = os.path.splitext(data_path)[1].lower()
        storage = create_storage(file_ext.replace(".", ""), data_path)

        # Read batch
        batch = storage.read_batch()

        # Generate statistics
        stats_gen = StatisticsGenerator()
        statistics = stats_gen.generate_statistics(data_path)

        fields_to_process = fields_to_check if fields_to_check else statistics.keys()

        for field in fields_to_process:
            if field not in batch:
                continue

            values = batch[field]

            # Skip non-numeric fields
            if not isinstance(values, np.ndarray) or values.dtype.kind not in "iuf":
                continue

            field_stats = statistics.get(field, {})

            # Handle different array shapes
            is_multi_dimensional = len(values.shape) > 1

            # Set thresholds based on statistics
            if field_stats and "mean" in field_stats and "std" in field_stats:
                lower_threshold = field_stats["mean"] - (
                    field_stats["std"] * self.threshold_std_multiplier
                )
                upper_threshold = field_stats["mean"] + (
                    field_stats["std"] * self.threshold_std_multiplier
                )
            else:
                # Default thresholds if no statistics available
                lower_threshold = -1e9
                upper_threshold = 1e9

            # Override with explicit thresholds if provided for rewards
            if field == "rewards":
                if self.min_reward is not None:
                    lower_threshold = self.min_reward
                if self.max_reward is not None:
                    upper_threshold = self.max_reward

            # First detect NaN values
            anomaly_indices = []

            if values.dtype.kind == "f":
                if is_multi_dimensional:
                    nan_mask = np.isnan(values).any(axis=1)
                else:
                    nan_mask = np.isnan(values)

                nan_indices = np.where(nan_mask)[0]
                if len(nan_indices) > 0:
                    anomaly_indices.extend(nan_indices.tolist())

            # For extreme outlier detection, use a more robust approach for rewards
            if field == "rewards":
                # Check for extreme values directly
                if is_multi_dimensional:
                    # For multi-dimensional arrays
                    for i in range(len(values)):
                        row = values[i]
                        if np.any(
                            np.abs(row) > 1000
                        ):  # Explicit check for extreme values
                            anomaly_indices.append(i)
                else:
                    # For 1D arrays
                    extreme_mask = np.abs(values) > 1000
                    extreme_indices = np.where(extreme_mask)[0]
                    anomaly_indices.extend(extreme_indices.tolist())

            # Also check using standard thresholds
            if is_multi_dimensional:
                for i in range(len(values)):
                    if i not in anomaly_indices:  # Skip if already flagged
                        row = values[i]
                        if not np.isnan(row).any():  # Skip rows with NaN
                            if np.any(
                                (row < lower_threshold) | (row > upper_threshold)
                            ):
                                anomaly_indices.append(i)
            else:
                # For 1D arrays
                valid_mask = (
                    ~np.isnan(values)
                    if values.dtype.kind == "f"
                    else np.ones_like(values, dtype=bool)
                )
                for i in range(len(values)):
                    if (
                        i not in anomaly_indices and valid_mask[i]
                    ):  # Skip if already flagged or NaN
                        if values[i] < lower_threshold or values[i] > upper_threshold:
                            anomaly_indices.append(i)

            # Record anomalies
            if anomaly_indices:
                unique_indices = sorted(list(set(anomaly_indices)))
                results["field_anomalies"][field] = {
                    "count": len(unique_indices),
                    "indices": unique_indices[:100],  # Limit to first 100
                    "threshold_low": float(lower_threshold),
                    "threshold_high": float(upper_threshold),
                }
                results["anomalies_detected"] = True
                results["anomaly_count"] += len(unique_indices)
                results["corrupt_transitions"].extend(unique_indices)

        # Remove duplicates from corrupt transitions
        results["corrupt_transitions"] = list(set(results["corrupt_transitions"]))
        results["corrupt_transitions"].sort()
        results["corrupt_transitions"] = results["corrupt_transitions"][
            :1000
        ]  # Limit output size

        return results


def validate_and_report(
    data_path: str, fields_to_check: list[str] | None = None
) -> dict[str, Any]:
    """
    Comprehensive validation and reporting function that combines all QA tools.

    Args:
        data_path: Path to the dataset file or directory
        fields_to_check: Optional list of fields to check for anomalies

    Returns:
        Dict with combined QA results
    """
    report = {
        "path": data_path,
        "filename": os.path.basename(data_path),
        "validation": None,
        "statistics": None,
        "anomalies": None,
        "summary": {
            "is_valid": False,
            "transition_count": 0,
            "anomaly_count": 0,
            "status": "unknown",
        },
    }

    # Run validation
    validator = DatasetValidator()
    validation_results = validator.validate_dataset(data_path)
    report["validation"] = validation_results

    # If valid, generate statistics and check for anomalies
    if validation_results["is_valid"]:
        # Generate statistics
        stats_gen = StatisticsGenerator()
        statistics = stats_gen.generate_statistics(data_path)
        report["statistics"] = statistics

        # Detect anomalies
        anomaly_detector = AnomalyDetector()
        anomalies = anomaly_detector.detect_anomalies(
            data_path, fields_to_check=fields_to_check
        )
        report["anomalies"] = anomalies

        # Update summary
        report["summary"]["is_valid"] = True
        report["summary"]["transition_count"] = validation_results["metadata"].get(
            "total_transitions", 0
        )
        report["summary"]["anomaly_count"] = anomalies["anomaly_count"]

        if anomalies["anomaly_count"] > 0:
            anomaly_pct = (
                anomalies["anomaly_count"] / report["summary"]["transition_count"]
            ) * 100
            if anomaly_pct > 5:
                report["summary"]["status"] = "poor"
            elif anomaly_pct > 1:
                report["summary"]["status"] = "warning"
            else:
                report["summary"]["status"] = "good"
        else:
            report["summary"]["status"] = "excellent"
    else:
        report["summary"]["status"] = "invalid"

    return report
