"""
Performance benchmarking utilities for storage backends.

This module provides tools for benchmarking different storage backends and
configurations to help users choose the optimal setup for their needs.
"""

import gc
import logging
import os
import shutil
import tempfile
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

# Fix the import path to use the updated storage module
from satrlgym.storage import create_storage

logger = logging.getLogger(__name__)


class StorageBenchmark:
    """
    Benchmark class for measuring storage performance.
    """

    def __init__(self, temp_dir=None):
        """
        Initialize the benchmark utilities.

        Args:
            temp_dir: Optional directory to use for temporary files
                      (if None, a new temporary directory will be created)
        """
        if temp_dir:
            self.temp_dir = temp_dir
            os.makedirs(self.temp_dir, exist_ok=True)
        else:
            self.temp_dir = tempfile.mkdtemp()

    def __del__(self):
        """Clean up temporary files on destruction."""
        try:
            shutil.rmtree(self.temp_dir)
        except BaseException:
            pass

    def _create_test_data(
        self, num_transitions: int, obs_dim: int = 10, use_float64: bool = False
    ) -> dict[str, np.ndarray]:
        """
        Create test data for benchmarking.

        Args:
            num_transitions: Number of transitions to generate
            obs_dim: Dimension of observation space
            use_float64: Whether to use float64 (True) or float32 (False)

        Returns:
            Dictionary containing test data
        """
        float_type = np.float64 if use_float64 else np.float32

        # Create random data with realistic dimensions
        data = {
            "observations": np.random.normal(0, 1, (num_transitions, obs_dim)).astype(
                float_type
            ),
            "actions": np.random.randint(0, 5, (num_transitions,)).astype(np.int32),
            "rewards": np.random.normal(0, 1, (num_transitions,)).astype(float_type),
            "next_observations": np.random.normal(
                0, 1, (num_transitions, obs_dim)
            ).astype(float_type),
            "dones": np.zeros((num_transitions,), dtype=bool),
        }

        return data

    def measure_write_throughput(
        self,
        storage_type: str,
        num_transitions: int = 100000,
        obs_dim: int = 10,
        batch_size: int = 1000,
        compression: str | None = None,
    ) -> dict[str, Any]:
        """
        Measure write throughput for a storage backend.

        Args:
            storage_type: Type of storage backend to test (e.g., 'npz', 'hdf5')
            num_transitions: Total number of transitions to write
            obs_dim: Dimension of observation space
            batch_size: Number of transitions per write batch
            compression: Compression type to use (backend-specific)

        Returns:
            Dictionary with benchmark results
        """
        # Create test data
        full_data = self._create_test_data(num_transitions, obs_dim)

        # Prepare output file
        file_path = os.path.join(
            self.temp_dir, f"write_benchmark_{storage_type}.{storage_type}"
        )
        if os.path.exists(file_path):
            os.remove(file_path)

        # Initialize results
        results = {
            "storage_type": storage_type,
            "num_transitions": num_transitions,
            "obs_dim": obs_dim,
            "batch_size": batch_size,
            "compression": compression,
            "write_time_seconds": 0,
            "throughput_transitions_per_second": 0,
            "file_size_bytes": 0,
            "storage_efficiency_bytes_per_transition": 0,
        }

        # Create kwargs for storage backend
        kwargs = {}
        if compression is not None:
            kwargs["compression"] = compression

        try:
            # Create storage instance
            storage = create_storage(storage_type, file_path, **kwargs)

            # Measure write time
            start_time = time.time()

            # Write data in batches
            for i in range(0, num_transitions, batch_size):
                end_idx = min(i + batch_size, num_transitions)
                batch = {k: v[i:end_idx] for k, v in full_data.items()}

                if i == 0:
                    storage.write_batch(batch)
                else:
                    storage.append_batch(batch)

            write_time = time.time() - start_time

            # Calculate metrics
            results["write_time_seconds"] = write_time
            results["throughput_transitions_per_second"] = num_transitions / write_time

            # Get file size
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                results["file_size_bytes"] = file_size
                results["storage_efficiency_bytes_per_transition"] = (
                    file_size / num_transitions
                )

        except Exception as e:
            logger.error(f"Error in write throughput measurement: {str(e)}")
            results["error"] = str(e)

        return results

    def measure_read_throughput(
        self,
        storage_type: str,
        num_transitions: int = 100000,
        obs_dim: int = 10,
        batch_size: int = 1000,
        compression: str | None = None,
        random_access: bool = False,
    ) -> dict[str, Any]:
        """
        Measure read throughput for a storage backend.

        Args:
            storage_type: Type of storage backend to test (e.g., 'npz', 'hdf5')
            num_transitions: Total number of transitions to read
            obs_dim: Dimension of observation space
            batch_size: Number of transitions per read batch
            compression: Compression type to use (backend-specific)
            random_access: Whether to test random access (True) or sequential (False)

        Returns:
            Dictionary with benchmark results
        """
        # Prepare file path
        file_path = os.path.join(
            self.temp_dir, f"read_benchmark_{storage_type}.{storage_type}"
        )

        # Initialize results
        results = {
            "storage_type": storage_type,
            "num_transitions": num_transitions,
            "obs_dim": obs_dim,
            "batch_size": batch_size,
            "compression": compression,
            "random_access": random_access,
            "read_time_seconds": 0,
            "throughput_transitions_per_second": 0,
        }

        try:
            # First create test data and write it to file
            test_data = self._create_test_data(num_transitions, obs_dim)

            # Create kwargs for storage backend
            kwargs = {}
            if compression is not None:
                kwargs["compression"] = compression

            # Write test data
            storage = create_storage(storage_type, file_path, **kwargs)
            storage.write_batch(test_data)

            # Measure read time
            start_time = time.time()

            if random_access:
                # Generate random indices
                indices = np.random.randint(
                    0,
                    num_transitions,
                    size=(num_transitions // batch_size) * batch_size,
                )

                # Read random batches
                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i : i + batch_size]
                    # Not all storage backends support random access, so we read in full
                    # and then extract the indices we want
                    full_batch = storage.read_batch()
                    batch = {
                        k: v[batch_indices]
                        for k, v in full_batch.items()
                        if isinstance(v, np.ndarray)
                    }
            else:
                # Read sequential batches
                for i in range(0, num_transitions, batch_size):
                    offset = i
                    storage.read_batch(batch_size=batch_size, offset=offset)

            read_time = time.time() - start_time

            # Calculate metrics
            results["read_time_seconds"] = read_time
            results["throughput_transitions_per_second"] = num_transitions / read_time

        except Exception as e:
            logger.error(f"Error in read throughput measurement: {str(e)}")
            results["error"] = str(e)

        return results

    def profile_memory_usage(
        self,
        storage_type: str,
        num_transitions: int = 100000,
        obs_dim: int = 10,
        compression: str | None = None,
    ) -> dict[str, Any]:
        """
        Profile memory usage for a storage backend.

        Args:
            storage_type: Type of storage backend to test (e.g., 'npz', 'hdf5')
            num_transitions: Total number of transitions
            obs_dim: Dimension of observation space
            compression: Compression type to use (backend-specific)

        Returns:
            Dictionary with memory usage results
        """
        # Prepare file path
        file_path = os.path.join(
            self.temp_dir, f"memory_benchmark_{storage_type}.{storage_type}"
        )

        # Initialize results
        results = {
            "storage_type": storage_type,
            "num_transitions": num_transitions,
            "obs_dim": obs_dim,
            "compression": compression,
            "baseline_memory_mb": 0,
            "peak_memory_mb": 0,
            "memory_overhead_mb": 0,
        }

        try:
            # Force garbage collection before measuring
            gc.collect()

            # Get baseline memory usage
            baseline_memory = psutil.Process(os.getpid()).memory_info().rss / (
                1024 * 1024
            )  # MB
            results["baseline_memory_mb"] = baseline_memory

            # Create test data
            test_data = self._create_test_data(num_transitions, obs_dim)

            # Create kwargs for storage backend
            kwargs = {}
            if compression is not None:
                kwargs["compression"] = compression

            # Write test data
            storage = create_storage(storage_type, file_path, **kwargs)
            storage.write_batch(test_data)

            # Measure memory during read
            gc.collect()
            storage = create_storage(storage_type, file_path, **kwargs)

            # Read full dataset
            before_read = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            storage.read_batch()
            after_read = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

            # Calculate metrics
            results["peak_memory_mb"] = after_read
            results["memory_overhead_mb"] = after_read - baseline_memory
            results["memory_increase_mb"] = after_read - before_read

        except Exception as e:
            logger.error(f"Error in memory profiling: {str(e)}")
            results["error"] = str(e)

        return results

    def compare_storage_backends(
        self,
        backend_types: list[str],
        num_transitions: int = 100000,
        obs_dim: int = 10,
        batch_size: int = 1000,
        compression_options: dict[str, str] | None = None,
        output_format: str = "dataframe",
    ) -> pd.DataFrame | dict[str, list[dict[str, Any]]]:
        """
        Run comparative benchmarks across multiple storage backends.

        Args:
            backend_types: List of storage backend types to test
            num_transitions: Number of transitions to use
            obs_dim: Dimension of observation space
            batch_size: Batch size for read/write operations
            compression_options: Dict mapping backend types to compression options
            output_format: Output format ('dataframe' or 'dict')

        Returns:
            Benchmark results in the requested format
        """
        compression_options = compression_options or {}
        results = {
            "write_benchmarks": [],
            "read_sequential_benchmarks": [],
            "read_random_benchmarks": [],
            "memory_benchmarks": [],
        }

        for backend_type in backend_types:
            compression = compression_options.get(backend_type)

            # Measure write throughput
            write_result = self.measure_write_throughput(
                backend_type, num_transitions, obs_dim, batch_size, compression
            )
            results["write_benchmarks"].append(write_result)

            # Measure sequential read throughput
            read_seq_result = self.measure_read_throughput(
                backend_type, num_transitions, obs_dim, batch_size, compression, False
            )
            results["read_sequential_benchmarks"].append(read_seq_result)

            # Measure random read throughput
            read_rand_result = self.measure_read_throughput(
                backend_type, num_transitions, obs_dim, batch_size, compression, True
            )
            results["read_random_benchmarks"].append(read_rand_result)

            # Measure memory usage
            memory_result = self.profile_memory_usage(
                backend_type, num_transitions, obs_dim, compression
            )
            results["memory_benchmarks"].append(memory_result)

        if output_format == "dataframe":
            dfs = {}
            for key, benchmark_list in results.items():
                dfs[key] = pd.DataFrame(benchmark_list)

            # Combine all dataframes
            combined_df = pd.concat(
                [
                    dfs["write_benchmarks"].assign(benchmark_type="write"),
                    dfs["read_sequential_benchmarks"].assign(
                        benchmark_type="read_sequential"
                    ),
                    dfs["read_random_benchmarks"].assign(benchmark_type="read_random"),
                    dfs["memory_benchmarks"].assign(benchmark_type="memory"),
                ]
            )

            return combined_df
        else:
            return results

    def plot_comparison_results(
        self,
        results: pd.DataFrame | dict,
        output_file: str | None = None,
        figsize: tuple[int, int] = (15, 12),
    ) -> None:
        """
        Plot comparison results.

        Args:
            results: Benchmark results from compare_storage_backends
            output_file: Path to save the plot (None for display only)
            figsize: Figure size (width, height) in inches
        """
        if isinstance(results, dict):
            # Convert dict to DataFrame
            dfs = {}
            for key, benchmark_list in results.items():
                dfs[key] = pd.DataFrame(benchmark_list)

            # Combine all dataframes
            results = pd.concat(
                [
                    dfs["write_benchmarks"].assign(benchmark_type="write"),
                    dfs["read_sequential_benchmarks"].assign(
                        benchmark_type="read_sequential"
                    ),
                    dfs["read_random_benchmarks"].assign(benchmark_type="read_random"),
                    dfs["memory_benchmarks"].assign(benchmark_type="memory"),
                ]
            )

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot throughput comparisons
        write_data = results[results.benchmark_type == "write"]
        if not write_data.empty:
            ax1 = axes[0, 0]
            write_data.plot.bar(
                x="storage_type",
                y="throughput_transitions_per_second",
                ax=ax1,
                legend=False,
                color="blue",
            )
            ax1.set_title("Write Throughput")
            ax1.set_ylabel("Transitions/second")
            ax1.set_xlabel("")
            ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # Plot read throughput comparisons
        read_data = results[
            (results.benchmark_type == "read_sequential")
            | (results.benchmark_type == "read_random")
        ]
        if not read_data.empty:
            ax2 = axes[0, 1]
            read_data.pivot(
                index="storage_type",
                columns="benchmark_type",
                values="throughput_transitions_per_second",
            ).plot.bar(ax=ax2)
            ax2.set_title("Read Throughput")
            ax2.set_ylabel("Transitions/second")
            ax2.set_xlabel("")
            ax2.grid(axis="y", linestyle="--", alpha=0.7)
            ax2.legend(["Sequential", "Random"])

        # Plot storage efficiency
        if not write_data.empty:
            ax3 = axes[1, 0]
            write_data.plot.bar(
                x="storage_type",
                y="storage_efficiency_bytes_per_transition",
                ax=ax3,
                legend=False,
                color="green",
            )
            ax3.set_title("Storage Efficiency")
            ax3.set_ylabel("Bytes/transition")
            ax3.set_xlabel("")
            ax3.grid(axis="y", linestyle="--", alpha=0.7)

        # Plot memory usage
        memory_data = results[results.benchmark_type == "memory"]
        if not memory_data.empty:
            ax4 = axes[1, 1]
            memory_data.plot.bar(
                x="storage_type",
                y="memory_overhead_mb",
                ax=ax4,
                legend=False,
                color="red",
            )
            ax4.set_title("Memory Overhead")
            ax4.set_ylabel("MB")
            ax4.set_xlabel("")
            ax4.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close(fig)


def run_benchmark(
    backend_types: list[str] = None, sizes: list[int] = None, output_dir: str = None
) -> dict[str, Any]:
    """
    Run comprehensive benchmark suite and generate reports.

    Args:
        backend_types: Storage backends to test (None for all available)
        sizes: Dataset sizes to test in transitions (None for default sizes)
        output_dir: Directory to save results (None for temp directory)

    Returns:
        Dictionary with paths to generated reports and plots
    """
    # Set defaults
    backend_types = backend_types or ["npz", "hdf5"]
    sizes = sizes or [10000, 100000]
    output_dir = output_dir or tempfile.mkdtemp()

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup benchmark
    benchmark = StorageBenchmark()

    # Setup compression options
    compression_options = {
        "npz": None,  # NPZ uses compression by default
        "hdf5": "gzip",
        "parquet": "snappy",
    }

    results = {}

    # Run benchmarks for each size
    for size in sizes:
        size_results = benchmark.compare_storage_backends(
            backend_types, num_transitions=size, compression_options=compression_options
        )

        # Generate plots
        plot_path = os.path.join(
            output_dir, f"benchmark_comparison_{size}_transitions.png"
        )
        benchmark.plot_comparison_results(size_results, output_file=plot_path)

        # Save raw results
        csv_path = os.path.join(output_dir, f"benchmark_results_{size}_transitions.csv")
        if isinstance(size_results, pd.DataFrame):
            size_results.to_csv(csv_path, index=False)

        results[f"{size}_transitions"] = {
            "dataframe": size_results,
            "plot_path": plot_path,
            "csv_path": csv_path,
        }

    # Generate summary report
    summary_path = os.path.join(output_dir, "benchmark_summary.md")

    with open(summary_path, "w") as f:
        f.write("# Storage Backend Performance Benchmark Summary\n\n")

        for size in sizes:
            f.write(f"## Results for {size} transitions\n\n")

            # Extract key metrics
            df = results[f"{size}_transitions"]["dataframe"]

            # Write throughput table
            f.write("### Write Throughput (transitions/second)\n\n")
            write_data = df[df.benchmark_type == "write"]
            f.write(
                write_data[
                    ["storage_type", "throughput_transitions_per_second"]
                ].to_markdown(index=False)
            )
            f.write("\n\n")

            # Read throughput table
            f.write("### Read Throughput (transitions/second)\n\n")
            read_data = df[
                (df.benchmark_type == "read_sequential")
                | (df.benchmark_type == "read_random")
            ]
            pivot_read = read_data.pivot(
                index="storage_type",
                columns="benchmark_type",
                values="throughput_transitions_per_second",
            ).reset_index()
            f.write(pivot_read.to_markdown(index=False))
            f.write("\n\n")

            # Storage efficiency table
            f.write("### Storage Efficiency (bytes/transition)\n\n")
            f.write(
                write_data[
                    ["storage_type", "storage_efficiency_bytes_per_transition"]
                ].to_markdown(index=False)
            )
            f.write("\n\n")

            # Memory usage table
            f.write("### Memory Overhead (MB)\n\n")
            memory_data = df[df.benchmark_type == "memory"]
            f.write(
                memory_data[["storage_type", "memory_overhead_mb"]].to_markdown(
                    index=False
                )
            )
            f.write("\n\n")

            # Add plot
            f.write(
                f"![Benchmark Plot]({os.path.basename(results[f'{size}_transitions']['plot_path'])})\n\n"
            )

    results["summary_path"] = summary_path

    return results
