"""
Representation Benchmarking System for SAT-RL.

This module provides tools to evaluate and compare different representation methods
for SAT problems, including metrics for state distinguishability, computational
efficiency, and visualization utilities.
"""

import time
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

# Try to import optional dependencies
try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from satrlgym.representation.basic_representation import ObservationEncoder

try:
    pass

    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False


class RepresentationBenchmark:
    """
    Benchmark system for evaluating and comparing different SAT problem representations.
    """

    def __init__(self, problem_generator: Callable, seed: int = 42):
        """
        Initialize the representation benchmark system.

        Args:
            problem_generator: Function that generates SAT problems for benchmarking
            seed: Random seed for reproducibility
        """
        self.problem_generator = problem_generator
        self.seed = seed
        self.encoders = {}
        self.results = {}
        np.random.seed(seed)
        torch.manual_seed(seed)

    def register_encoder(self, name: str, encoder: ObservationEncoder):
        """
        Register an observation encoder for benchmarking.

        Args:
            name: Name to identify the encoder
            encoder: Encoder object implementing the ObservationEncoder interface
        """
        self.encoders[name] = encoder

    def run_benchmark(
        self,
        num_problems: int = 10,
        num_states_per_problem: int = 100,
        verbose: bool = True,
    ) -> dict:
        """
        Run benchmark tests on all registered encoders.

        Args:
            num_problems: Number of different SAT problems to test
            num_states_per_problem: Number of states to generate per problem
            verbose: Whether to show progress information

        Returns:
            Dictionary containing benchmark results
        """
        results = {
            "state_distinguishability": {},
            "computational_efficiency": {},
            "memory_usage": {},
            "state_samples": {},
            "encoding_samples": {},
            "metadata": {
                "num_problems": num_problems,
                "num_states_per_problem": num_states_per_problem,
                "seed": self.seed,
            },
        }

        if verbose:
            print(f"Running benchmarks on {len(self.encoders)} encoders...")

        # Generate problems for benchmarking
        problems = []
        for i in range(num_problems):
            problem = self.problem_generator(seed=self.seed + i)
            problems.append(problem)

        # Store a sample of states for visualization
        sample_problem = problems[0]
        sample_states = self._generate_states(sample_problem, num_states_per_problem)
        results["state_samples"] = sample_states[:10]  # Store just a few samples

        # Run benchmarks for each encoder
        for name, encoder in self.encoders.items():
            if verbose:
                print(f"Benchmarking {name}...")

            encoder_results = {
                "state_distinguishability": [],
                "computational_time": [],
                "encoding_samples": [],
            }

            # Run on all problems
            for problem_idx, problem in enumerate(problems):
                # Generate states for this problem
                states = self._generate_states(problem, num_states_per_problem)

                # Measure encoding time
                start_time = time.time()
                encodings = [
                    encoder.encode(state) for state in tqdm(states, disable=not verbose)
                ]
                end_time = time.time()
                computation_time = (end_time - start_time) / num_states_per_problem

                # Compute state distinguishability metric
                distinguishability = self._compute_state_distinguishability(encodings)

                # Store results for this problem
                encoder_results["state_distinguishability"].append(distinguishability)
                encoder_results["computational_time"].append(computation_time)

                # Store sample encodings for the first problem
                if problem_idx == 0:
                    encoder_results["encoding_samples"] = encodings[:10]

            # Aggregate results across problems
            results["state_distinguishability"][name] = np.mean(
                encoder_results["state_distinguishability"]
            )
            results["computational_efficiency"][name] = np.mean(
                encoder_results["computational_time"]
            )
            results["encoding_samples"][name] = encoder_results["encoding_samples"]

        self.results = results
        return results

    def _generate_states(self, problem: dict, num_states: int) -> list[dict]:
        """
        Generate different states for a given problem by randomly assigning variables.

        Args:
            problem: SAT problem (dictionary with 'clauses' and metadata)
            num_states: Number of states to generate

        Returns:
            List of state dictionaries with variable assignments
        """
        num_variables = problem.get("num_variables", len(problem.get("variables", [])))
        clauses = problem["clauses"]

        states = []
        for _ in range(num_states):
            # Randomly assign variables with different percentages of assignment
            assignment_percentage = np.random.uniform(0.3, 1.0)
            num_assigned = int(num_variables * assignment_percentage)

            var_indices = np.random.choice(num_variables, num_assigned, replace=False)
            var_assignments = {}

            for var_idx in var_indices:
                # 1-indexed variables
                var_assignments[var_idx + 1] = bool(np.random.randint(0, 2))

            # Calculate clause satisfaction based on assignments
            clause_satisfaction = []
            for clause in clauses:
                satisfied = False
                for lit in clause:
                    var_idx = abs(lit)
                    if var_idx in var_assignments:
                        is_negated = lit < 0
                        value = var_assignments[var_idx]
                        if (not is_negated and value) or (is_negated and not value):
                            satisfied = True
                            break
                clause_satisfaction.append(satisfied)

            state = {
                "variable_assignments": var_assignments,
                "clause_satisfaction": clause_satisfaction,
                "clauses": clauses,
            }
            states.append(state)

        return states

    def _compute_state_distinguishability(self, encodings: list) -> float:
        """
        Compute a metric for how well the representation distinguishes between different states.

        Args:
            encodings: List of encoded states

        Returns:
            Distinguishability score (higher is better)
        """
        # Handle different encoder output formats
        if isinstance(encodings[0], dict):
            # For structured encodings (like GNN outputs), use the graph_embedding
            if "graph_embedding" in encodings[0]:
                # Extract graph embeddings to numpy array
                vectors = np.array([enc["graph_embedding"] for enc in encodings])
            elif "variables" in encodings[0]:
                # Flatten variable encodings
                vectors = np.array([enc["variables"].flatten() for enc in encodings])
            else:
                # Fallback: concatenate all arrays in the dict
                vectors = np.array(
                    [
                        np.concatenate([v.flatten() for v in enc.values()])
                        for enc in encodings
                    ]
                )
        elif isinstance(encodings[0], (np.ndarray, torch.Tensor)):
            # For array-like encodings, convert to numpy and flatten
            vectors = np.array(
                [
                    (
                        enc.flatten()
                        if isinstance(enc, np.ndarray)
                        else enc.cpu().numpy().flatten()
                    )
                    for enc in encodings
                ]
            )
        else:
            raise ValueError(f"Unsupported encoding type: {type(encodings[0])}")

        # Compute pairwise distances
        distances = pairwise_distances(vectors)

        # Compute metrics based on the distance matrix
        # 1. Average distance between different states (higher is better)
        # 2. Variance of distances (higher means more distinguishable ranges)
        avg_distance = np.mean(distances[np.triu_indices(len(distances), k=1)])
        distance_variance = np.var(distances[np.triu_indices(len(distances), k=1)])

        # Combined distinguishability score (normalized to [0, 1] range)
        # This is a heuristic that rewards both high average distance and high variance
        distinguishability = (
            avg_distance * np.sqrt(distance_variance)
        ) / vectors.shape[1]

        return distinguishability

    def generate_report(self, output_path: str | None = None) -> dict:
        """
        Generate a comprehensive report of benchmarking results.

        Args:
            output_path: Optional path to save the report HTML/PDF

        Returns:
            Dictionary with report data
        """
        if not self.results:
            raise ValueError("No benchmark results available. Run benchmark first.")

        report = {
            "state_distinguishability": self.results["state_distinguishability"],
            "computational_efficiency": self.results["computational_efficiency"],
            "visualizations": {},
        }

        # Create visualizations
        self._create_report_visualizations(report)

        # Save report if path specified
        if output_path:
            self._save_report(report, output_path)

        return report

    def _create_report_visualizations(self, report: dict):
        """
        Create visualizations for the benchmark report.

        Args:
            report: Report dictionary to add visualizations to
        """
        # Bar chart for state distinguishability
        fig, ax = plt.subplots(figsize=(10, 6))
        encoders = list(report["state_distinguishability"].keys())
        values = list(report["state_distinguishability"].values())

        sns.barplot(x=encoders, y=values, ax=ax)
        ax.set_title("State Distinguishability by Encoder")
        ax.set_ylabel("Distinguishability Score")
        ax.set_xlabel("Encoder")
        report["visualizations"]["distinguishability_chart"] = fig

        # Bar chart for computational efficiency (lower is better)
        fig, ax = plt.subplots(figsize=(10, 6))
        encoders = list(report["computational_efficiency"].keys())
        values = list(report["computational_efficiency"].values())

        sns.barplot(x=encoders, y=values, ax=ax)
        ax.set_title("Computational Efficiency by Encoder (Lower is Better)")
        ax.set_ylabel("Average Encoding Time (seconds)")
        ax.set_xlabel("Encoder")
        report["visualizations"]["efficiency_chart"] = fig

        # Create dimensionality reduction visualizations for each encoder
        for name, sample_encodings in self.results["encoding_samples"].items():
            # Create dimensionality reduction plots if we have enough samples
            if len(sample_encodings) >= 10:
                try:
                    fig = self._create_embedding_visualization(sample_encodings, name)
                    report["visualizations"][f"{name}_embedding"] = fig
                except Exception as e:
                    print(f"Could not create embedding visualization for {name}: {e}")

    def _create_embedding_visualization(self, encodings: list, name: str) -> plt.Figure:
        """
        Create dimensionality reduction visualizations for encoded states.

        Args:
            encodings: List of encoded states
            name: Name of the encoder

        Returns:
            Matplotlib figure with visualizations
        """
        # Extract features for dimensionality reduction
        if isinstance(encodings[0], dict):
            if "graph_embedding" in encodings[0]:
                features = np.array([enc["graph_embedding"] for enc in encodings])
            elif "variables" in encodings[0]:
                features = np.array([enc["variables"].flatten() for enc in encodings])
            else:
                features = np.array(
                    [
                        np.concatenate([v.flatten() for v in enc.values()])
                        for enc in encodings
                    ]
                )
        else:
            features = np.array(
                [
                    (
                        enc.flatten()
                        if isinstance(enc, np.ndarray)
                        else enc.cpu().numpy().flatten()
                    )
                    for enc in encodings
                ]
            )

        # Determine how many visualizations to create based on available methods
        num_panels = 2  # PCA and t-SNE always available
        if UMAP_AVAILABLE:
            num_panels += 1

        # Create multi-panel figure
        fig, axs = plt.subplots(1, num_panels, figsize=(6 * num_panels, 6))
        fig.suptitle(f"Dimensionality Reduction Visualization for {name}", fontsize=16)

        panel_idx = 0

        # 1. PCA
        try:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(features)
            axs[panel_idx].scatter(
                pca_result[:, 0],
                pca_result[:, 1],
                c=range(len(features)),
                cmap="viridis",
            )
            axs[panel_idx].set_title("PCA")
            var_explained = pca.explained_variance_ratio_.sum() * 100
            axs[panel_idx].set_xlabel(f"Explained variance: {var_explained:.1f}%")
            panel_idx += 1
        except Exception as e:
            axs[panel_idx].text(
                0.5, 0.5, f"PCA failed: {str(e)}", ha="center", va="center"
            )
            panel_idx += 1

        # 2. t-SNE
        try:
            tsne = TSNE(n_components=2, random_state=self.seed)
            tsne_result = tsne.fit_transform(features)
            axs[panel_idx].scatter(
                tsne_result[:, 0],
                tsne_result[:, 1],
                c=range(len(features)),
                cmap="viridis",
            )
            axs[panel_idx].set_title("t-SNE")
            panel_idx += 1
        except Exception as e:
            axs[panel_idx].text(
                0.5, 0.5, f"t-SNE failed: {str(e)}", ha="center", va="center"
            )
            panel_idx += 1

        # 3. UMAP (if available)
        if UMAP_AVAILABLE and panel_idx < len(axs):
            try:
                umap_reducer = umap.UMAP(random_state=self.seed)
                umap_result = umap_reducer.fit_transform(features)
                axs[panel_idx].scatter(
                    umap_result[:, 0],
                    umap_result[:, 1],
                    c=range(len(features)),
                    cmap="viridis",
                )
                axs[panel_idx].set_title("UMAP")
            except Exception as e:
                axs[panel_idx].text(
                    0.5, 0.5, f"UMAP failed: {str(e)}", ha="center", va="center"
                )

        plt.tight_layout()
        return fig

    def _save_report(self, report: dict, output_path: str):
        """
        Save the benchmark report.

        Args:
            report: Report dictionary with data and visualizations
            output_path: Path to save the report
        """
        # Save visualizations as separate image files
        for name, fig in report["visualizations"].items():
            fig.savefig(f"{output_path}_{name}.png", bbox_inches="tight")

        # Create a summary text file
        with open(f"{output_path}_summary.txt", "w") as f:
            f.write("SAT-RL Representation Benchmark Summary\n")
            f.write("=====================================\n\n")

            f.write("State Distinguishability Results:\n")
            for encoder, score in report["state_distinguishability"].items():
                f.write(f"  {encoder}: {score:.4f}\n")

            f.write("\nComputational Efficiency Results:\n")
            for encoder, time in report["computational_efficiency"].items():
                f.write(f"  {encoder}: {time:.4f} seconds per state\n")


class AblationTester:
    """
    Performs controlled ablation tests on representation components.
    """

    def __init__(self, base_encoder: Any, problem_generator: Callable):
        """
        Initialize the ablation tester.

        Args:
            base_encoder: Base encoder to perform ablations on
            problem_generator: Function to generate test problems
        """
        self.base_encoder = base_encoder
        self.problem_generator = problem_generator
        self.ablation_results = {}

    def run_ablation(
        self,
        component_configs: list[dict],
        num_problems: int = 5,
        num_states_per_problem: int = 50,
        metrics: list[str] = ["distinguishability", "efficiency"],
    ) -> dict:
        """
        Run ablation tests with different component configurations.

        Args:
            component_configs: List of dictionaries with component configurations to test
                Each dict should include a 'name' and parameters for the encoder
            num_problems: Number of problems to test on
            num_states_per_problem: Number of states to generate per problem
            metrics: List of metrics to evaluate ('distinguishability', 'efficiency')

        Returns:
            Dictionary with ablation test results
        """
        results = {
            "distinguishability": {},
            "efficiency": {},
            "improvement": {},
            "component_importance": {},
        }

        # Create benchmark for testing
        benchmark = RepresentationBenchmark(self.problem_generator)

        # Test each configuration
        for config in component_configs:
            name = config.pop("name")
            is_base = config.pop("is_base", False)

            # Create encoder with this configuration
            # This assumes the encoder class can be instantiated with the config parameters
            try:
                if isinstance(self.base_encoder, type):
                    # If base_encoder is a class, instantiate it with the config
                    encoder = self.base_encoder(**config)
                else:
                    # Otherwise, assume it's a factory function
                    encoder = self.base_encoder(config)

                # Register the encoder with the benchmark
                benchmark.register_encoder(name, encoder)

                # Restore the popped values for future reference
                config["name"] = name
                if is_base:
                    config["is_base"] = is_base
            except Exception as e:
                print(f"Failed to create encoder for config {name}: {e}")
                # Restore the popped values
                config["name"] = name
                if is_base:
                    config["is_base"] = is_base
                continue

        # Run the benchmark if we have any valid encoders
        if benchmark.encoders:
            benchmark_results = benchmark.run_benchmark(
                num_problems=num_problems, num_states_per_problem=num_states_per_problem
            )

            # Extract relevant metrics for each configuration
            for name in benchmark.encoders.keys():
                if "distinguishability" in metrics:
                    results["distinguishability"][name] = benchmark_results[
                        "state_distinguishability"
                    ][name]

                if "efficiency" in metrics:
                    results["efficiency"][name] = benchmark_results[
                        "computational_efficiency"
                    ][name]

            # Calculate relative importance of each component
            base_config_name = next(
                (c["name"] for c in component_configs if c.get("is_base", False)),
                component_configs[0]["name"],
            )

            if base_config_name in results["distinguishability"]:
                base_distinguishability = results["distinguishability"].get(
                    base_config_name, 0
                )
                base_efficiency = results["efficiency"].get(
                    base_config_name, float("inf")
                )

                for name in benchmark.encoders.keys():
                    if name != base_config_name:
                        # Calculate relative improvement (positive is better)
                        dist_improvement = (
                            results["distinguishability"].get(name, 0)
                            - base_distinguishability
                        ) / max(base_distinguishability, 1e-10)
                        # For efficiency, negative is better (faster)
                        eff_improvement = (
                            base_efficiency
                            - results["efficiency"].get(name, float("inf"))
                        ) / max(base_efficiency, 1e-10)

                        results["improvement"][name] = {
                            "distinguishability": dist_improvement,
                            "efficiency": eff_improvement,
                        }

        # Store results
        self.ablation_results = results
        return results

    def visualize_ablation_results(self) -> dict[str, plt.Figure]:
        """
        Create visualizations of ablation test results.

        Returns:
            Dictionary of matplotlib figures
        """
        if not self.ablation_results:
            raise ValueError("No ablation results. Run ablation tests first.")

        figures = {}

        # 1. Distinguishability comparison
        if (
            "distinguishability" in self.ablation_results
            and self.ablation_results["distinguishability"]
        ):
            fig, ax = plt.subplots(figsize=(10, 6))
            configs = list(self.ablation_results["distinguishability"].keys())
            values = list(self.ablation_results["distinguishability"].values())

            sns.barplot(x=configs, y=values, ax=ax)
            ax.set_title("State Distinguishability by Configuration")
            ax.set_ylabel("Distinguishability Score")
            ax.set_xlabel("Configuration")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            figures["distinguishability"] = fig

        # 2. Efficiency comparison
        if (
            "efficiency" in self.ablation_results
            and self.ablation_results["efficiency"]
        ):
            fig, ax = plt.subplots(figsize=(10, 6))
            configs = list(self.ablation_results["efficiency"].keys())
            values = list(self.ablation_results["efficiency"].values())

            sns.barplot(x=configs, y=values, ax=ax)
            ax.set_title("Computational Efficiency by Configuration (Lower is Better)")
            ax.set_ylabel("Average Encoding Time (seconds)")
            ax.set_xlabel("Configuration")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            figures["efficiency"] = fig

        # 3. Improvement comparison
        if (
            "improvement" in self.ablation_results
            and self.ablation_results["improvement"]
        ):
            # Prepare data for plotting
            configs = list(self.ablation_results["improvement"].keys())
            dist_improvements = [
                self.ablation_results["improvement"][c]["distinguishability"]
                for c in configs
            ]
            eff_improvements = [
                self.ablation_results["improvement"][c]["efficiency"] for c in configs
            ]

            # Create plot with two metrics
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(configs))
            width = 0.35

            ax.bar(x - width / 2, dist_improvements, width, label="Distinguishability")
            ax.bar(x + width / 2, eff_improvements, width, label="Efficiency")

            ax.set_title("Relative Improvement vs Base Configuration")
            ax.set_ylabel("Relative Improvement")
            ax.set_xlabel("Configuration")
            ax.set_xticks(x)
            ax.set_xticklabels(configs, rotation=45, ha="right")
            ax.legend()
            ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)

            plt.tight_layout()
            figures["improvement"] = fig

        return figures


class ComputationalEfficiencyTracker:
    """
    Tracks computational efficiency of different representation methods.
    """

    def __init__(self, encoders: dict[str, ObservationEncoder]):
        """
        Initialize the computational efficiency tracker.

        Args:
            encoders: Dictionary mapping names to encoder instances
        """
        self.encoders = encoders
        self.results = {}

    def run_efficiency_test(
        self,
        problem_sizes: list[int],
        problem_generator: Callable,
        num_states_per_size: int = 20,
        warmup: bool = True,
    ) -> dict:
        """
        Run efficiency tests across different problem sizes.

        Args:
            problem_sizes: List of problem sizes (number of variables)
            problem_generator: Function to generate problems of given size
            num_states_per_size: Number of states to test for each size
            warmup: Whether to run warmup iterations before timing

        Returns:
            Dictionary with efficiency results
        """
        results = {
            "problem_sizes": problem_sizes,
            "encoding_time": {name: [] for name in self.encoders},
            "memory_usage": {name: [] for name in self.encoders},
        }

        for size in problem_sizes:
            print(f"Testing problem size: {size} variables")

            # Generate a problem of this size
            problem = problem_generator(num_vars=size)

            # Generate states
            states = []
            for _ in range(num_states_per_size):
                # Create partial assignments (30-70% of variables assigned)
                num_assigned = int(np.random.uniform(0.3, 0.7) * size)
                var_assignments = {
                    i + 1: bool(np.random.randint(0, 2))
                    for i in np.random.choice(size, num_assigned, replace=False)
                }

                state = {
                    "variable_assignments": var_assignments,
                    "clauses": problem["clauses"],
                }
                states.append(state)

            # Test each encoder
            for name, encoder in self.encoders.items():
                # Warmup (to avoid JIT compilation effects in the timing)
                if warmup:
                    for _ in range(3):
                        _ = encoder.encode(states[0])

                # Measure encoding time
                start_time = time.time()
                for state in states:
                    _ = encoder.encode(state)
                end_time = time.time()

                # Calculate average time per state
                avg_time = (end_time - start_time) / len(states)
                results["encoding_time"][name].append(avg_time)

                # Memory usage estimation is tricky and platform-dependent
                # This is a placeholder for a more sophisticated approach
                if PSUTIL_AVAILABLE:
                    try:
                        import os

                        # Measure memory before and after encoding a batch
                        process = psutil.Process(os.getpid())
                        mem_before = process.memory_info().rss
                        encoded_states = [encoder.encode(state) for state in states[:5]]
                        mem_after = process.memory_info().rss

                        # Estimate memory per state
                        memory_per_state = (mem_after - mem_before) / 5
                        results["memory_usage"][name].append(
                            memory_per_state / 1024 / 1024
                        )  # Convert to MB
                    except Exception as e:
                        print(f"Memory usage estimation failed: {e}")
                        results["memory_usage"][name].append(None)
                else:
                    # Fallback if psutil not available
                    results["memory_usage"][name].append(None)

        self.results = results
        return results

    def visualize_efficiency_results(self) -> dict[str, plt.Figure]:
        """
        Create visualizations of efficiency test results.

        Returns:
            Dictionary of matplotlib figures
        """
        if not self.results:
            raise ValueError("No efficiency results. Run efficiency tests first.")

        figures = {}

        # 1. Encoding time vs problem size
        fig, ax = plt.subplots(figsize=(10, 6))

        for name in self.encoders:
            ax.plot(
                self.results["problem_sizes"],
                self.results["encoding_time"][name],
                marker="o",
                label=name,
            )

        ax.set_title("Encoding Time vs Problem Size")
        ax.set_xlabel("Number of Variables")
        ax.set_ylabel("Average Encoding Time (seconds)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        figures["encoding_time"] = fig

        # 2. Memory usage vs problem size (if available)
        memory_data_available = any(
            all(x is not None for x in self.results["memory_usage"][name])
            for name in self.encoders
        )

        if memory_data_available:
            fig, ax = plt.subplots(figsize=(10, 6))

            for name in self.encoders:
                if all(x is not None for x in self.results["memory_usage"][name]):
                    ax.plot(
                        self.results["problem_sizes"],
                        self.results["memory_usage"][name],
                        marker="o",
                        label=name,
                    )

            ax.set_title("Memory Usage vs Problem Size")
            ax.set_xlabel("Number of Variables")
            ax.set_ylabel("Memory Usage per State (MB)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            figures["memory_usage"] = fig

        return figures
