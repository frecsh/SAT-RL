#!/usr/bin/env python
"""
Dataset Quality Assurance CLI tool.

This script provides a command-line interface for running dataset validation,
statistics generation, and anomaly detection on experience replay datasets.

Usage:
    python dataset_qa_cli.py validate /path/to/dataset.npz
    python dataset_qa_cli.py stats /path/to/dataset.npz
    python dataset_qa_cli.py anomalies /path/to/dataset.npz
    python dataset_qa_cli.py report /path/to/dataset.npz
    python dataset_qa_cli.py batch /path/to/dataset/directory

Examples:
    # Run all QA checks and generate a full report
    python dataset_qa_cli.py report data/experiences/episode_001.npz

    # Validate a batch of files and output as JSON
    python dataset_qa_cli.py batch data/experiences/ --format json
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

from satrlgym.utils.dataset_qa import (
    AnomalyDetector,
    DatasetValidator,
    StatisticsGenerator,
    validate_and_report,
)

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_dict(data: dict, indent: int = 0) -> None:
    """
    Pretty print a dictionary with specified indentation.

    Args:
        data: Dictionary to print
        indent: Indentation level
    """
    for key, value in data.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_dict(value, indent + 2)
        elif isinstance(value, list):
            if len(value) > 0 and not isinstance(value[0], (dict, list)):
                if len(value) > 10:
                    print(
                        " " * indent
                        + f"{key}: [{value[0]}, {value[1]}, ... ({len(value)} items)]"
                    )
                else:
                    print(" " * indent + f"{key}: {value}")
            else:
                print(" " * indent + f"{key}:")
                for i, item in enumerate(value[:5]):
                    if isinstance(item, dict):
                        print(" " * (indent + 2) + f"[{i}]:")
                        print_dict(item, indent + 4)
                    else:
                        print(" " * (indent + 2) + f"[{i}]: {item}")
                if len(value) > 5:
                    print(" " * (indent + 2) + f"... ({len(value) - 5} more items)")
        else:
            print(" " * indent + f"{key}: {value}")


def validate_command(args: argparse.Namespace) -> None:
    """Run validation on a dataset file and print results."""
    validator = DatasetValidator()
    results = validator.validate_dataset(args.file)

    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        print("\n===== Dataset Validation Results =====")
        print(f"File: {args.file}")
        print(f"Valid: {results['is_valid']}")

        if results["errors"]:
            print("\nErrors:")
            for error in results["errors"]:
                print(f"  - {error}")

        if results["warnings"]:
            print("\nWarnings:")
            for warning in results["warnings"]:
                print(f"  - {warning}")

        if "metadata" in results and results["metadata"]:
            print("\nMetadata:")
            for key, value in results["metadata"].items():
                print(f"  {key}: {value}")


def stats_command(args: argparse.Namespace) -> None:
    """Generate statistics for a dataset file and print results."""
    stats_gen = StatisticsGenerator()
    statistics = stats_gen.generate_statistics(args.file)

    if args.format == "json":
        print(json.dumps(statistics, indent=2))
    else:
        print("\n===== Dataset Statistics =====")
        print(f"File: {args.file}")

        for field, field_stats in statistics.items():
            print(f"\nField: {field}")
            for stat_name, stat_value in field_stats.items():
                print(f"  {stat_name}: {stat_value}")


def anomalies_command(args: argparse.Namespace) -> None:
    """Detect anomalies in a dataset file and print results."""
    detector = AnomalyDetector(
        threshold_std_multiplier=args.threshold,
        min_reward=args.min_reward,
        max_reward=args.max_reward,
    )
    results = detector.detect_anomalies(args.file)

    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        print("\n===== Dataset Anomaly Detection =====")
        print(f"File: {args.file}")
        print(f"Anomalies detected: {results['anomalies_detected']}")
        print(f"Total anomalies: {results['anomaly_count']}")

        if results["field_anomalies"]:
            print("\nAnomaly details by field:")
            for field, anomalies in results["field_anomalies"].items():
                print(f"\n  Field: {field}")
                print(f"  Count: {anomalies['count']}")
                print(
                    f"  Threshold range: [{anomalies['threshold_low']:.4f}, {anomalies['threshold_high']:.4f}]"
                )
                if len(anomalies["indices"]) > 10:
                    print(f"  Indices: {anomalies['indices'][:10]} ... (more)")
                else:
                    print(f"  Indices: {anomalies['indices']}")


def report_command(args: argparse.Namespace) -> None:
    """Generate a comprehensive report for a dataset file."""
    report = validate_and_report(args.file)

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print("\n===== Dataset Quality Assurance Report =====")
        print(f"File: {args.file}")
        print(f"Status: {report['summary']['status'].upper()}")
        print(f"Valid: {report['summary']['is_valid']}")
        print(f"Transition count: {report['summary']['transition_count']}")
        print(f"Anomaly count: {report['summary']['anomaly_count']}")

        if not report["summary"]["is_valid"]:
            print("\nValidation errors:")
            for error in report["validation"]["errors"]:
                print(f"  - {error}")

        if report["statistics"]:
            print("\nField statistics summary:")
            for field, stats in report["statistics"].items():
                print(f"\n  Field: {field}")
                print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")

        if report["anomalies"] and report["anomalies"]["field_anomalies"]:
            print("\nAnomaly summary:")
            for field, anomalies in report["anomalies"]["field_anomalies"].items():
                print(f"  {field}: {anomalies['count']} anomalies")


def batch_command(args: argparse.Namespace) -> None:
    """Process a batch of files in a directory."""
    # Get all files with supported extensions
    extensions = [".npz", ".hdf5", ".h5", ".jsonl", ".parquet"]
    file_paths = []

    for ext in extensions:
        pattern = os.path.join(args.directory, f"*{ext}")
        file_paths.extend(glob.glob(pattern))

    print(f"Found {len(file_paths)} files to process")

    # Process each file
    results = []
    for i, file_path in enumerate(file_paths):
        if i > 0 and i % 10 == 0:
            print(f"Processed {i}/{len(file_paths)} files...")

        report = validate_and_report(file_path)
        results.append(
            {
                "file": os.path.basename(file_path),
                "path": file_path,
                "status": report["summary"]["status"],
                "is_valid": report["summary"]["is_valid"],
                "transition_count": report["summary"]["transition_count"],
                "anomaly_count": report["summary"]["anomaly_count"],
            }
        )

    # Output results
    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        print("\n===== Batch Processing Results =====")
        print(f"Directory: {args.directory}")
        print(f"Files processed: {len(results)}")

        # Group by status
        status_groups = {}
        for result in results:
            status = result["status"]
            if status not in status_groups:
                status_groups[status] = []
            status_groups[status].append(result)

        print("\nResults by status:")
        for status, group in status_groups.items():
            print(f"\n  {status.upper()}: {len(group)} files")

            for result in group[:5]:  # Show first 5 files in each group
                print(
                    f"    - {result['file']}: {result['transition_count']} transitions, {result['anomaly_count']} anomalies"
                )

            if len(group) > 5:
                print(f"    ... ({len(group) - 5} more files)")

        # Print warning for invalid files
        invalid_files = [r for r in results if not r["is_valid"]]
        if invalid_files:
            print(f"\nWARNING: {len(invalid_files)} invalid files found!")
            for invalid in invalid_files[:10]:
                print(f"  - {invalid['file']}")
            if len(invalid_files) > 10:
                print(f"  ... ({len(invalid_files) - 10} more)")


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Dataset Quality Assurance tools")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a dataset file")
    validate_parser.add_argument("file", help="Path to the dataset file")
    validate_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    # Statistics command
    stats_parser = subparsers.add_parser(
        "stats", help="Generate statistics for a dataset file"
    )
    stats_parser.add_argument("file", help="Path to the dataset file")
    stats_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    # Anomalies command
    anomalies_parser = subparsers.add_parser(
        "anomalies", help="Detect anomalies in a dataset file"
    )
    anomalies_parser.add_argument("file", help="Path to the dataset file")
    anomalies_parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Threshold multiplier for anomaly detection (default: 3.0)",
    )
    anomalies_parser.add_argument(
        "--min-reward", type=float, default=None, help="Minimum valid reward value"
    )
    anomalies_parser.add_argument(
        "--max-reward", type=float, default=None, help="Maximum valid reward value"
    )
    anomalies_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    # Report command
    report_parser = subparsers.add_parser(
        "report", help="Generate a comprehensive report"
    )
    report_parser.add_argument("file", help="Path to the dataset file")
    report_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch", help="Process a batch of files in a directory"
    )
    batch_parser.add_argument(
        "directory", help="Path to the directory containing dataset files"
    )
    batch_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    args = parser.parse_args()

    if args.command == "validate":
        validate_command(args)
    elif args.command == "stats":
        stats_command(args)
    elif args.command == "anomalies":
        anomalies_command(args)
    elif args.command == "report":
        report_command(args)
    elif args.command == "batch":
        batch_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
