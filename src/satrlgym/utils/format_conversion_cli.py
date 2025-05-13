#!/usr/bin/env python
"""
Command-line interface for format conversion utilities.
"""
import argparse
import logging
import sys

from .format_conversion import get_converter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert between different RL data formats"
    )

    # Required arguments
    parser.add_argument(
        "action",
        choices=["import", "export"],
        help="Action to perform: import (external to internal) or export (internal to external)",
    )

    parser.add_argument(
        "format", choices=["rlds", "d4rl", "gym"], help="Format to convert from/to"
    )

    parser.add_argument("source", type=str, help="Source file or directory")

    parser.add_argument("target", type=str, help="Target file or directory")

    # Optional arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing data (default: 1000)",
    )

    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["none", "zstd", "gzip", "lz4"],
        help="Compression format for internal data (default: zstd)",
    )

    # Format-specific arguments
    # RLDS arguments
    parser.add_argument(
        "--rlds-compression",
        default="",
        choices=["", "GZIP", "ZLIB"],
        help="Compression for RLDS TFRecord files",
    )

    # Gym arguments
    parser.add_argument(
        "--episode-per-file",
        action="store_true",
        help="Save each episode in a separate file when exporting to Gym format",
    )

    # D4RL arguments
    parser.add_argument(
        "--metadata",
        type=str,
        default="{}",
        help="JSON string with metadata to include when exporting to D4RL format",
    )

    # General arguments
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times)",
    )

    return parser.parse_args()


def setup_logging(verbosity):
    """Set up logging based on verbosity level."""
    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    level = levels.get(verbosity, logging.DEBUG)
    logging.getLogger().setLevel(level)


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    try:
        # Get the converter for the specified format
        converter = get_converter(args.format)

        # Prepare kwargs for the converter
        kwargs = {
            "batch_size": args.batch_size,
            "compression": args.compression,
        }

        # Add format-specific arguments
        if args.format == "rlds":
            kwargs["rlds_compression"] = args.rlds_compression
        elif args.format == "gym":
            kwargs["episode_per_file"] = args.episode_per_file
        elif args.format == "d4rl":
            import json

            kwargs["metadata"] = json.loads(args.metadata)

        # Run the conversion
        if args.action == "import":
            result_path = converter.import_data(args.source, args.target, **kwargs)
            logger.info(f"Import complete. Data saved to {result_path}")
        else:  # export
            result_path = converter.export_data(args.source, args.target, **kwargs)
            logger.info(f"Export complete. Data saved to {result_path}")

        return 0

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install the required package for the selected format.")
        return 1

    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
