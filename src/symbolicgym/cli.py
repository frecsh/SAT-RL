"""
SymbolicGym CLI launcher for training, evaluation, and benchmarking
"""
import argparse
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="SymbolicGym CLI")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=str, required=True)
    train_parser.add_argument("--debug", action="store_true")

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--config", type=str, required=True)
    eval_parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    if args.command == "train":
        with open(args.config) as f:
            config = yaml.safe_load(f)
        # TODO: Call training script with config and debug flag
        print(f"[CLI] Training with config: {args.config}, debug={args.debug}")
    elif args.command == "eval":
        with open(args.config) as f:
            config = yaml.safe_load(f)
        # TODO: Call eval script with config and debug flag
        print(f"[CLI] Evaluating with config: {args.config}, debug={args.debug}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
