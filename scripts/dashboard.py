"""Minimal Dashboard for SymbolicGym Results
- Loads logs/results and provides interactive visualization
- Optionally use Streamlit for web-based dashboard.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os

import pandas as pd

try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Allow test to force fallback mode
if os.environ.get("SYMGYM_FORCE_NO_STREAMLIT") == "1":
    STREAMLIT_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description="SymbolicGym Results Dashboard")
    parser.add_argument(
        "--input", type=str, required=True, help="Input results directory"
    )
    args = parser.parse_args()
    input_dir = Path(args.input)

    try:
        from src.symbolicgym.utils import logging as logging_utils
    except ImportError:
        import importlib

        importlib.import_module("src.symbolicgym.utils.logging")

    if STREAMLIT_AVAILABLE:
        st.title("SymbolicGym Results Dashboard")
        st.write("Results directory:", str(input_dir))
        st.info(
            "If running via 'streamlit run', pass arguments after '--', e.g.:\n"
            "  streamlit run scripts/dashboard.py -- --input results/"
        )
        # Try to load and display main results table
        results_file = input_dir / "sat_benchmark_results.csv"
        if results_file.exists():
            df = pd.read_csv(results_file)
            st.subheader("SAT Benchmark Results")
            st.dataframe(df)
            if "solve_rate" in df.columns:
                st.line_chart(df["solve_rate"])
            if "steps" in df.columns:
                st.line_chart(df["steps"])
        else:
            st.warning(f"No results file found at {results_file}")
        # TODO: Add more plots/tables as needed
    else:
        print(
            "Streamlit not installed. Please install with 'pip install streamlit' for web dashboard."
        )
        print(
            "If running via 'streamlit run', pass arguments after '--', e.g.:\n"
            "  streamlit run scripts/dashboard.py -- --input results/"
        )
        # Fallback: print summary table from results/sat_benchmark_results.csv
        results_file = input_dir / "sat_benchmark_results.csv"
        if results_file.exists():
            df = pd.read_csv(results_file)
            print("\nSAT Benchmark Results (first 10 rows):")
            print(df.head(10).to_string(index=False))
            if "solve_rate" in df.columns:
                print("\nSolve rate summary:")
                print(df["solve_rate"].describe())
            if "steps" in df.columns:
                print("\nSteps summary:")
                print(df["steps"].describe())
        else:
            print(f"No results file found at {results_file}")


if __name__ == "__main__":
    main()
