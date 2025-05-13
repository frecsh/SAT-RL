#!/usr/bin/env python3
"""
Generate an architecture diagram for the SAT+RL project.
This script creates a visual representation of how the different components
of the SAT+RL project interact with each other.
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def create_box(ax, x, y, width, height, name, color="lightblue", alpha=0.7):
    """Create a box with rounded corners and a label."""
    rect = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=patches.BoxStyle("Round", pad=0.02, rounding_size=0.1),
        facecolor=color,
        alpha=alpha,
        edgecolor="black",
        linewidth=1,
    )
    ax.add_patch(rect)
    ax.text(
        x + width / 2, y + height / 2, name, ha="center", va="center", fontweight="bold"
    )
    return rect


def create_arrow(
    ax, start, end, color="black", style="->", width=1.5, alpha=0.7, label=None
):
    """Create an arrow from start to end."""
    x1, y1 = start
    x2, y2 = end

    # Use FancyArrowPatch for proper arrow styling
    arrow = patches.FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        connectionstyle="arc3,rad=0.1",  # Slightly curved arrow
        mutation_scale=15,  # Size of arrow head
        linewidth=width,
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
    )
    ax.add_patch(arrow)

    if label:
        # Add label in the middle of the arrow
        # Calculate position with some offset for the curved arrow
        middle_x = (x1 + x2) / 2
        middle_y = (y1 + y2) / 2 + 1  # Slight offset for curved arrow
        ax.text(
            middle_x,
            middle_y,
            label,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
        )


def generate_architecture_diagram():
    """Generate the complete architecture diagram."""
    fig, ax = plt.subplots(figsize=(16, 12))

    # Set up the plot
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 75)
    ax.axis("off")
    ax.set_title("SAT+RL Project Architecture", fontsize=16, pad=20)

    # Define color scheme
    colors = {
        "core": "#AED6F1",  # light blue
        "agent": "#D5F5E3",  # light green
        "oracle": "#FADBD8",  # light red
        "gan": "#E8DAEF",  # light purple
        "curriculum": "#FCF3CF",  # light yellow
        "anytime": "#F5CBA7",  # light orange
        "benchmark": "#D6DBDF",  # light gray
    }

    # Create main component groups
    create_box(ax, 5, 60, 90, 10, "SAT+RL Architecture", color="#D6EAF8", alpha=0.3)

    # Core components
    core = create_box(ax, 5, 45, 25, 10, "Core Components", color=colors["core"])

    # Agent approaches
    agents = create_box(ax, 35, 45, 30, 10, "Agent Approaches", color=colors["agent"])

    # Oracle and GAN components
    oracle_gan = create_box(
        ax, 70, 45, 25, 10, "Oracles & GANs", color=colors["oracle"]
    )

    # Enhanced approaches (new implementations)
    enhanced = create_box(
        ax, 5, 30, 90, 10, "Enhanced Approaches", color="#D6EAF8", alpha=0.3
    )

    # Deep Q-Learning
    dqn = create_box(ax, 10, 15, 20, 10, "Deep Q-Learning", color=colors["agent"])

    # Improved GAN
    improved_gan = create_box(ax, 35, 15, 20, 10, "Improved GAN", color=colors["gan"])

    # Oracle Distillation
    distillation = create_box(
        ax, 60, 15, 20, 10, "Oracle Distillation", color=colors["oracle"]
    )

    # Curriculum Learning
    curriculum = create_box(
        ax, 22.5, 2, 20, 10, "Curriculum Learning", color=colors["curriculum"]
    )

    # Anytime Solver
    anytime = create_box(ax, 47.5, 2, 20, 10, "Anytime Solver", color=colors["anytime"])

    # Benchmark Analysis
    benchmark = create_box(
        ax, 85, 30, 10, 10, "Benchmark Analysis", color=colors["benchmark"]
    )

    # Add specific components as text
    component_text = """
    Core Components:
    - main.py
    - sat_problems.py

    Agent Approaches:
    - multi_q_sat.py (Cooperative)
    - multi_q_sat_comp.py (Competitive)
    - multi_q_sat_comm.py (Communicative)
    - multi_q_sat_oracle.py (Oracle-guided)

    Oracles & GANs:
    - sat_oracle.py
    - sat_gan.py
    - progressive_sat_gan.py

    Deep Q-Learning:
    - deep_q_sat_agent.py

    Improved GAN:
    - improved_sat_gan.py

    Oracle Distillation:
    - oracle_distillation_agent.py

    Curriculum Learning:
    - curriculum_sat_learner.py

    Anytime Solver:
    - anytime_sat_solver.py

    Benchmark Analysis:
    - analyze_benchmarks.py
    - compare.py
    """

    fig.text(0.02, 0.02, component_text, fontsize=9, family="monospace")

    # Create connecting arrows
    # Core to agent approaches
    create_arrow(ax, (30, 50), (35, 50), label="Problem Definition")

    # Agent to Oracle/GAN
    create_arrow(ax, (65, 50), (70, 50), label="Integration")

    # Core to enhanced approaches
    create_arrow(ax, (20, 45), (20, 40), label="Base Methods")

    # Agent to enhanced approaches
    create_arrow(ax, (50, 45), (50, 40), label="Advanced Techniques")

    # Oracle/GAN to enhanced approaches
    create_arrow(ax, (80, 45), (80, 40), label="Solution Generation")

    # Enhanced to specific implementations
    create_arrow(ax, (20, 30), (20, 25), label="Function\nApproximation")

    # Enhanced to GAN improvements
    create_arrow(ax, (45, 30), (45, 25), label="Experience\nReplay")

    # Enhanced to Oracle Distillation
    create_arrow(ax, (70, 30), (70, 25), label="Knowledge\nTransfer")

    # DQN to Curriculum
    create_arrow(ax, (20, 15), (30, 12), label="")

    # Improved GAN to Curriculum
    create_arrow(ax, (40, 15), (35, 12), label="Progressive\nTraining")

    # Deep Q-Learning to Anytime Solver
    create_arrow(ax, (25, 15), (50, 12), label="")

    # Distillation to Anytime Solver
    create_arrow(ax, (65, 15), (60, 12), label="")

    # Analysis connections
    create_arrow(ax, (90, 30), (70, 10), style="-|>", label="Evaluation")

    # Demo Integration
    demo = create_box(
        ax,
        35,
        22,
        30,
        5,
        "satrlgym_demo.py: Demo Integration",
        color="#D6EAF8",
        alpha=1,
    )

    create_arrow(ax, (30, 15), (37, 22), style="-|>")
    create_arrow(ax, (45, 15), (45, 22), style="-|>")
    create_arrow(ax, (60, 15), (53, 22), style="-|>")
    create_arrow(ax, (32.5, 7), (40, 22), style="-|>", label="")
    create_arrow(ax, (57.5, 7), (50, 22), style="-|>", label="")

    # Save the diagram
    plt.savefig("satrlgym_architecture.png", dpi=300, bbox_inches="tight")
    print("Architecture diagram saved as 'satrlgym_architecture.png'")


if __name__ == "__main__":
    generate_architecture_diagram()
