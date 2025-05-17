"""Unified visualization dashboard for SymbolicGym."""

import matplotlib.pyplot as plt


def show_dashboard(figures, layout=(1, 1)):
    """Display multiple matplotlib figures in a dashboard layout.

    Args:
        figures: list of matplotlib Figure or Axes
        layout: tuple (rows, cols)
    """
    rows, cols = layout
    fig, axs = plt.subplots(rows, cols, squeeze=False)
    for i, ax in enumerate(axs.flat):
        if i < len(figures):
            f = figures[i]
            # Accept both Figure and Axes
            if hasattr(f, "axes") and isinstance(f, plt.Figure):
                # Draw the contents of the figure onto the dashboard axis
                for child_ax in f.axes:
                    for artist in child_ax.get_children():
                        try:
                            artist.figure = fig
                            ax.add_artist(artist)
                        except Exception:
                            pass
                    ax.set_xlim(child_ax.get_xlim())
                    ax.set_ylim(child_ax.get_ylim())
                    ax.set_title(child_ax.get_title())
            elif hasattr(f, "imshow") or hasattr(f, "plot"):
                # If it's an Axes, skip for now (could add more logic here)
                continue
    plt.tight_layout()
    plt.show()
