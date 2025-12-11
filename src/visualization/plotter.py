"""Module for plotting 3D data."""

import matplotlib.pyplot as plt
import numpy as np


def plot_3d_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    output_path: str | None = None,
    show: bool = False,
):
    """Plots a 3D surface.

    Args:
        x: Grid X coordinates.
        y: Grid Y coordinates.
        z: Grid Z coordinates (heights).
        output_path: Path to save the plot.
        show: Whether to show the plot interactively.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot surface
    # Use a colormap that resembles terrain (e.g., 'terrain' or 'gist_earth')
    surf = ax.plot_surface(
        x, y, z, cmap="terrain", linewidth=0, antialiased=False, alpha=0.8
    )

    # Add colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5, label="Height")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")
    ax.set_title("3D Terrain Model")

    # Invert Y axis to match image coordinates (top-left origin)
    ax.invert_yaxis()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved 3D plot to {output_path}")

    if show:
        plt.show()

    plt.close(fig)
