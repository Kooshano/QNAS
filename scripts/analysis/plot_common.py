"""Shared plotting configuration utilities for analysis scripts."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class PlotConfig:
    """Centralized configuration for all plots."""
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    font_size: int = 14
    file_ext: str = "png"
    dpi: int = 300
    figsize_small: tuple = (8, 6)
    figsize_medium: tuple = (12, 8)
    figsize_large: tuple = (16, 10)
    figsize_wide: tuple = (18, 6)
    colors: List = field(default_factory=lambda: sns.color_palette("colorblind"))

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)

    def ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


def save_figure(fig, name: str, config: PlotConfig = None, category: str = None):
    """Save figure with consistent naming and logging."""
    if config is None:
        config = PlotConfig()

    filename = f"{category}_{name}.{config.file_ext}" if category else f"{name}.{config.file_ext}"
    filepath = config.output_dir / filename

    if config.file_ext == "svg":
        fig.savefig(filepath, format="svg", bbox_inches="tight")
    else:
        fig.savefig(filepath, dpi=config.dpi, bbox_inches="tight")

    plt.close(fig)
    print(f"  [SAVED] {filename}")
    return filepath


def log_section(title: str) -> None:
    """Print section header for consistent logging."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def log_step(step_num: int, description: str) -> None:
    """Print step info for consistent logging."""
    print(f"\n[{step_num}] {description}")
