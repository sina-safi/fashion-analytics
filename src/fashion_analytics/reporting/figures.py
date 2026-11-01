"""Reusable plotting helpers that save figures to outputs/figures/."""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd

from fashion_analytics.config import FIGURES_DIR

logger = logging.getLogger(__name__)


def save_bar(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    filename: str,
    horizontal: bool = False,
    rotate: int = 45,
) -> None:
    """Save a bar chart of ``df[value_col]`` indexed by row labels."""
    fig, ax = plt.subplots(figsize=(10, 5))

    if horizontal:
        ax.barh(df.index, df[value_col])
        ax.set_xlabel(value_col.capitalize())
    else:
        ax.bar(df.index.astype(str), df[value_col])
        ax.set_ylabel(value_col.capitalize())

    ax.set_title(title)
    plt.xticks(rotation=0 if horizontal else rotate)
    plt.tight_layout()

    path = FIGURES_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)


def save_line(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    filename: str,
) -> None:
    """Save a line chart of ``df[y_col]`` over ``df[x_col]``."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df[x_col], df[y_col], marker="o", markersize=3)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_col.capitalize())
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = FIGURES_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)