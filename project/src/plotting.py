from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from .utils import ensure_dir


def plot_confusion_matrix(cm, title: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=["Normal", "Attack"]).plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_precision_recall(y_true, y_scores, title: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_scores, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_roc_curve(y_true, y_scores, title: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_scores, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_feature_importance(feature_names: list[str], importances, title: str, output_path: Path, top_n: int = 20) -> None:
    ensure_dir(output_path.parent)
    values = np.asarray(importances)
    if values.ndim != 1 or values.size == 0:
        return

    order = np.argsort(values)[-top_n:]
    names = [feature_names[i] for i in order]
    selected = values[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, selected)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
