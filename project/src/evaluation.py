from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from .models import ModelSpec, prepare_estimator
from .preprocessing import build_preprocessor
from .sampling import get_sampler, summarize_resampling


SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "balanced_accuracy": "balanced_accuracy",
}


def build_training_pipeline(model_spec: ModelSpec, strategy_name: str, X_template: pd.DataFrame, y_train: pd.Series, random_seed: int):
    preprocessor, _, _ = build_preprocessor(X_template, scale_numeric=model_spec.needs_scaling)
    estimator = prepare_estimator(model_spec, strategy_name, y_train)
    sampler = get_sampler(strategy_name, random_seed)

    steps = [("preprocessor", preprocessor)]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", estimator))
    return ImbPipeline(steps=steps)


def get_scores(fitted_pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(fitted_pipeline, "predict_proba"):
        probabilities = fitted_pipeline.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return np.zeros(probabilities.shape[0], dtype=float)
    if hasattr(fitted_pipeline, "decision_function"):
        scores = np.asarray(fitted_pipeline.decision_function(X), dtype=float)
        if scores.max() > scores.min():
            return (scores - scores.min()) / (scores.max() - scores.min())
        return np.zeros_like(scores, dtype=float)
    return np.asarray(fitted_pipeline.predict(X), dtype=float)


def choose_threshold(y_true, y_scores, strategy: str, recall_priority_target: float) -> tuple[float, dict[str, float]]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    thresholds = np.append(thresholds, 1.0)
    f1_scores = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0,
    )

    if strategy == "recall_priority":
        valid = np.where(recalls >= recall_priority_target)[0]
        best_idx = valid[np.argmax(precisions[valid])] if len(valid) > 0 else int(np.argmax(recalls))
    else:
        best_idx = int(np.argmax(f1_scores))

    threshold = float(np.clip(thresholds[best_idx], 0.0, 1.0))
    diagnostics = {
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx]),
        "f1": float(f1_scores[best_idx]),
    }
    return threshold, diagnostics


def compute_metrics(y_true, y_scores, threshold: float) -> dict[str, float | list[list[int]]]:
    y_pred = (y_scores >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def run_cross_validation(pipeline, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int, random_seed: int):
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=splitter,
        scoring=SCORING,
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )
    return {metric.replace("test_", "cv_"): float(np.mean(values)) for metric, values in cv_results.items() if metric.startswith("test_")}


def fit_and_evaluate_candidate(
    model_spec: ModelSpec,
    strategy_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_seed: int,
    threshold_strategy: str,
    recall_priority_target: float,
):
    pipeline = build_training_pipeline(model_spec, strategy_name, X_train, y_train, random_seed)

    fit_start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    score_start = time.perf_counter()
    val_scores = get_scores(pipeline, X_val)
    threshold, threshold_diag = choose_threshold(y_val, val_scores, threshold_strategy, recall_priority_target)
    val_metrics = compute_metrics(y_val, val_scores, threshold)
    score_seconds = time.perf_counter() - score_start

    return {
        "pipeline": pipeline,
        "threshold": threshold,
        "threshold_diagnostics": threshold_diag,
        "validation_metrics": val_metrics,
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds,
    }


def evaluate_on_test(pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float):
    scores = get_scores(pipeline, X_test)
    metrics = compute_metrics(y_test, scores, threshold)
    report = classification_report(y_test, (scores >= threshold).astype(int), digits=4, zero_division=0, output_dict=True)
    return scores, metrics, report


def derive_resampling_summary(pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, dict[str, int]] | None:
    if "sampler" not in pipeline.named_steps:
        return None
    transformed = pipeline.named_steps["preprocessor"].transform(X_train)
    _, y_resampled = pipeline.named_steps["sampler"].fit_resample(transformed, y_train)
    return summarize_resampling(y_train, y_resampled)


def save_classification_report(report: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
