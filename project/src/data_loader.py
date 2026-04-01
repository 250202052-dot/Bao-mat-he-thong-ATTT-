from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import detect_imbalance_severity


BENIGN_TOKENS = {"0", "benign", "normal", "normality", "legitimate", "safe"}


@dataclass
class DatasetBundle:
    name: str
    features: pd.DataFrame
    target: pd.Series
    label_mapping: dict[str, int]
    stats: dict[str, object]


def _binarize_labels(series: pd.Series, positive_class=None) -> tuple[pd.Series, dict[str, int]]:
    clean = series.copy()

    if pd.api.types.is_numeric_dtype(clean) and clean.nunique(dropna=True) <= 2:
        values = sorted(v for v in clean.dropna().unique())
        negative_value = values[0]
        positive_value = positive_class if positive_class is not None else values[-1]
        y = (clean == positive_value).astype(int)
        return y, {str(negative_value): 0, str(positive_value): 1}

    if pd.api.types.is_numeric_dtype(clean) and clean.nunique(dropna=True) > 2:
        unique_values = sorted(v for v in clean.dropna().unique())
        if 0 in unique_values:
            y = (clean != 0).astype(int)
            return y, {"0": 0, "non_zero": 1}
        positive_value = positive_class if positive_class is not None else unique_values[-1]
        y = (clean == positive_value).astype(int)
        return y, {"other": 0, str(positive_value): 1}

    normalized = clean.astype(str).str.strip()
    if positive_class is not None and normalized.eq(str(positive_class)).any():
        y = (normalized == str(positive_class)).astype(int)
        return y, {"other": 0, str(positive_class): 1}

    mapping: dict[str, int] = {}
    for label in normalized.unique().tolist():
        mapping[label] = 0 if label.lower() in BENIGN_TOKENS else 1
    y = normalized.map(mapping).astype(int)
    return y, mapping


def _safe_sample(df: pd.DataFrame, y: pd.Series, random_seed: int, max_rows=None, sample_frac=None):
    if max_rows is None and sample_frac is None:
        return df.reset_index(drop=True), y.reset_index(drop=True)

    if sample_frac is not None:
        sample_size = max(1, int(len(df) * float(sample_frac)))
    else:
        sample_size = min(int(max_rows), len(df))

    if sample_size >= len(df):
        return df.reset_index(drop=True), y.reset_index(drop=True)

    sampled_df, _, sampled_y, _ = train_test_split(
        df,
        y,
        train_size=sample_size,
        stratify=y,
        random_state=random_seed,
    )
    return sampled_df.reset_index(drop=True), sampled_y.reset_index(drop=True)


def load_dataset(name: str, dataset_cfg: dict, random_seed: int) -> DatasetBundle:
    csv_path = Path(dataset_cfg["path"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    label_column = dataset_cfg["label_column"]
    if label_column not in df.columns:
        raise KeyError(f"Label column '{label_column}' not found in {csv_path}")

    target, label_mapping = _binarize_labels(df[label_column], dataset_cfg.get("positive_class"))
    df = df.drop(columns=[label_column])

    drop_columns = [col for col in dataset_cfg.get("drop_columns", []) if col in df.columns]
    if drop_columns:
        df = df.drop(columns=drop_columns)

    df, target = _safe_sample(
        df=df,
        y=target,
        random_seed=random_seed,
        max_rows=dataset_cfg.get("max_rows"),
        sample_frac=dataset_cfg.get("sample_frac"),
    )

    numeric_columns = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [col for col in df.columns if col not in numeric_columns]
    class_counts = target.value_counts().sort_index()
    minority_count = int(class_counts.min())
    majority_count = int(class_counts.max())
    minority_ratio = minority_count / max(majority_count, 1)

    stats = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "label_distribution": {str(k): int(v) for k, v in class_counts.items()},
        "imbalance_ratio_minority_to_majority": float(minority_ratio),
        "imbalance_severity": detect_imbalance_severity(minority_ratio),
        "missing_values_total": int(df.isna().sum().sum()),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "dropped_columns": drop_columns,
        "label_mapping": label_mapping,
    }

    return DatasetBundle(
        name=name,
        features=df,
        target=target,
        label_mapping=label_mapping,
        stats=stats,
    )


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    validation_size: float,
    random_seed: int,
):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_seed,
    )

    val_relative_size = validation_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative_size,
        stratify=y_train_val,
        random_state=random_seed,
    )

    return (
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_val.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )
