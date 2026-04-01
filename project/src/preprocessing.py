from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


def detect_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_columns = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [col for col in X.columns if col not in numeric_columns]
    return numeric_columns, categorical_columns


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns, categorical_columns = detect_feature_types(X)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_columns, categorical_columns


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    feature_names: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue
        if columns is None or len(columns) == 0:
            continue
        if hasattr(transformer, "named_steps") and "encoder" in transformer.named_steps:
            encoder = transformer.named_steps["encoder"]
            try:
                check_is_fitted(encoder)
            except Exception:
                continue
            feature_names.extend(encoder.get_feature_names_out(columns).tolist())
        else:
            feature_names.extend(list(columns))
    return feature_names
