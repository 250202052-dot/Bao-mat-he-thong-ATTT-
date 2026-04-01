from __future__ import annotations

from dataclasses import dataclass

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier, RUSBoostClassifier

from .imbalance_estimators import FocalLogisticRegression

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


@dataclass
class ModelSpec:
    name: str
    estimator: object
    supports_class_weight: bool
    needs_scaling: bool
    family: str
    allowed_strategies: tuple[str, ...] | None = None


def get_model_specs(random_seed: int) -> dict[str, ModelSpec]:
    specs = {
        "logistic_regression": ModelSpec(
            "logistic_regression",
            LogisticRegression(max_iter=2000, solver="liblinear", random_state=random_seed),
            True,
            True,
            "linear",
            None,
        ),
        "random_forest": ModelSpec(
            "random_forest",
            RandomForestClassifier(n_estimators=250, n_jobs=-1, random_state=random_seed),
            True,
            False,
            "tree",
            None,
        ),
        "gradient_boosting": ModelSpec(
            "gradient_boosting",
            GradientBoostingClassifier(random_state=random_seed),
            False,
            False,
            "tree",
            None,
        ),
        "extra_trees": ModelSpec(
            "extra_trees",
            ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=random_seed),
            True,
            False,
            "tree",
            None,
        ),
        "linear_svm": ModelSpec(
            "linear_svm",
            SGDClassifier(loss="hinge", max_iter=2000, tol=1e-3, early_stopping=True, random_state=random_seed),
            True,
            True,
            "linear",
            None,
        ),
        "knn": ModelSpec("knn", KNeighborsClassifier(n_neighbors=7), False, True, "distance", None),
        "decision_tree": ModelSpec(
            "decision_tree",
            DecisionTreeClassifier(random_state=random_seed),
            True,
            False,
            "tree",
            None,
        ),
        "balanced_random_forest": ModelSpec(
            "balanced_random_forest",
            BalancedRandomForestClassifier(n_estimators=250, random_state=random_seed, n_jobs=-1),
            False,
            False,
            "ensemble",
            ("baseline",),
        ),
        "easy_ensemble": ModelSpec(
            "easy_ensemble",
            EasyEnsembleClassifier(n_estimators=20, random_state=random_seed, n_jobs=-1),
            False,
            False,
            "ensemble",
            ("baseline",),
        ),
        "rusboost": ModelSpec(
            "rusboost",
            RUSBoostClassifier(random_state=random_seed),
            False,
            False,
            "ensemble",
            ("baseline",),
        ),
        "focal_logistic": ModelSpec(
            "focal_logistic",
            FocalLogisticRegression(random_state=random_seed, gamma=2.0),
            False,
            True,
            "loss",
            ("baseline",),
        ),
    }

    if XGBClassifier is not None:
        specs["xgboost"] = ModelSpec(
            "xgboost",
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=random_seed,
                n_jobs=-1,
            ),
            False,
            False,
            "boosting",
            None,
        )

    if LGBMClassifier is not None:
        specs["lightgbm"] = ModelSpec(
            "lightgbm",
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_seed,
                n_jobs=-1,
                verbosity=-1,
            ),
            True,
            False,
            "boosting",
            None,
        )

    return specs


def prepare_estimator(model_spec: ModelSpec, strategy_name: str, y_train) -> object:
    if model_spec.allowed_strategies is not None and strategy_name not in model_spec.allowed_strategies:
        raise ValueError(
            f"Model '{model_spec.name}' is only intended for strategies: {', '.join(model_spec.allowed_strategies)}"
        )

    estimator = clone(model_spec.estimator)
    if strategy_name != "class_weight":
        return estimator

    if model_spec.supports_class_weight and hasattr(estimator, "set_params"):
        return estimator.set_params(class_weight="balanced")

    if model_spec.name == "xgboost":
        counts = y_train.value_counts()
        scale_pos_weight = counts.get(0, 1) / max(counts.get(1, 1), 1)
        return estimator.set_params(scale_pos_weight=scale_pos_weight)

    if model_spec.name == "lightgbm":
        return estimator.set_params(class_weight="balanced")

    raise ValueError(f"Model '{model_spec.name}' does not support class_weight.")
