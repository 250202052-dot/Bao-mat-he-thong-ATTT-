from __future__ import annotations

from collections import Counter

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, KMeansSMOTE, RandomOverSampler, SMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler


def get_sampler(strategy_name: str, random_seed: int):
    # Data-level methods
    if strategy_name in {"baseline", "class_weight"}:
        return None
    if strategy_name == "random_oversample":
        return RandomOverSampler(random_state=random_seed)
    if strategy_name == "smote":
        return SMOTE(random_state=random_seed)
    # BSMOTE in many papers usually refers to Borderline-SMOTE.
    if strategy_name == "borderline_smote":
        return BorderlineSMOTE(random_state=random_seed)
    if strategy_name == "svm_smote":
        return SVMSMOTE(random_state=random_seed)
    if strategy_name == "kmeans_smote":
        return KMeansSMOTE(random_state=random_seed)
    if strategy_name == "adasyn":
        return ADASYN(random_state=random_seed)
    # RUS = random under-sampling.
    if strategy_name == "random_undersample":
        return RandomUnderSampler(random_state=random_seed)
    # SMOTE-STL is commonly implemented as SMOTE + Tomek Links.
    if strategy_name == "smote_tomek":
        return SMOTETomek(random_state=random_seed)
    if strategy_name == "smoteenn":
        return SMOTEENN(random_state=random_seed)
    raise KeyError(f"Unknown sampling strategy: {strategy_name}")


def summarize_resampling(y_before, y_after) -> dict[str, dict[str, int]]:
    return {
        "before": {str(k): int(v) for k, v in Counter(y_before).items()},
        "after": {str(k): int(v) for k, v in Counter(y_after).items()},
    }
