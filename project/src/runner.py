from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .data_loader import load_dataset, split_dataset
from .evaluation import (
    derive_resampling_summary,
    evaluate_on_test,
    fit_and_evaluate_candidate,
    run_cross_validation,
    save_classification_report,
)
from .models import get_model_specs
from .plotting import plot_confusion_matrix, plot_feature_importance, plot_precision_recall, plot_roc_curve
from .report import build_markdown_report
from .utils import ensure_dir, load_config, save_json, set_random_seed, setup_logging


def _rank_results(results_df: pd.DataFrame) -> pd.DataFrame:
    return results_df.sort_values(
        by=["val_f1", "val_recall", "val_pr_auc", "val_balanced_accuracy", "val_mcc"],
        ascending=False,
    ).reset_index(drop=True)


def _build_improvement_table(results_df: pd.DataFrame) -> pd.DataFrame:
    baseline_rows = results_df[results_df["sampling_strategy"] == "baseline"][["model_name", "val_f1", "val_recall", "val_mcc"]]
    if baseline_rows.empty:
        return pd.DataFrame()
    merged = results_df.merge(baseline_rows, on="model_name", suffixes=("", "_baseline"))
    merged["delta_val_f1"] = merged["val_f1"] - merged["val_f1_baseline"]
    merged["delta_val_recall"] = merged["val_recall"] - merged["val_recall_baseline"]
    merged["delta_val_mcc"] = merged["val_mcc"] - merged["val_mcc_baseline"]
    return merged[
        ["model_name", "sampling_strategy", "delta_val_f1", "delta_val_recall", "delta_val_mcc"]
    ].sort_values(by=["delta_val_f1", "delta_val_recall"], ascending=False)


def _build_strategy_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_pr_auc",
        "val_balanced_accuracy",
        "val_mcc",
    ]
    existing = [col for col in metric_columns if col in results_df.columns]
    if not existing:
        return pd.DataFrame()
    summary = (
        results_df.groupby("sampling_strategy", dropna=False)[existing]
        .mean()
        .reset_index()
        .sort_values(by=["val_f1", "val_recall", "val_accuracy"], ascending=False)
    )
    return summary


def _build_before_after_tables(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = [
        "model_name",
        "sampling_strategy",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_pr_auc",
        "val_balanced_accuracy",
        "val_mcc",
    ]
    existing = [col for col in metric_columns if col in results_df.columns]
    before_df = results_df[results_df["sampling_strategy"] == "baseline"][existing].copy()
    after_df = results_df[results_df["sampling_strategy"] != "baseline"][existing].copy()
    return before_df, after_df


def _build_paper_table_before(dataset_name: str, results_df: pd.DataFrame) -> pd.DataFrame:
    before_df = results_df[results_df["sampling_strategy"] == "baseline"].copy()
    if before_df.empty:
        return pd.DataFrame()

    table = before_df[
        [
            "model_name",
            "val_accuracy",
            "val_f1",
            "val_recall",
            "val_precision",
        ]
    ].copy()
    table.insert(0, "dataset", dataset_name)
    table = table.rename(
        columns={
            "model_name": "algorithm",
            "val_accuracy": "accuracy",
            "val_f1": "f1_score",
            "val_recall": "recall",
            "val_precision": "precision",
        }
    )
    return table.sort_values(by=["f1_score", "recall", "accuracy"], ascending=False).reset_index(drop=True)


def _build_paper_table_after(dataset_name: str, results_df: pd.DataFrame) -> pd.DataFrame:
    after_df = results_df[results_df["sampling_strategy"] != "baseline"].copy()
    if after_df.empty:
        return pd.DataFrame()

    table = after_df[
        [
            "sampling_strategy",
            "model_name",
            "val_accuracy",
            "val_f1",
            "val_recall",
            "val_precision",
        ]
    ].copy()
    table.insert(0, "dataset", dataset_name)
    table = table.rename(
        columns={
            "sampling_strategy": "technique",
            "model_name": "algorithm",
            "val_accuracy": "accuracy",
            "val_f1": "f1_score",
            "val_recall": "recall",
            "val_precision": "precision",
        }
    )
    return table.sort_values(by=["technique", "f1_score", "recall"], ascending=[True, False, False]).reset_index(drop=True)


def _log_dataset_stats(logger, dataset_name: str, stats: dict) -> None:
    logger.info("Dataset %s | rows=%s cols=%s", dataset_name, stats["rows"], stats["columns"])
    logger.info("Label distribution: %s", stats["label_distribution"])
    logger.info(
        "Imbalance ratio(minority/majority)=%.6f severity=%s",
        stats["imbalance_ratio_minority_to_majority"],
        stats["imbalance_severity"],
    )
    logger.info("Missing values total: %s", stats["missing_values_total"])
    logger.info("Numeric columns: %s | Categorical columns: %s", len(stats["numeric_columns"]), len(stats["categorical_columns"]))


def _save_feature_importance_if_available(best_pipeline, output_dir: Path) -> None:
    model = best_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
    from .preprocessing import get_feature_names

    feature_names = get_feature_names(best_pipeline.named_steps["preprocessor"])
    plot_feature_importance(
        feature_names=feature_names,
        importances=model.feature_importances_,
        title="Feature Importance",
        output_path=output_dir / "plots" / "best_feature_importance.png",
    )


def _save_best_artifacts(dataset_name: str, dataset_output_dir: Path, best_row: pd.Series, best_pipeline, y_test, test_scores, test_metrics, report_dict) -> None:
    ensure_dir(dataset_output_dir / "models")
    ensure_dir(dataset_output_dir / "reports")
    ensure_dir(dataset_output_dir / "plots")

    model_path = dataset_output_dir / "models" / "best_model.joblib"
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "threshold": float(best_row["threshold"]),
            "metadata": best_row.to_dict(),
        },
        model_path,
    )

    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        title=f"{dataset_name} Best Confusion Matrix",
        output_path=dataset_output_dir / "plots" / "best_confusion_matrix.png",
    )
    plot_precision_recall(
        y_test,
        test_scores,
        title=f"{dataset_name} Best Precision-Recall Curve",
        output_path=dataset_output_dir / "plots" / "best_precision_recall_curve.png",
    )
    plot_roc_curve(
        y_test,
        test_scores,
        title=f"{dataset_name} Best ROC Curve",
        output_path=dataset_output_dir / "plots" / "best_roc_curve.png",
    )
    save_classification_report(report_dict, dataset_output_dir / "reports" / "best_classification_report.json")
    _save_feature_importance_if_available(best_pipeline, dataset_output_dir)
    save_json(best_row.to_dict(), dataset_output_dir / "best_model_summary.json")


def run_benchmark_project(config_path: Path) -> None:
    project_root = config_path.parent
    config = load_config(config_path)
    output_root = ensure_dir(project_root / config["output_root"])
    logger = setup_logging(config.get("log_level", "INFO"), output_root / "run.log")
    set_random_seed(int(config["random_seed"]))

    all_rankings = []
    all_errors = []
    all_paper_before = []
    all_paper_after = []

    for dataset_name, dataset_cfg in config["datasets"].items():
        logger.info("=" * 90)
        logger.info("Starting dataset: %s", dataset_name)
        dataset_output_dir = ensure_dir(output_root / dataset_name)

        try:
            bundle = load_dataset(dataset_name, dataset_cfg, int(config["random_seed"]))
        except Exception as exc:
            logger.exception("Failed to load dataset %s", dataset_name)
            all_errors.append({"dataset_name": dataset_name, "stage": "load", "error": str(exc)})
            continue

        _log_dataset_stats(logger, dataset_name, bundle.stats)
        save_json(bundle.stats, dataset_output_dir / "dataset_stats.json")

        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
            bundle.features,
            bundle.target,
            test_size=float(config["test_size"]),
            validation_size=float(config["validation_size"]),
            random_seed=int(config["random_seed"]),
        )

        if y_train.nunique() < 2 or y_val.nunique() < 2 or y_test.nunique() < 2:
            logger.error(
                "Dataset %s does not contain both classes after split. train=%s val=%s test=%s",
                dataset_name,
                y_train.value_counts().to_dict(),
                y_val.value_counts().to_dict(),
                y_test.value_counts().to_dict(),
            )
            all_errors.append(
                {
                    "dataset_name": dataset_name,
                    "stage": "split",
                    "error": "One or more splits contain a single class only.",
                }
            )
            continue

        model_specs = get_model_specs(int(config["random_seed"]))
        selected_models = [name for name in config["models"] if name in model_specs]
        unavailable_models = [name for name in config["models"] if name not in model_specs]
        if unavailable_models:
            logger.warning("Skipped unavailable optional models: %s", unavailable_models)

        result_rows = []
        sampling_summaries = []

        for strategy_name in config["sampling_strategies"]:
            for model_name in selected_models:
                model_spec = model_specs[model_name]
                if model_spec.allowed_strategies is not None and strategy_name not in model_spec.allowed_strategies:
                    logger.info(
                        "Skipping dataset=%s model=%s strategy=%s because the model is restricted to %s",
                        dataset_name,
                        model_name,
                        strategy_name,
                        list(model_spec.allowed_strategies),
                    )
                    continue
                logger.info("Running dataset=%s model=%s strategy=%s", dataset_name, model_name, strategy_name)
                try:
                    candidate = fit_and_evaluate_candidate(
                        model_spec=model_spec,
                        strategy_name=strategy_name,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        random_seed=int(config["random_seed"]),
                        threshold_strategy=str(config["threshold_strategy"]),
                        recall_priority_target=float(config["recall_priority_target"]),
                    )
                    cv_summary = run_cross_validation(
                        candidate["pipeline"],
                        X_train,
                        y_train,
                        cv_folds=int(config["cv_folds"]),
                        random_seed=int(config["random_seed"]),
                    )
                    resampling_summary = derive_resampling_summary(candidate["pipeline"], X_train, y_train)
                    if resampling_summary is not None:
                        sampling_summaries.append(
                            {
                                "dataset_name": dataset_name,
                                "model_name": model_name,
                                "sampling_strategy": strategy_name,
                                "before_counts": resampling_summary["before"],
                                "after_counts": resampling_summary["after"],
                            }
                        )

                    row = {
                        "dataset_name": dataset_name,
                        "model_name": model_name,
                        "sampling_strategy": strategy_name,
                        "threshold": candidate["threshold"],
                        "fit_seconds": candidate["fit_seconds"],
                        "validation_seconds": candidate["score_seconds"],
                        **cv_summary,
                        **{f"val_{k}": v for k, v in candidate["validation_metrics"].items() if k != "confusion_matrix"},
                    }
                    result_rows.append(row)
                    logger.info(
                        "Done model=%s strategy=%s | val_f1=%.4f val_recall=%.4f val_mcc=%.4f",
                        model_name,
                        strategy_name,
                        row["val_f1"],
                        row["val_recall"],
                        row["val_mcc"],
                    )
                except Exception as exc:
                    logger.exception("Experiment failed for dataset=%s model=%s strategy=%s", dataset_name, model_name, strategy_name)
                    all_errors.append(
                        {
                            "dataset_name": dataset_name,
                            "model_name": model_name,
                            "sampling_strategy": strategy_name,
                            "stage": "train_or_validate",
                            "error": str(exc),
                        }
                    )

        if not result_rows:
            logger.warning("No successful experiments for dataset %s", dataset_name)
            continue

        results_df = pd.DataFrame(result_rows)
        ranked_df = _rank_results(results_df)
        improvement_df = _build_improvement_table(ranked_df)
        strategy_summary_df = _build_strategy_summary(ranked_df)
        before_df, after_df = _build_before_after_tables(ranked_df)
        paper_before_df = _build_paper_table_before(dataset_name, ranked_df)
        paper_after_df = _build_paper_table_after(dataset_name, ranked_df)

        ranked_df.to_csv(dataset_output_dir / "benchmark_results.csv", index=False)
        ranked_df.head(int(config["top_k_ranked"])).to_csv(dataset_output_dir / "top_ranked_configs.csv", index=False)
        improvement_df.to_csv(dataset_output_dir / "minority_improvement.csv", index=False)
        strategy_summary_df.to_csv(dataset_output_dir / "strategy_summary.csv", index=False)
        before_df.to_csv(dataset_output_dir / "before_imbalance_handling.csv", index=False)
        after_df.to_csv(dataset_output_dir / "after_imbalance_handling.csv", index=False)
        paper_before_df.to_csv(dataset_output_dir / "table_iii_before_handling.csv", index=False)
        paper_after_df.to_csv(dataset_output_dir / "table_iv_after_handling.csv", index=False)
        if sampling_summaries:
            pd.DataFrame(sampling_summaries).to_csv(dataset_output_dir / "resampling_summary.csv", index=False)

        best_row = ranked_df.iloc[0]
        best_candidate = fit_and_evaluate_candidate(
            model_spec=model_specs[str(best_row["model_name"])],
            strategy_name=str(best_row["sampling_strategy"]),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            random_seed=int(config["random_seed"]),
            threshold_strategy=str(config["threshold_strategy"]),
            recall_priority_target=float(config["recall_priority_target"]),
        )
        test_scores, test_metrics, report_dict = evaluate_on_test(
            best_candidate["pipeline"],
            X_test,
            y_test,
            threshold=float(best_row["threshold"]),
        )

        best_summary = pd.Series(
            {
                **best_row.to_dict(),
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_mcc": test_metrics["mcc"],
            }
        )

        _save_best_artifacts(dataset_name, dataset_output_dir, best_summary, best_candidate["pipeline"], y_test, test_scores, test_metrics, report_dict)
        build_markdown_report(
            dataset_name=dataset_name,
            dataset_stats=bundle.stats,
            ranked_df=ranked_df,
            best_summary=best_summary.to_dict(),
            improvement_df=improvement_df.head(10),
            output_path=dataset_output_dir / "report.md",
        )

        concise_ranking = ranked_df[
            [
                "dataset_name",
                "model_name",
                "sampling_strategy",
                "val_accuracy",
                "val_precision",
                "val_recall",
                "val_f1",
                "val_pr_auc",
                "val_balanced_accuracy",
                "val_mcc",
            ]
        ].copy()
        concise_ranking["best_test_f1"] = float(test_metrics["f1"])
        concise_ranking["best_test_mcc"] = float(test_metrics["mcc"])
        all_rankings.append(concise_ranking)
        if not paper_before_df.empty:
            all_paper_before.append(paper_before_df)
        if not paper_after_df.empty:
            all_paper_after.append(paper_after_df)
        logger.info("Top 5 for %s:\n%s", dataset_name, ranked_df.head(5).to_string(index=False))

    if all_rankings:
        pd.concat(all_rankings, ignore_index=True).to_csv(output_root / "all_datasets_rankings.csv", index=False)
    if all_paper_before:
        pd.concat(all_paper_before, ignore_index=True).to_csv(output_root / "table_iii_before_handling_all.csv", index=False)
    if all_paper_after:
        pd.concat(all_paper_after, ignore_index=True).to_csv(output_root / "table_iv_after_handling_all.csv", index=False)
    if all_errors:
        pd.DataFrame(all_errors).to_csv(output_root / "error_log.csv", index=False)

    logger.info("Benchmark run completed. Outputs saved under %s", output_root)
