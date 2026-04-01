from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_markdown_report(
    dataset_name: str,
    dataset_stats: dict,
    ranked_df: pd.DataFrame,
    best_summary: dict,
    improvement_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        f"# Benchmark Report - {dataset_name}",
        "",
        "## Dataset Overview",
        f"- Rows: {dataset_stats['rows']:,}",
        f"- Columns: {dataset_stats['columns']:,}",
        f"- Label distribution: {dataset_stats['label_distribution']}",
        f"- Imbalance ratio (minority/majority): {dataset_stats['imbalance_ratio_minority_to_majority']:.6f}",
        f"- Imbalance severity: {dataset_stats['imbalance_severity']}",
        f"- Missing values: {dataset_stats['missing_values_total']:,}",
        "",
        "## Best Configuration",
        f"- Model: `{best_summary['model_name']}`",
        f"- Imbalance strategy: `{best_summary['sampling_strategy']}`",
        f"- Validation F1: {best_summary['val_f1']:.4f}",
        f"- Validation Recall: {best_summary['val_recall']:.4f}",
        f"- Test F1: {best_summary['test_f1']:.4f}",
        f"- Test Recall: {best_summary['test_recall']:.4f}",
        f"- Test PR-AUC: {best_summary['test_pr_auc']:.4f}",
        f"- Test MCC: {best_summary['test_mcc']:.4f}",
        f"- Chosen threshold: {best_summary['threshold']:.4f}",
        "",
        "## Top 5 Validation Configurations",
        "```text",
        ranked_df.head(5).to_string(index=False),
        "```",
        "",
        "## Minority-Class Improvement vs Baseline",
    ]

    if not improvement_df.empty:
        lines.extend(["```text", improvement_df.to_string(index=False), "```"])
    else:
        lines.append("No improvement table available.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "- Resampling often improves minority recall and F1, but can raise false positives.",
            "- Class-weight is usually a safer choice when you need less aggressive changes to the class distribution.",
            "- MCC and Balanced Accuracy are emphasized because they remain informative when raw Accuracy is misleading.",
            "- For anomaly/attack detection, prefer the model that keeps recall high enough while controlling false alarms.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
