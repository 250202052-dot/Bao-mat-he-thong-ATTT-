# Benchmark Report - cic_iomt_2024

## Dataset Overview
- Rows: 5,000
- Columns: 79
- Label distribution: {'0': 48, '1': 4952}
- Imbalance ratio (minority/majority): 0.009693
- Imbalance severity: extreme
- Missing values: 0

## Best Configuration
- Model: `random_forest`
- Imbalance strategy: `baseline`
- Validation F1: 0.9990
- Validation Recall: 1.0000
- Test F1: 0.9960
- Test Recall: 0.9970
- Test PR-AUC: 1.0000
- Test MCC: 0.5551
- Chosen threshold: 0.3440

## Top 5 Validation Configurations
```text
 dataset_name        model_name sampling_strategy  threshold  fit_seconds  validation_seconds  cv_accuracy  cv_precision  cv_recall    cv_f1  cv_roc_auc  cv_pr_auc  cv_balanced_accuracy  val_accuracy  val_precision  val_recall   val_f1  val_roc_auc  val_pr_auc  val_balanced_accuracy  val_mcc
cic_iomt_2024     random_forest          baseline   0.344000     0.822509            0.169837     0.993333      0.995637   0.997645 0.996639    0.978165   0.999618              0.766679         0.998       0.997984         1.0 0.998991     0.999596    0.999996                    0.9 0.893525
cic_iomt_2024     random_forest             smote   0.404000     0.769694            0.097187     0.994000      0.996972   0.996972 0.996972    0.978418   0.999617              0.837772         0.998       0.997984         1.0 0.998991     0.999394    0.999994                    0.9 0.893525
cic_iomt_2024     random_forest            adasyn   0.248000     1.069748            0.147545     0.994000      0.996638   0.997308 0.996972    0.977733   0.999610              0.820083         0.998       0.997984         1.0 0.998991     0.999394    0.999994                    0.9 0.893525
cic_iomt_2024     random_forest       smote_tomek   0.404000     1.279735            0.129138     0.994000      0.996638   0.997308 0.996972    0.978346   0.999617              0.820083         0.998       0.997984         1.0 0.998991     0.999394    0.999994                    0.9 0.893525
cic_iomt_2024 gradient_boosting          baseline   0.007072     1.814446            0.024047     0.993000      0.995966   0.996972 0.996469    0.937969   0.998869              0.784200         0.998       0.997984         1.0 0.998991     0.999293    0.999993                    0.9 0.893525
```

## Minority-Class Improvement vs Baseline
```text
       model_name sampling_strategy  delta_val_f1  delta_val_recall  delta_val_mcc
    decision_tree            adasyn      0.000004           0.00202      -0.024943
              knn             smote      0.000003           0.00202      -0.033944
              knn       smote_tomek      0.000003           0.00202      -0.033944
              knn            adasyn      0.000003           0.00202      -0.033944
    random_forest          baseline      0.000000           0.00000       0.000000
    random_forest             smote      0.000000           0.00000       0.000000
    random_forest            adasyn      0.000000           0.00000       0.000000
    random_forest       smote_tomek      0.000000           0.00000       0.000000
gradient_boosting          baseline      0.000000           0.00000       0.000000
gradient_boosting             smote      0.000000           0.00000       0.000000
```

## Interpretation
- Resampling often improves minority recall and F1, but can raise false positives.
- Class-weight is usually a safer choice when you need less aggressive changes to the class distribution.
- MCC and Balanced Accuracy are emphasized because they remain informative when raw Accuracy is misleading.
- For anomaly/attack detection, prefer the model that keeps recall high enough while controlling false alarms.