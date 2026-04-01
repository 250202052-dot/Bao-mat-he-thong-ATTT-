# Benchmark Report - cic_unsw

## Dataset Overview
- Rows: 5,000
- Columns: 79
- Label distribution: {'0': 4873, '1': 127}
- Imbalance ratio (minority/majority): 0.026062
- Imbalance severity: severe
- Missing values: 0

## Best Configuration
- Model: `knn`
- Imbalance strategy: `adasyn`
- Validation F1: 0.7869
- Validation Recall: 0.9600
- Test F1: 0.6970
- Test Recall: 0.9200
- Test PR-AUC: 0.5747
- Test MCC: 0.7098
- Chosen threshold: 0.8571

## Top 5 Validation Configurations
```text
dataset_name    model_name sampling_strategy  threshold  fit_seconds  validation_seconds  cv_accuracy  cv_precision  cv_recall    cv_f1  cv_roc_auc  cv_pr_auc  cv_balanced_accuracy  val_accuracy  val_precision  val_recall   val_f1  val_roc_auc  val_pr_auc  val_balanced_accuracy  val_mcc
    cic_unsw           knn            adasyn   0.857143     0.076606            0.067102     0.978000      0.541212   0.987179 0.698718    0.984726   0.592929              0.982471         0.987       0.666667        0.96 0.786885     0.993026    0.653487               0.973846 0.794237
    cic_unsw random_forest          baseline   0.236000     1.015108            0.288876     0.979000      0.623656   0.467611 0.533597    0.990078   0.603075              0.730043         0.988       0.709677        0.88 0.785714     0.994872    0.803416               0.935385 0.784391
    cic_unsw random_forest       smote_tomek   0.476000     0.978428            0.126369     0.982667      0.624074   0.803981 0.701645    0.991879   0.650089              0.895661         0.988       0.709677        0.88 0.785714     0.993559    0.757764               0.935385 0.784391
    cic_unsw random_forest             smote   0.488000     1.047794            0.197980     0.983333      0.642857   0.778003 0.702778    0.992301   0.660470              0.883356         0.988       0.709677        0.88 0.785714     0.992882    0.702330               0.935385 0.784391
    cic_unsw random_forest            adasyn   0.532000     1.208292            0.179364     0.981667      0.617619   0.739204 0.672051    0.992297   0.658239              0.863615         0.988       0.709677        0.88 0.785714     0.993497    0.691887               0.935385 0.784391
```

## Minority-Class Improvement vs Baseline
```text
       model_name  sampling_strategy  delta_val_f1  delta_val_recall  delta_val_mcc
    decision_tree              smote      0.233470              0.32       0.230318
    decision_tree        smote_tomek      0.219141              0.32       0.215895
    decision_tree             adasyn      0.182250              0.24       0.178080
    decision_tree random_undersample      0.093635              0.48       0.124250
              knn             adasyn      0.091233              0.00       0.079069
              knn              smote      0.058446             -0.04       0.044686
              knn        smote_tomek      0.058446             -0.04       0.044686
gradient_boosting              smote      0.047919             -0.08       0.038204
gradient_boosting             adasyn      0.028689             -0.04       0.021964
gradient_boosting        smote_tomek      0.028689             -0.04       0.021964
```

## Interpretation
- Resampling often improves minority recall and F1, but can raise false positives.
- Class-weight is usually a safer choice when you need less aggressive changes to the class distribution.
- MCC and Balanced Accuracy are emphasized because they remain informative when raw Accuracy is misleading.
- For anomaly/attack detection, prefer the model that keeps recall high enough while controlling false alarms.