# IDS Benchmark Project: Step-by-Step Guide

This guide is written for someone who has never used this project before and wants to:

1. run the benchmark pipeline as-is
2. understand what the project does
3. run the same pipeline again with a different dataset

The main entrypoint of the project is [main.py](E:\Đồ án thạc sĩ\An toàn bảo mật hệ thống ATTT\project\main.py), and it reads settings from [paper_style_config.yaml](E:\Đồ án thạc sĩ\An toàn bảo mật hệ thống ATTT\project\paper_style_config.yaml).

## 1. What This Project Does

The project benchmarks multiple machine learning models and multiple imbalance-handling techniques on intrusion-detection datasets.

For each dataset, it:

1. loads the CSV file
2. finds the label column
3. converts the labels into binary classes:
   - `0 = benign`
   - `1 = attack`
4. drops columns you do not want to train on
5. splits the data into:
   - train
   - validation
   - test
6. trains multiple models with multiple sampling strategies
7. ranks the results
8. saves reports, tables, plots, and the best trained model

## 2. Project Structure

Important files:

- [main.py](E:\Đồ án thạc sĩ\An toàn bảo mật hệ thống ATTT\project\main.py): project entrypoint
- [paper_style_config.yaml](E:\Đồ án thạc sĩ\An toàn bảo mật hệ thống ATTT\project\paper_style_config.yaml): full benchmark configuration
- [requirements.txt](E:\Đồ án thạc sĩ\An toàn bảo mật hệ thống ATTT\project\requirements.txt): Python packages
- [src/runner.py](E:\Đồ án thạc sĩ\An toàn bảo mật hệ thống ATTT\project\src\runner.py): full benchmark pipeline
- [src/data_loader.py](E:\Đồ án thạc sĩ\An toàn bảo mật hệ thống ATTT\project\src\data_loader.py): dataset loading and label binarization
- [src/preprocessing.py](E:\Đồ án thạc sĩ\An toàn bảo mật hệ thống ATTT\project\src\preprocessing.py): preprocessing pipeline

Main outputs after a run:

- [paper_style_outputs](E:\Đồ án thạc sĩ\An toàn bảo mật hệ thống ATTT\project\paper_style_outputs)
- per-dataset subfolders such as:
  - `paper_style_outputs/cic_unsw`
  - `paper_style_outputs/cic_iomt_2024`

## 3. Prerequisites

You need:

- Python 3.10+ recommended
- a CSV dataset file
- enough RAM for the models and dataset size you choose

## 4. Create a Python Environment

From the `project` folder:

```powershell
cd "E:\Đồ án thạc sĩ\An toàn bảo mật hệ thống ATTT\project"
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r .\requirements.txt
```

If you already have `.venv_clean`, you can use that instead.

## 5. Run the Default Pipeline

This runs the benchmark exactly as defined in [paper_style_config.yaml](E:\Đồ án thạc sĩ\An toàn bảo mật hệ thống ATTT\project\paper_style_config.yaml):

```powershell
cd "E:\Đồ án thạc sĩ\An toàn bảo mật hệ thống ATTT\project"
.\.venv\Scripts\python.exe .\main.py --config .\paper_style_config.yaml
```

If you are using `.venv_clean`:

```powershell
cd "E:\Đồ án thạc sĩ\An toàn bảo mật hệ thống ATTT\project"
..\.venv_clean\Scripts\python.exe .\main.py --config .\paper_style_config.yaml
```

## 6. What Happens During the Run

The pipeline will loop through all datasets listed under `datasets:` in the config.

For each dataset it will try combinations of:

- sampling strategies:
  - `baseline`
  - `smote`
  - `adasyn`
  - `random_undersample`
  - `smote_tomek`
- models:
  - `logistic_regression`
  - `decision_tree`
  - `random_forest`
  - `knn`
  - `gradient_boosting`

It ranks results mainly by:

- `val_f1`
- `val_recall`
- `val_pr_auc`
- `val_balanced_accuracy`
- `val_mcc`

## 7. Where to Find the Results

After the run, check:

- `paper_style_outputs/<dataset_name>/benchmark_results.csv`
- `paper_style_outputs/<dataset_name>/top_ranked_configs.csv`
- `paper_style_outputs/<dataset_name>/best_model_summary.json`
- `paper_style_outputs/<dataset_name>/models/best_model.joblib`
- `paper_style_outputs/<dataset_name>/report.md`

Also check the aggregated outputs:

- `paper_style_outputs/all_datasets_rankings.csv`
- `paper_style_outputs/table_iii_before_handling_all.csv`
- `paper_style_outputs/table_iv_after_handling_all.csv`

## 8. How Labels Are Converted to Benign vs Attack

This project turns labels into binary classes in [src/data_loader.py](E:\Đồ án thạc sĩ\An toàn bảo mật hệ thống ATTT\project\src\data_loader.py).

Important behaviour:

- If labels are already numeric and binary, it keeps them as binary.
- If labels are numeric with more than two values:
  - `0` becomes benign
  - non-zero becomes attack
- If labels are text:
  - labels like `benign`, `normal`, `safe` become class `0`
  - everything else becomes class `1`

So before adding a new dataset, you should verify that this binary mapping makes sense for your dataset.

## 9. How to Run the Project with a New Dataset

### Step 1. Prepare the CSV

Your new dataset should be in CSV format.

Check:

- what the label column is called
- whether labels are numeric or text
- which columns should not be used for training
- whether the dataset is too large to test quickly

### Step 2. Open the config file

Edit [paper_style_config.yaml](E:\Đồ án thạc sĩ\An toàn bảo mật hệ thống ATTT\project\paper_style_config.yaml).

Inside the `datasets:` section, add a new block.

Example:

```yaml
datasets:
  my_new_dataset:
    path: "D:\\Datasets\\my_new_dataset.csv"
    label_column: "Label"
    positive_class: 1
    drop_columns:
      - "Flow ID"
      - "Src IP"
      - "Dst IP"
      - "Timestamp"
    max_rows: 5000
    sample_frac:
```

### Step 3. Set the fields correctly

Meaning of each dataset field:

- `path`: absolute path to the CSV file
- `label_column`: column containing the class label
- `positive_class`: which value should be treated as attack
- `drop_columns`: columns to remove before training
- `max_rows`: optional row cap for faster experiments
- `sample_frac`: optional fractional sample instead of `max_rows`

### Step 4. Choose `positive_class` correctly

Examples:

- If your dataset uses `0` for benign and `1` for attack:
  - use `positive_class: 1`
- If your dataset uses text labels like `Benign`, `DoS`, `Reconnaissance`:
  - you can leave `positive_class` as empty and let the loader binarize text labels
- If your dataset has one specific attack label you want as positive:
  - set that exact value

### Step 5. Choose `drop_columns`

Typical columns to drop:

- identifiers:
  - `Flow ID`
  - `ID`
- IP-related fields if you do not want leakage:
  - `Src IP`
  - `Dst IP`
- timestamps:
  - `Timestamp`
- metadata not intended for training:
  - `Attack Name`
  - source file tracking fields

Do not drop columns blindly. If a column contains real traffic features, keep it.

### Step 6. Start small first

For a new dataset, first run with:

```yaml
max_rows: 5000
```

This lets you verify:

- the file loads correctly
- label mapping is correct
- no preprocessing errors happen
- the outputs are generated as expected

Later, if everything works, you can change:

```yaml
max_rows:
```

to use the full dataset.

## 10. Run the Pipeline Again with the New Dataset

After saving the config:

```powershell
cd "E:\Đồ án thạc sĩ\An toàn bảo mật hệ thống ATTT\project"
.\.venv\Scripts\python.exe .\main.py --config .\paper_style_config.yaml
```

If you keep old datasets in the config, the pipeline will run them too.

If you want to benchmark only the new dataset, temporarily remove or comment out the old dataset blocks and leave only the new one.

## 11. Example: Run Only One New Dataset

Suppose you want only:

```yaml
datasets:
  my_new_dataset:
    path: "D:\\Datasets\\my_new_dataset.csv"
    label_column: "Label"
    positive_class: 1
    drop_columns:
      - "Flow ID"
      - "Src IP"
      - "Dst IP"
      - "Timestamp"
    max_rows: 5000
    sample_frac:
```

Then run:

```powershell
cd "E:\Đồ án thạc sĩ\An toàn bảo mật hệ thống ATTT\project"
.\.venv\Scripts\python.exe .\main.py --config .\paper_style_config.yaml
```

Outputs will appear in:

- `paper_style_outputs/my_new_dataset`

## 12. Common Problems

### Dataset file not found

Error like:

```text
FileNotFoundError: Dataset not found
```

Fix:

- make sure `path:` in the YAML is correct
- prefer an absolute path

### Label column not found

Error like:

```text
KeyError: Label column 'Label' not found
```

Fix:

- open the CSV
- confirm the exact column name
- update `label_column:` in YAML

### One split contains only one class

This means the dataset or sample is too small or too imbalanced after sampling.

Fix:

- increase `max_rows`
- use the full dataset
- verify the label distribution

### Too slow

Fix:

- reduce `max_rows`
- reduce the number of models in `models:`
- reduce the number of strategies in `sampling_strategies:`
- reduce `cv_folds`

## 13. Recommended Workflow for a New Dataset

Use this order:

1. add the new dataset to the YAML
2. keep `max_rows: 5000`
3. run the pipeline once
4. inspect:
   - `dataset_stats.json`
   - `benchmark_results.csv`
   - `best_model_summary.json`
5. confirm label mapping is correct
6. confirm dropped columns are correct
7. rerun with the full dataset if needed

## 14. Quick Command Summary

Create environment:

```powershell
cd "E:\Đồ án thạc sĩ\An toàn bảo mật hệ thống ATTT\project"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r .\requirements.txt
```

Run benchmark:

```powershell
.\.venv\Scripts\python.exe .\main.py --config .\paper_style_config.yaml
```

Check outputs:

```powershell
dir .\paper_style_outputs
```

## 15. Final Notes

For a new dataset, the two most important things to verify are:

1. the label mapping is turning the dataset into the correct binary problem
2. the dropped columns are not accidentally removing useful features or keeping leakage columns

If you get those two right, the rest of the pipeline should work smoothly for most CSV intrusion-detection datasets.
