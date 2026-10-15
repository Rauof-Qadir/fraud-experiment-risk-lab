# Fraud Experiment & Risk Statistics Lab

A reproducible fraud-detection experimentation project that compares rule-based transaction screening strategies using cost-sensitive evaluation, chronological validation, statistical analysis, and 
automated tests.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/Pandas-Data_Analysis-150458?logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/SciPy-Statistics-8CAAE6?logo=scipy&logoColor=white" alt="SciPy"/>
  <img src="https://img.shields.io/badge/Testing-Pytest-0A9EDC?logo=pytest&logoColor=white" alt="Pytest"/>
  <img src="https://img.shields.io/badge/Project-Research-orange" alt="Research Project"/>
</p>


## Project Overview

Fraud detection is an imbalanced classification problem: fraudulent transactions are rare, while incorrectly flagging legitimate transactions can create operational costs.

This project investigates how different rule-based strategies affect:

- Fraud detection recall
- Precision and false-positive rate
- Confusion matrix metrics
- Estimated financial cost
- Threshold selection under different cost assumptions
- Performance on a future chronological test period

**Important:** This is an experimental analytics project, not a production-ready banking fraud system.

## Key Results

The selected threshold was chosen using validation data and evaluated on a separate, later test period.

| Metric | Baseline rule | Selected threshold rule |
|---|---:|---:|
| Threshold | No amount threshold | 750,000 |
| Test transactions | 89,466 | 89,466 |
| True positives | 1,252 | 472 |
| False positives | 36,727 | 1,983 |
| False negatives | 0 | 780 |
| Precision | 3.30% | 19.23% |
| Recall | 100.00% | 37.70% |
| Estimated cost | $183,635 | $126,915 |

### Main Findings

- Estimated test-set cost decreased by approximately **30.9%** under the chosen cost assumptions.
- False positives decreased by approximately **94.6%**.
- Precision improved from approximately 3.30% to 19.23%.
- Recall declined to approximately 37.70%, meaning the selected rule missed many fraud cases.
- The cluster-bootstrap 95% confidence interval for estimated savings was approximately **−$10,156 to $132,951**, including zero.
- The results therefore require additional validation across different time periods.

These results demonstrate a trade-off between operational workload and fraud detection. The selected threshold should not be treated as universally optimal.

## Methodology

### 1. Exploratory Data Analysis

Investigate transaction types, fraud prevalence, class imbalance, and data quality.

### 2. Baseline Rule

Flag transactions of type `TRANSFER` or `CASH_OUT` as suspicious.

### 3. Threshold Rule

Flag transactions when both conditions are met:

- Transaction type is `TRANSFER` or `CASH_OUT`.
- Transaction amount is at least 750,000.

The threshold is experimental and is not an industry-standard value.

### 4. Cost-Sensitive Evaluation

Compare rules using illustrative costs:

- False positive (FP): $5
- False negative (FN): $150

Estimated cost:

`Total Cost = (FP × FP Cost) + (FN × FN Cost)`

These costs are assumptions for experimentation, not verified costs from a financial institution.

### 5. Chronological Validation

Sort unique transaction time steps and divide them into chronological subsets:

- Training period: first 70% of unique steps
- Validation period: next 15%
- Test period: final 15%

Select the threshold using validation data, then evaluate the selected rule on the held-out test period.

### 6. Statistical Analysis

Use a paired McNemar exact test to compare classification-error disagreements and a time-step cluster bootstrap to estimate uncertainty in cost savings.

The statistical methods rely on assumptions, including how observations depend on one another over time. Their results should be interpreted alongside the dataset's temporal drift and the limited number of test-period time steps.

### 7. Automated Testing

Use `pytest` to check rule behavior, including threshold boundaries, transaction types, invalid amounts, and numeric strings.

## Project Structure

```text
fraud-experiment-risk-lab/
├── data/
│   └── raw/
│       └── transactions.csv
├── reports/
│   ├── advanced_metrics_comparison.csv
│   ├── baseline_confusion_matrix.png
│   ├── new_rule_confusion_matrix.png
│   ├── threshold_sensitivity.csv
│   ├── validation_threshold_results.csv
│   ├── final_test_results.csv
│   └── statistical_comparison.csv
├── src/
│   ├── data_loader.py
│   ├── data_audit.py
│   ├── eda.py
│   ├── baseline.py
│   ├── baseline_analysis.py
│   ├── cost_analysis.py
│   ├── new_rule.py
│   ├── compare_rules.py
│   ├── confusion_analysis.py
│   ├── advanced_metrics.py
│   ├── threshold_analysis.py
│   ├── cost_sensitivity.py
│   ├── experiment_split.py
│   ├── validation_experiment.py
│   └── statistical_comparison.py
├── tests/
│   └── test_statistical_rules.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Tech Stack
```
Python • Pandas • NumPy • SciPy • Matplotlib • Seaborn • scikit-learn • pytest

```



## Installation

Python 3.11 or a compatible version is recommended.

Clone the repository:

```bash
git clone <https://github.com/Rauof-Qadir/fraud-experiment-risk-lab>
cd fraud-experiment-risk-lab
```

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the dataset at:

```text
data/raw/transactions.csv
```

Make sure you have permission to redistribute the dataset before publishing it.

## Running the Experiments

Run the chronological validation and test evaluation:

```bash
python -m src.validation_experiment
```

Run the statistical comparison:

```bash
python -m src.statistical_comparison
```

Run automated tests:

```bash
python -m pytest tests/ -v
```

Some exploratory scripts may take additional time because the dataset contains millions of transactions.

## Reports

The `reports/` directory contains experiment outputs, including metric comparisons, confusion matrices, threshold sensitivity results, validation results, and statistical comparisons.

Generated reports should be reproducible from the documented code and data. Verify that each report is current before publishing it.

## Limitations

- The analysis uses a single dataset and one chronological test window.
- Fraud prevalence changes substantially across the chronological splits, suggesting potential temporal drift.
- Rule-based screening is limited compared with richer machine-learning models and contextual risk signals.
- Cost estimates depend on hypothetical FP and FN cost assumptions.
- Statistical significance does not automatically establish business value or deployment readiness.
- Passing unit tests verifies tested software behavior, not the accuracy of fraud labels or the general effectiveness of the rule.
- No live banking integration or real-time production deployment is included.

## Future Improvements

- Evaluate performance across multiple chronological test windows.
- Add precision-recall curves and threshold calibration analysis.
- Compare rules with logistic regression and tree-based models.
- Track experiment configurations and results.
- Add data validation and automated report generation.
- Investigate drift monitoring and alert-volume constraints.
- Add model explainability and a reproducible experiment dashboard.

## Disclaimer

This project is for educational and research purposes. The selected threshold and estimated costs are experimental assumptions, not financial advice or a validated fraud policy.

## Author

**Rauof Qadir**

Software Engineering Graduate | Data Science | Financial Risk Analytics

GitHub: <https://github.com/Rauof-Qadir>

LinkedIn: <linkedin.com/in/rauof-qadir-a68a25338>