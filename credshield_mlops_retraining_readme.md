# 🛡️ CredShield — Banking Credit Scoring & Fraud Detection MLOps
### MLOps Practical Assessment (MSE-2): Model Quality Monitoring & Retraining Workflow

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2.svg)](https://mlflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipelines-F7931E.svg)](https://scikit-learn.org/)

* **Student Name:** Vikas Soni 
* **Roll Number:** 2400290100346  
* **Course / Year:** B.Tech, 3rd Year CSE  
* **Repository:** [github.com/vikas0327/Credit-Fraud-MLOps](https://github.com/vikas0327/Credit-Fraud-MLOps)  

---

## 📋 Overview & Objective

In production machine learning systems, model performance can degrade over time as new data arrives. The objective of this project is to extend the existing **CredShield** banking ML system (comprising **Credit Scoring** and **Fraud Detection** models) with an automated **model-quality monitoring, candidate retraining, and governance (quality-gate) workflow**.

Rather than blindly overwriting deployed models when new data arrives, this pipeline validates incoming batches, benchmarks the active **Champion** models, trains isolated **Candidate** models on combined historical + new data, and enforces a strict quality gate tracked via **MLflow**.

---

## 🏗️ System Architecture & Components

| Component | Details / Implementation |
| :--- | :--- |
| **Credit Scoring Model** | `RandomForest` Classifier (Registered Champion) |
| **Fraud Detection Model** | `LogisticRegression` Classifier (Registered Champion) |
| **Monitoring & Retraining Script** | `src/monitor_retrain.py` |
| **Experiment & Registry Tracking** | MLflow Tracking Server & Model Registry |
| **Champion Artifacts** | `artifacts/credit.joblib`, `artifacts/fraud.joblib` |
| **Candidate Artifacts** | `artifacts/candidates/credit_candidate.joblib`, `artifacts/candidates/fraud_candidate.joblib` |
| **Evaluation Reports** | `reports/credit_retraining_report.json`, `reports/fraud_retraining_report.json` |
| **Governance Policy** | Strict non-destructive promotion; champions are never overwritten automatically |

---

## ⚙️ End-to-End Workflow (Tasks A – E)

### 🔹 Task A — Prepare & Validate New Data
When a new labelled batch arrives, the workflow runs automated data-quality checks before touching any model:
* **Schema & Missing Values:** Verifies required feature columns match the historical dataset and checks for `NaN`/null entries.
* **Duplicate & ID Integrity:** Checks for duplicate records across the batch and ensures transaction `id` values in the fraud batch do not overlap with historical records.
* **Dataset Summary:** Logs batch dimensions and class distributions.

| Workflow | Incoming Batch Shape | Validation Status | Target Class Distribution |
| :--- | :--- | :--- | :--- |
| **Credit Scoring** | `80 rows × 10 columns` | ✅ Passed (`0` missing, `0` duplicates) | Class `1` (Bad Risk): **40** \| Class `0` (Good Risk): **40** |
| **Fraud Detection** | `80 rows × 31 columns` | ✅ Passed (`0` missing, `0` duplicates, unique IDs) | Class `0` (Legit): **68** \| Class `1` (Fraud): **12** |

---

### 🔹 Task B — Evaluate the Existing Champion Model
The registered **Champion** models are loaded from the MLflow Model Registry and evaluated against the full incoming labelled batch to establish a baseline on new data.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Credit Champion** *(Incoming Batch)* | `0.6625` | `0.7600` | `0.4750` | `0.5846` |
| **Fraud Champion** *(Incoming Batch)* | `0.7250` | `0.2222` | `0.3333` | `0.2667` |

---

### 🔹 Task C — Retrain Candidate Model
To ensure a leakage-free, apples-to-apples comparison:
1. **Stratified Holdout Split:** The incoming batch is split **75% for retraining** and **25% as a clean holdout set**.
2. **Data Combination:** Historical training data is merged with the 75% incoming training split.
3. **Pipeline Cloning & Fitting:** The Champion's preprocessing and estimator pipeline is cloned and fitted on the combined dataset.
4. **Isolated Persistence & Tracking:** Candidate models are saved separately under `artifacts/candidates/` and logged to dedicated MLflow experiments (`credit-monitoring-retraining` and `fraud-monitoring-retraining`).

---

### 🔹 Task D — Quality Gate & Metric Justification

#### 🎯 Metric Selection Justification: **F1-Score & Accuracy**
* **Why F1-Score?** Both **Fraud Detection** (heavily imbalanced, where missing fraud is costly and false alarms frustrate customers) and **Credit Risk Scoring** (where false approvals lead to financial default and false rejections lose business) require balancing **Precision** and **Recall**. F1-score penalizes extreme trade-offs between the two.
* **Why Accuracy alongside F1-Score?** Including Accuracy as a co-requirement ensures overall classification stability across both majority and minority classes, preventing a candidate from gaming F1-score at the expense of broad prediction degradation.

#### 🚦 Decision Rule & Holdout Comparison
A candidate is marked **ELIGIBLE FOR PROMOTION** if and only if on the exact same **25% incoming holdout set**:
$$\text{Candidate F1-Score} \ge \text{Champion F1-Score} \quad \text{AND} \quad \text{Candidate Accuracy} \ge \text{Champion Accuracy}$$

| Workflow | Holdout Metric | Champion Score | Candidate Score | Quality Gate Result | Governance Action |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Credit Scoring** | Accuracy<br>F1-Score | `0.7500`<br>`0.6667` | `0.5500`<br>`0.4706` | ❌ **FAIL** | Retain `RandomForest` Champion; do not promote candidate. |
| **Fraud Detection** | Accuracy<br>F1-Score | `0.8000`<br>`0.5000` | `0.6000`<br>`0.3333` | ❌ **FAIL** | Retain `LogisticRegression` Champion; do not promote candidate. |

> **Note:** A `FAIL` decision demonstrates that the quality gate is functioning properly—preventing underperforming candidates trained on noisy/synthetic shifts from degrading production service.

---

### 🔹 Task E — Execution Guide & Generated Artifacts

#### 🚀 Running the Workflow
Execute the monitoring and retraining pipeline from the repository root:
```bash
python -m src.monitor_retrain
```
Upon successful completion, the console outputs:
```text
ALL MONITORING AND RETRAINING WORKFLOWS COMPLETED.
```

#### 📊 Launching MLflow UI
To inspect logged parameters, holdout metrics, quality-gate decisions, and saved artifacts:
```bash
python -m mlflow ui --host 0.0.0.0 --port 5000
```
Open `http://localhost:5000` (or forward port `5000` in GitHub Codespaces).

#### 📁 Output Artifacts Summary
* `artifacts/candidates/credit_candidate.joblib` — Separately persisted Credit candidate pipeline.
* `artifacts/candidates/fraud_candidate.joblib` — Separately persisted Fraud candidate pipeline.
* `reports/credit_retraining_report.json` — Detailed JSON audit report for Credit scoring.
* `reports/fraud_retraining_report.json` — Detailed JSON audit report for Fraud detection.
* `artifacts/credit.joblib` & `artifacts/fraud.joblib` — Active Champions (preserved untouched).

---

## ⚠️ Scope & Limitations
* **Synthetic Demonstration Batches:** The incoming batches used in this assessment were generated as synthetic test batches to validate pipeline mechanics and model governance; they do not represent live banking production data.
* **CI/CD Separation:** This workflow executes locally/in Codespaces and logs to MLflow independently of the repository's GitHub Actions CI YAML workflow.