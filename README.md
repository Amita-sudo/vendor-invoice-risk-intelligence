#  Vendor Invoice Risk Intelligence System
## **Data Science Pipeline for Freight Cost Prediction & Automated Risk Auditing**

This repository contains the raw, production-ready machine learning backend built to automate corporate invoice compliance auditing. The system uses historical enterprise transaction logs to predict baseline shipping costs and classify incoming vendor invoices into automated approval workflows or flags them for high-risk manual review.

---

## Project Overview & Architecture
Instead of relying on rigid, hardcoded rules or subjective manual verification, this project implements a complete, data-driven machine learning script pipeline:

1. **`data_preprocessing.py`**: Automated ETL script that connects to the `inventory.db` SQL database, processes raw purchase ledger matrices, calculates operational billing lags, and engineers clean numeric features.
2. **`model_evaluation.py`**: Library containing our core training algorithms. It trains both price forecasting models (Linear Regression, Decision Tree, Random Forest Regressors) and binary classification models to detect billing variances.
3. **`train_classification.py`**: The main orchestration pipeline script that loads fresh invoice records, splits datasets, evaluates system accuracy, and freezes the top-performing model weights directly to a serialized file (`.pkl`) for production deployment.

---

##  Business Value & Objectives
* **Prevent Financial Leakage:** Eradicates manual oversight loopholes and directly catches hidden vendor billing spikes or shipping overcharges.
* **Automate Compliance Routing:** Accelerates transaction velocity by auto-approving low-risk invoices while cleanly routing high-risk anomalies to finance managers.
* **Scalable Data Guardrails:** Replaces manual guesses with rigorous mathematical curves capable of evaluating thousands of lines of data instantly.

---

##  Feature Engineering & Core Metrics
The models audit transactions across three foundational numeric clusters:
* **`invoice_quantity` / `invoice_dollars`**: Dimensions tracking total volumes and billing scale.
* **`Freight` / `freight_per_unit`**: The target focus metric isolating precise shipping overhead margins.
* **`days_po_to_invoice` / `avg_receiving_delay`**: Time lag markers derived via SQL `julianday` calculations to pinpoint rushed or irregular backend billing behavior.

---

## 🛠️ Technology Stack & Dependencies
* **Core Language:** Python 
* **Data Storage:** SQLite3 (SQL Ledger Engine)
* **Processing & Analytics:** Pandas, NumPy, and Seaborn
* **Machine Learning:** Scikit-Learn (Logistic Regression, Decision Trees, Random Forest Classifiers)
* **Statistical Auditing:** SciPy Stats (Two-Sample Welch's T-Test for anomaly variance)

---

## 🚀 How to Run This Engine Locally

1. Clone this repository to your local terminal path:
   ```bash
   git clone https://github.com
   cd vendor-invoice-risk-intelligence
   ```

2. Install the necessary machine learning libraries:
   ```bash
   pip install pandas numpy scikit-learn scipy
   ```

3. Execute the automated data extraction and model deployment script:
   ```bash
   python train_classification.py
   ```
