#  Vendor Invoice Risk Intelligence System
## **Data Science Pipeline for Freight Cost Prediction & Automated Anomaly Flagging**

---

##  Why This Project is Required in the Real World (Problem Statement)

In global logistics and supply chain networks, **accounts payable teams process thousands of vendor invoices every single day.** Historically, auditing these bills for accuracy has been a massive operational challenge due to three critical real-world problems:

1. **Hidden Financial Leakage:** Overcharging by a few dollars on freight or slightly altering item totals across thousands of invoices adds up to **millions of dollars in lost corporate revenue** annually that goes completely unnoticed by standard accounting software.
2. **The "Human Review" Bottleneck:** It is physically impossible for human auditors to manually check every line item, purchase order, delivery date, and freight charge on every invoice. This leads to a dangerous tradeoff: teams either slow down business operations to check invoices, or they rush through and approve fraudulent or mistaken charges just to keep up.
3. **Flawed Rule-Based Systems:** Traditional accounting rule systems use basic, hardcoded thresholds (e.g., "flag if freight is over \$50"). Smart or fraudulent vendors easily bypass these rules by spreading out overcharges across multiple categories or keeping them just under the alert threshold.

###  The Solution: Data-Driven Machine Learning Guardrails
This project moves businesses away from manual guesswork and rigid thresholds toward a **predictive, automated audit engine**. By training machine learning models on historical transaction footprints, the system automatically builds mathematical boundaries of what a "normal" invoice looks like. It instantly isolates sophisticated overcharges, detects unnatural shipping delays, and flag risks for human verification—saving millions in revenue while accelerating standard auto-approvals [url].

---

## 📂 Project Structure & File Architecture

The repository is modularly structured into dedicated script files and interactive playgrounds, separating data pipeline mechanics, model training operations, and deployment functions:

###  Core Pipeline Scripts
*   **`data_preprocessing.py`**: Automated ETL script that connects to the database, extracts raw transaction ledgers, handles null calculations, and engineers clean structural features [url].
*   **`model_evaluation.py`**: Custom metrics library containing our background training algorithms and scoring functions. It evaluates model accuracy, errors, and standard deviations [url].
*   **`train_classification.py`**: The main orchestration training script. It loads the dataset, splits records, runs optimization parameters across classifiers, and saves the final production files [url].

###  Production & Deployment Scripts
*   **`app.py`**: The raw user dashboard interface script constructed to accept manual parameter updates and display immediate audit verdicts [url].
*   **`predict.py`**: A lightweight standalone prediction engine designed to parse single row queries dynamically [url].
*   **`inference_classification.py`**: The automated operational batch tool engineered to scan incoming monthly spreadsheet logs, execute vector predictions, and isolate high-risk targets [url].

###  Interactive Experiment Notebooks
*   **`Predicting Freight Cost.ipynb`**: The exploratory workbook tracking early stage data analysis, baseline price regression curves, and correlation visualizations [url].
*   **`Invoice_Flagging.ipynb`**: The advanced classification development notebook containing feature distribution analysis, Welch's T-Test logic, and model benchmarking comparisons [url].

---

##  Business Value & Objectives
*   **Prevent Financial Leakage:** Eradicates manual oversight loopholes and directly catches hidden vendor billing spikes or shipping overcharges [url].
*   **Automate Compliance Routing:** Accelerates transaction velocity by auto-approving low-risk invoices while cleanly routing high-risk anomalies to finance managers [url].
*   **Scalable Data Guardrails:** Replaces manual guesses with rigorous mathematical curves capable of evaluating thousands of lines of data instantly [url].

---

## ⚙️ Feature Engineering & Core Metrics
The models audit transactions across three foundational numeric clusters [url]:
*   **`invoice_quantity` / `invoice_dollars`**: Dimensions tracking total volumes and billing scale [url].
*   **`Freight`**: The calculated focus target metric isolating precise shipping overhead margins (`Dollars` - `Expected Cost`) [url].
*   **`days_po_to_invoice` / `avg_receiving_delay`**: Time lag markers derived via SQL `julianday` calculations to pinpoint rushed or irregular backend billing behavior [url, url].

---

## 🛠️ Technology Stack & Dependencies
*   **Core Language:** Python 
*   **Data Storage:** SQLite3 
*   **Processing & Analytics:** Pandas, NumPy, and Seaborn
*   **Machine Learning:** Scikit-Learn (Logistic Regression, Decision Trees, Random Forest Classifiers) [url]
*   **Statistical Auditing:** SciPy Stats (Two-Sample Welch's T-Test for anomaly variance) [url]

---

