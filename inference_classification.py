import sys
import pickle
import pandas as pd
from pathlib import Path

def run_classification_inference(csv_data_input_path: str):
    """
    Loads the saved model file and runs a live scan to detect overcharges.
    """
    model_path = Path("/content/drive/MyDrive/Freight_Cost_Project/models/best_invoice_classifier.pkl")
    if not model_path.exists():
        print("❌ Error: The model file does not exist yet. Run train_classification.py first!")
        return

    # 1. Load the frozen model weights back into live system memory
    with open(model_path, "rb") as f:
        deployed_classifier = pickle.load(f)

    # 2. Extract new incoming invoice batch sheets
    df = pd.read_csv(csv_data_input_path)
    
    # 3. Pull required features
    X_new = df[['Quantity', 'Dollars', 'Freight']]

    # 4. INFERENCING RUN: Make predictions on unseen live rows
    df["Risk_Flag"] = deployed_classifier.predict(X_new)

    # 5. Save the flagged results file for the accounts payable team
    output_report = "/content/drive/MyDrive/Freight_Cost_Project/flagged_audit_report.csv"
    df[df["Risk_Flag"] == 1].to_csv(output_report, index=False)

    print(f"🚀 Inferencing complete! Flagged spreadsheet exported to:\n{output_report}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_classification_inference(sys.argv[1])
