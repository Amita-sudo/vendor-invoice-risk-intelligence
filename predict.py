import sys
import pickle
import pandas as pd
from pathlib import Path

def run_production_inference(new_invoices_path: str):
    """
    Loads the deployed ML model and automatically scans a fresh batch of invoices.
    """
    # 1. Load the frozen model file from disk
    model_file = Path("/content/drive/MyDrive/Freight_Cost_Project/models/best_invoice_classifier.pkl")
    if not model_file.exists():
        raise FileNotFoundError("❌ Deployed model file not found! Train it first.")
        
    with open(model_file, 'rb') as file:
        deployed_model = pickle.load(file)
        
    # 2. Read the new spreadsheet or database batch uploaded by the company
    df = pd.read_csv(new_invoices_path)
    
    # 3. Extract the features the classifier expects to see
    X_new = df[['invoice_quantity', 'invoice_dollars', 'Freight', 
                'days_po_to_invoice', 'days_to_pay', 'total_brands', 
                'total_item_quantity', 'total_item_dollars', 'avg_receiving_delay']]
    
    # 4. Let the machine learning system predict risk categories automatically
    df['Anomaly_Flag'] = deployed_model.predict(X_new)
    
    # 5. Export a priority checklist report for the auditing team
    flagged_report = df[df['Anomaly_Flag'] == 1]
    output_report_path = "/content/drive/MyDrive/Freight_Cost_Project/flagged_audit_report.csv"
    flagged_report.to_csv(output_report_path, index=False)
    
    print(f"🚀 Scan complete! Found {len(flagged_report)} high-risk anomalies.")
    print(f"📁 Downloadable report exported to: {output_report_path}")

if __name__ == "__main__":
    # If this file is called from the command line, it executes instantly
    if len(sys.argv) > 1:
        run_production_inference(sys.argv[1])
