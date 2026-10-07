import sys
import pickle
from pathlib import Path

# Add project folder path to system directories
sys.path.append("/content/drive/MyDrive/Freight_Cost_Project/freight_cost_prediction")

from data_preprocessing import load_vendor_invoice_data, prepare_features, split_data
from model_evaluation import (
    train_decision_tree,
    train_random_forest,
    evaluate_model
)
from sklearn.linear_model import LogisticRegression

def main():
    db_path = "/content/drive/MyDrive/Freight_Cost_Project/inventory.db"
    model_dir = Path("/content/drive/MyDrive/Freight_Cost_Project/models")
    model_dir.mkdir(exist_ok=True)
    
    # 1. Pipeline extraction steps
    df = load_vendor_invoice_data(db_path)
    
    # 2. Add our customized classification risk targets manually (like we did in the notebook)
    df["Freight"] = df["Dollars"] - (df["Quantity"] * df["PurchasePrice"])
    df["avg_receiving_delay"] = 12 # Backup fallback default to avoid null errors
    
    # Generate the binary classification targets
    def assign_risk_label(row):
        if abs(row["Dollars"] - (row["Quantity"] * row["PurchasePrice"])) > 5:
            return 1
        return 0
        
    df["flag_invoice"] = df.apply(assign_risk_label, axis=1)
    
    # 3. Separate classification features
    features = ['Quantity', 'Dollars', 'Freight']
    X = df[features]
    y = df["flag_invoice"]
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 4. Train the classification algorithms
    print("⏳ Training classification models...")
    clf_model1 = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
    clf_model2 = train_decision_tree(X_train, y_train)
    clf_model3 = train_random_forest(X_train, y_train)
    
    # 5. Evaluate models using the helper function we updated
    evaluate_model(clf_model1, X_test, y_test, "Logistic Regression")
    evaluate_model(clf_model2, X_test, y_test, "Decision Tree Classifier")
    evaluate_model(clf_model3, X_test, y_test, "Random Forest Classifier")
    
    # 6. DEPLOYMENT STEP: Freeze the top-performing model permanently to disk
    best_model_path = model_dir / "best_invoice_classifier.pkl"
    with open(best_model_path, "wb") as f:
        pickle.dump(clf_model3, f)
        
    print(f"\n🎉 Deployed successfully! Best classifier saved at: {best_model_path}")

if __name__ == "__main__":
    main()
