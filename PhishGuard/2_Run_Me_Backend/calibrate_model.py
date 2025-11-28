# calibrate_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---- Load your dataset ----
df = pd.read_csv("dataset_phishing_87 features.csv")

# Adjust this column name if your label differs (check the dataset headers)
target_col = "status"

# Separate features and labels
X = df.drop(columns=[target_col])
y = df[target_col]

# ---- Split into train/validation sets ----
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Train a new Random Forest ----
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# ---- Calibrate the model ----
calibrated_rf = CalibratedClassifierCV(rf, method="sigmoid", cv=3)
calibrated_rf.fit(X_val, y_val)

# ---- Evaluate ----
y_pred = calibrated_rf.predict(X_val)
print("\nModel evaluation after calibration:")
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred))
print("Recall:", recall_score(y_val, y_pred))
print("F1 Score:", f1_score(y_val, y_pred))

# ---- Save calibrated model ----
joblib.dump(calibrated_rf, "rf_phishing_model_calibrated.pkl")
print("\nSaved calibrated model as rf_phishing_model_calibrated.pkl")
