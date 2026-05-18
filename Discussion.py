import argparse
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

MODEL_FILE = "knn_model.pkl"


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)



# Load saved model


banner("LOADING SAVED MODEL")

with open(MODEL_FILE, "rb") as f:
    bundle = pickle.load(f)

model         = bundle["model"]
scaler        = bundle["scaler"]
col_medians   = bundle["col_medians"]
feature_names = bundle["feature_names"]

print("Model loaded successfully.")



# Load unseen test CSV


parser = argparse.ArgumentParser()
parser.add_argument("--test", required=True, metavar="TEST_CSV",
                    help="Path to the unseen test CSV file.")
args = parser.parse_args()

banner("LOADING UNSEEN TEST DATA")

df_test = pd.read_csv(args.test)
print(f"Test set shape: {df_test.shape}")

has_labels = "GamePopularity" in df_test.columns



# UNSEEN CSV


banner("PREPROCESSING")

# Extract labels if present
y_true = None
if has_labels:
    y_true = df_test["GamePopularity"].copy()
    df_test = df_test.drop(columns=["GamePopularity"])

# Drop same columns as training
drop_cols = [
    'QueryID', 'ResponseID', 'QueryName', 'ResponseName', 'ReleaseDate',
    'PriceCurrency', 'SupportEmail', 'SupportURL', 'AboutText', 'Background',
    'ShortDescrip', 'DetailedDescrip', 'DRMNotice', 'ExtUserAcctNotice',
    'HeaderImage', 'LegalNotice',
    'Reviews', 'SupportedLanguages', 'Website',
    'PCMinReqsText', 'PCRecReqsText', 'LinuxMinReqsText', 'LinuxRecReqsText',
    'MacMinReqsText', 'MacRecReqsText'
]
df_test.drop(columns=[c for c in drop_cols if c in df_test.columns], inplace=True)

# Keep only numeric columns
df_test = df_test.select_dtypes(include=[np.number])

# Align columns to what was seen during training
for col in feature_names:
    if col not in df_test.columns:
        df_test[col] = 0
df_test = df_test[feature_names]

# Handle missing values
df_test = df_test.fillna(col_medians)

print(f"Missing values after filling: {df_test.isna().sum().sum()}")
print("Preprocessing complete.")

# Scale using saved scaler
X_test = scaler.transform(df_test)



# Predict


banner("PREDICTING")

t0            = time.time()
y_pred        = model.predict(X_test)
test_time     = time.time() - t0

print(f"Predictions (first 20): {y_pred[:20]}")
print(f"Test time              : {test_time:.4f}s")



# Show accuracy if labels are present


if has_labels:
    banner("RESULTS")
    acc = accuracy_score(y_true, y_pred)
    print(f"Classification Accuracy : {acc:.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred))



# Save predictions to CSV


pd.DataFrame({"PredictedGamePopularity": y_pred}).to_csv("test_predictions.csv", index=False)
print("\nPredictions saved to: test_predictions.csv")