"""
discussion_milestone1.py
Online Games Popularity Prediction - Milestone 1

Run this file on the day of the discussion with the unseen regression test CSV.

Usage:
  python discussion_milestone1.py --test unseen_test.csv
"""

import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


MODEL_FILE = "milestone1_models.pkl"


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def count_languages(text):
    if pd.isna(text) or str(text).strip() == "":
        return 0
    text = str(text)
    for sep in [",", ";", "|"]:
        text = text.replace(sep, " ")
    return len([w.strip() for w in text.split() if len(w.strip()) > 2])


# ─────────────────────────────────────────────────────────────────────────────
# Load saved models
# ─────────────────────────────────────────────────────────────────────────────

banner("LOADING SAVED MODELS")

with open(MODEL_FILE, "rb") as f:
    bundle = pickle.load(f)

ridge_model    = bundle["ridge_model"]
gbr_model      = bundle["gradient_boosting_model"]
scaler         = bundle["scaler"]
top_features   = bundle["selected_features"]
feature_names  = bundle["all_feature_names"]

print("Models loaded successfully.")
print(f"Ridge model     : loaded")
print(f"Gradient Boost  : loaded")


# ─────────────────────────────────────────────────────────────────────────────
# Load unseen test CSV
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--test", required=True, metavar="TEST_CSV",
                    help="Path to the unseen test CSV file.")
args = parser.parse_args()

banner("LOADING UNSEEN TEST DATA")

df_test = pd.read_csv(args.test)
print(f"Test set shape: {df_test.shape}")

TARGET = "RecommendationCount"
has_labels = TARGET in df_test.columns


# ─────────────────────────────────────────────────────────────────────────────
# UNSEEN CSV
# ─────────────────────────────────────────────────────────────────────────────

banner("PREPROCESSING")

# Extract labels if present
y_true = None
if has_labels:
    y_true = pd.to_numeric(df_test[TARGET], errors="coerce")
    df_test = df_test.drop(columns=[TARGET])

# Feature engineering — same as training
if "SupportedLanguages" in df_test.columns:
    df_test["LanguageCount"] = df_test["SupportedLanguages"].apply(count_languages)
else:
    df_test["LanguageCount"] = 0

if "ReleaseDate" in df_test.columns:
    release_year = pd.to_datetime(df_test["ReleaseDate"], errors="coerce").dt.year
    df_test["ReleaseYear"] = release_year.fillna(release_year.median()).astype(int)
else:
    df_test["ReleaseYear"] = 2000

platform_cols = [c for c in ["PlatformWindows", "PlatformLinux", "PlatformMac"] if c in df_test.columns]
df_test["PlatformCount"] = df_test[platform_cols].astype(int).sum(axis=1) if platform_cols else 0

if "PriceInitial" in df_test.columns and "PriceFinal" in df_test.columns:
    df_test["IsOnSale"] = (
        (pd.to_numeric(df_test["PriceInitial"], errors="coerce") >
         pd.to_numeric(df_test["PriceFinal"],   errors="coerce")) &
        (pd.to_numeric(df_test["PriceInitial"], errors="coerce") > 0)
    ).astype(int)
else:
    df_test["IsOnSale"] = 0

genre_cols    = [c for c in df_test.columns if c.startswith("GenreIs")]
category_cols = [c for c in df_test.columns if c.startswith("Category")]
df_test["GenreCount"]    = df_test[genre_cols].astype(int).sum(axis=1)    if genre_cols    else 0
df_test["CategoryCount"] = df_test[category_cols].astype(int).sum(axis=1) if category_cols else 0

# Drop same columns as training
columns_to_remove = [
    "QueryID", "ResponseID", "QueryName", "ResponseName",
    "SteamSpyOwners", "SteamSpyOwnersVariance",
    "SteamSpyPlayersEstimate", "SteamSpyPlayersVariance",
    "SupportEmail", "SupportURL", "AboutText", "Background",
    "ShortDescrip", "DetailedDescrip", "DRMNotice", "ExtUserAcctNotice",
    "HeaderImage", "LegalNotice", "Reviews", "Website",
    "PCMinReqsText", "PCRecReqsText", "LinuxMinReqsText",
    "LinuxRecReqsText", "MacMinReqsText", "MacRecReqsText",
    "SupportedLanguages", "ReleaseDate", "PriceCurrency",
]
df_test.drop(columns=[c for c in columns_to_remove if c in df_test.columns], inplace=True)

# Fill missing values — cannot drop rows from test set
for col in df_test.columns:
    if df_test[col].dtype == "bool":
        df_test[col] = df_test[col].fillna(False).astype(int)
    elif pd.api.types.is_numeric_dtype(df_test[col]):
        df_test[col] = df_test[col].fillna(df_test[col].median())
    else:
        df_test[col] = df_test[col].fillna("Unknown")

# Encode boolean columns
for col in df_test.select_dtypes(include=["bool"]).columns:
    df_test[col] = df_test[col].astype(int)

# Encode categorical columns
cat_cols = [c for c in df_test.select_dtypes(include=["object"]).columns]
if cat_cols:
    df_test = pd.get_dummies(df_test, columns=cat_cols, drop_first=True)

# Align columns to training feature names
for col in feature_names:
    if col not in df_test.columns:
        df_test[col] = 0
df_test = df_test[feature_names]

df_test = df_test.apply(pd.to_numeric, errors="coerce").fillna(0)

# Scale using saved scaler
X_test_scaled = scaler.transform(df_test)

# Select top features
X_test_selected = pd.DataFrame(X_test_scaled, columns=feature_names)[top_features].values

print(f"Missing values after filling: {pd.DataFrame(X_test_scaled).isna().sum().sum()}")
print("Preprocessing complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────────────────────────────────────

banner("PREDICTING")

# Ridge predictions
ridge_pred_log = ridge_model.predict(X_test_selected)
ridge_pred     = np.expm1(np.clip(ridge_pred_log, 0, None))

# Gradient Boosting predictions
gbr_pred_log = gbr_model.predict(X_test_selected)
gbr_pred     = np.expm1(np.clip(gbr_pred_log, 0, None))

print(f"Ridge predictions     (first 5): {ridge_pred[:5].round(2)}")
print(f"Gradient Boost predictions (first 5): {gbr_pred[:5].round(2)}")


# ─────────────────────────────────────────────────────────────────────────────
# Show MSE and R2 if labels are present
# ─────────────────────────────────────────────────────────────────────────────

if has_labels:
    banner("RESULTS")

    y_true_clean  = y_true.fillna(y_true.median())
    y_true_log    = np.log1p(y_true_clean)

    ridge_mse = mean_squared_error(y_true_log, ridge_pred_log)
    ridge_r2  = r2_score(y_true_log, ridge_pred_log)

    gbr_mse   = mean_squared_error(y_true_log, gbr_pred_log)
    gbr_r2    = r2_score(y_true_log, gbr_pred_log)

    print(f"Ridge Regression:")
    print(f"  MSE : {ridge_mse:.4f}")
    print(f"  R2  : {ridge_r2:.4f}")

    print(f"\nGradient Boosting Regressor:")
    print(f"  MSE : {gbr_mse:.4f}")
    print(f"  R2  : {gbr_r2:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Save predictions to CSV
# ─────────────────────────────────────────────────────────────────────────────

pd.DataFrame({
    "Ridge_Prediction":            ridge_pred.round(2),
    "GradientBoosting_Prediction": gbr_pred.round(2),
}).to_csv("milestone1_test_predictions.csv", index=False)

print("\nPredictions saved to: milestone1_test_predictions.csv")