import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20
DATA_PATH = "train_data (1).csv"
TARGET = "RecommendationCount"
PLOT_DIR = "milestone1_plots"


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def find_data_file(path):
    """Try the current folder first, then the same folder as this script."""
    if os.path.exists(path):
        return path

    script_folder = os.path.dirname(os.path.abspath(__file__))
    second_try = os.path.join(script_folder, path)
    if os.path.exists(second_try):
        return second_try

    raise FileNotFoundError(
        "train_data.csv was not found. Put it in the same folder as this Python file."
    )


def count_languages(text):
    """Simple count based on the SupportedLanguages text column."""
    if pd.isna(text) or str(text).strip() == "":
        return 0

    text = str(text)
    for sep in [",", ";", "|"]:
        text = text.replace(sep, " ")

    words = [w.strip() for w in text.split() if len(w.strip()) > 2]
    return len(words)


def evaluate_model(y_true_log, y_pred_log, label):
    """Return regression metrics on log scale and original scale."""
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    r2_log = r2_score(y_true_log, y_pred_log)

    y_true_original = np.expm1(y_true_log)
    y_pred_original = np.expm1(np.clip(y_pred_log, 0, None))

    mae_original = mean_absolute_error(y_true_original, y_pred_original)
    rmse_original = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
    r2_original = r2_score(y_true_original, y_pred_original)

    return {
        "Model": label,
        "MAE log": round(mae_log, 4),
        "RMSE log": round(rmse_log, 4),
        "R2 log": round(r2_log, 4),
        "MAE original": round(mae_original, 2),
        "RMSE original": round(rmse_original, 2),
        "R2 original": round(r2_original, 4),
    }


def save_actual_vs_predicted_plot(y_true_log, y_pred_log, title, file_name):
    """Plot actual values against predicted values."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(np.clip(y_pred_log, 0, None))

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.45)

    limit = max(y_true.max(), y_pred.max())
    plt.plot([0, limit], [0, limit], linestyle="--")

    plt.xlabel("Actual RecommendationCount")
    plt.ylabel("Predicted RecommendationCount")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, file_name), dpi=300)
    plt.close()


def save_residual_plot(y_true_log, y_pred_log, title, file_name):
    """Plot prediction errors against predicted values."""
    residuals = y_true_log - y_pred_log

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred_log, residuals, alpha=0.45)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted value (log scale)")
    plt.ylabel("Residual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, file_name), dpi=300)
    plt.close()



# Step 1: Load data

banner("STEP 1: LOAD DATA")

DATA_PATH = find_data_file(DATA_PATH)
df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape[0]:,} rows and {df.shape[1]} columns")

if TARGET not in df.columns:
    raise ValueError(f"The target column '{TARGET}' was not found in the dataset.")

print("\nTarget summary:")
print(df[TARGET].describe().round(2))
print(f"Missing target values: {df[TARGET].isna().sum()}")

# Remove rows with missing target because the model cannot train without y values.
df = df.dropna(subset=[TARGET]).copy()
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).copy()
df = df[df[TARGET] >= 0].copy()



# Step 2: Create a few useful extra columns

banner("STEP 2: FEATURE ENGINEERING")

df_eng = df.copy()

if "SupportedLanguages" in df_eng.columns:
    df_eng["LanguageCount"] = df_eng["SupportedLanguages"].apply(count_languages)
else:
    df_eng["LanguageCount"] = 0

if "ReleaseDate" in df_eng.columns:
    release_year = pd.to_datetime(df_eng["ReleaseDate"], errors="coerce").dt.year
    df_eng["ReleaseYear"] = release_year.fillna(release_year.median()).astype(int)
else:
    df_eng["ReleaseYear"] = 2000

platform_cols = [c for c in ["PlatformWindows", "PlatformLinux", "PlatformMac"] if c in df_eng.columns]
if platform_cols:
    df_eng["PlatformCount"] = df_eng[platform_cols].astype(int).sum(axis=1)
else:
    df_eng["PlatformCount"] = 0

if "PriceInitial" in df_eng.columns and "PriceFinal" in df_eng.columns:
    df_eng["IsOnSale"] = (
        (pd.to_numeric(df_eng["PriceInitial"], errors="coerce") >
         pd.to_numeric(df_eng["PriceFinal"], errors="coerce")) &
        (pd.to_numeric(df_eng["PriceInitial"], errors="coerce") > 0)
    ).astype(int)
else:
    df_eng["IsOnSale"] = 0

genre_cols = [c for c in df_eng.columns if c.startswith("GenreIs")]
if genre_cols:
    df_eng["GenreCount"] = df_eng[genre_cols].astype(int).sum(axis=1)
else:
    df_eng["GenreCount"] = 0

category_cols = [c for c in df_eng.columns if c.startswith("Category")]
if category_cols:
    df_eng["CategoryCount"] = df_eng[category_cols].astype(int).sum(axis=1)
else:
    df_eng["CategoryCount"] = 0

print("New columns added: LanguageCount, ReleaseYear, PlatformCount, IsOnSale, GenreCount, CategoryCount")



# Step 3: Remove columns that should not be used for modelling

banner("STEP 3: REMOVE UNSUITABLE COLUMNS")

# These columns were checked and removed because they are IDs, long text fields,
# or popularity-related fields that are too close to the target value.
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

existing_removed = [c for c in columns_to_remove if c in df_eng.columns]
df_model = df_eng.drop(columns=existing_removed)

print(f"Removed columns: {len(existing_removed)}")
print(f"Columns left for modelling, including target: {df_model.shape[1]}")



# Step 4: Basic cleaning and encoding

banner("STEP 4: CLEANING AND ENCODING")

print("Missing values before cleaning:")
missing_before = df_model.isna().sum()
print(missing_before[missing_before > 0])

for col in df_model.columns:
    if col == TARGET:
        continue

    if df_model[col].dtype == "bool":
        df_model[col] = df_model[col].fillna(False).astype(int)
    elif pd.api.types.is_numeric_dtype(df_model[col]):
        df_model[col] = df_model[col].fillna(df_model[col].median())
    else:
        # Keep this simple: small categorical columns are encoded later.
        df_model[col] = df_model[col].fillna("Unknown")

# Convert any remaining boolean columns to 0/1.
bool_cols = df_model.select_dtypes(include=["bool"]).columns
for col in bool_cols:
    df_model[col] = df_model[col].astype(int)

# Encode remaining categorical columns if any are still present.
cat_cols = [c for c in df_model.select_dtypes(include=["object"]).columns if c != TARGET]
if cat_cols:
    print(f"Encoding categorical columns: {cat_cols}")
    df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

print(f"Missing values after cleaning: {df_model.isna().sum().sum()}")



# Step 5: Outlier handling on continuous numeric columns

banner("STEP 5: OUTLIER HANDLING")

numeric_features = [c for c in df_model.columns if c != TARGET and pd.api.types.is_numeric_dtype(df_model[c])]
continuous_features = [c for c in numeric_features if df_model[c].nunique() > 10]

print(f"Continuous columns checked for outliers: {len(continuous_features)}")

for col in continuous_features:
    q1 = df_model[col].quantile(0.25)
    q3 = df_model[col].quantile(0.75)
    iqr = q3 - q1

    if iqr == 0 or pd.isna(iqr):
        continue

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    clipped_count = ((df_model[col] < lower) | (df_model[col] > upper)).sum()
    df_model[col] = df_model[col].clip(lower, upper)

    if clipped_count > 0:
        print(f"{col}: clipped {clipped_count} values")



# Step 6: Split X and y

banner("STEP 6: PREPARE X AND y")

X = df_model.drop(columns=[TARGET])
y_original = df_model[TARGET]
y = np.log1p(y_original)

# Make sure every feature is numeric before sklearn receives the data.
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())

print(f"X shape: {X.shape}")
print(f"Target skew before log transform: {y_original.skew():.2f}")
print(f"Target skew after log transform: {y.skew():.2f}")



# Step 7: Train/test split and scaling

banner("STEP 7: TRAIN TEST SPLIT AND SCALING")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

feature_names = X.columns.tolist()

print(f"Training rows: {X_train.shape[0]:,}")
print(f"Testing rows: {X_test.shape[0]:,}")
print(f"Number of features before selection: {X_train.shape[1]}")



# Step 8: Feature selection using Random Forest importance

banner("STEP 8: FEATURE SELECTION")

rf_selector = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_selector.fit(X_train_scaled, y_train)

importances = pd.Series(rf_selector.feature_importances_, index=feature_names)
importances_sorted = importances.sort_values(ascending=False)

TOP_N = min(25, len(importances_sorted))
top_features = importances_sorted.head(TOP_N).index.tolist()

print(f"Selected top {TOP_N} features:")
for feature in top_features:
    print(f"{feature}: {importances[feature]:.4f}")

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)

print("\nSpearman check for the first 10 selected features:")
for feature in top_features[:10]:
    corr, p_value = spearmanr(X_train_scaled_df[feature], y_train)
    print(f"{feature}: r={corr:.3f}, p={p_value:.3g}")

X_train_selected = X_train_scaled_df[top_features].values
X_test_selected = X_test_scaled_df[top_features].values



# Step 9: Model 1 - Ridge Regression

banner("STEP 9: RIDGE REGRESSION")

ridge_grid = {"alpha": [0.1, 1.0, 10.0, 100.0]}
ridge_search = GridSearchCV(
    Ridge(),
    ridge_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)
ridge_search.fit(X_train_selected, y_train)
ridge_model = ridge_search.best_estimator_

ridge_train_pred = ridge_model.predict(X_train_selected)
ridge_test_pred = ridge_model.predict(X_test_selected)

print(f"Best alpha: {ridge_search.best_params_['alpha']}")
print(f"CV RMSE: {np.sqrt(-ridge_search.best_score_):.4f}")



# Step 10: Model 2 - Gradient Boosting Regressor

banner("STEP 10: GRADIENT BOOSTING REGRESSOR")

gbr_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 4],
}

gbr_search = GridSearchCV(
    GradientBoostingRegressor(random_state=RANDOM_STATE),
    gbr_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)
gbr_search.fit(X_train_selected, y_train)
gbr_model = gbr_search.best_estimator_

gbr_train_pred = gbr_model.predict(X_train_selected)
gbr_test_pred = gbr_model.predict(X_test_selected)

print(f"Best parameters: {gbr_search.best_params_}")
print(f"CV RMSE: {np.sqrt(-gbr_search.best_score_):.4f}")



# Step 11: Evaluation

banner("STEP 11: EVALUATION")

results = [
    evaluate_model(y_train, ridge_train_pred, "Ridge train"),
    evaluate_model(y_test, ridge_test_pred, "Ridge test"),
    evaluate_model(y_train, gbr_train_pred, "Gradient Boosting train"),
    evaluate_model(y_test, gbr_test_pred, "Gradient Boosting test"),
]

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\nCross-validation R2 on selected training data:")
for name, model in [("Ridge", ridge_model), ("Gradient Boosting", gbr_model)]:
    scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring="r2", n_jobs=-1)
    print(f"{name}: {scores.mean():.4f} +/- {scores.std():.4f}")

ridge_test_r2 = results_df.loc[results_df["Model"] == "Ridge test", "R2 log"].iloc[0]
gbr_test_r2 = results_df.loc[results_df["Model"] == "Gradient Boosting test", "R2 log"].iloc[0]

if gbr_test_r2 > ridge_test_r2:
    print("\nBetter test result: Gradient Boosting Regressor")
else:
    print("\nBetter test result: Ridge Regression")



# Step 12: Feature importance and coefficients

banner("STEP 12: FEATURE IMPORTANCE")

gbr_importance = pd.Series(gbr_model.feature_importances_, index=top_features).sort_values(ascending=False)
ridge_coefficients = pd.Series(np.abs(ridge_model.coef_), index=top_features).sort_values(ascending=False)

print("Top Gradient Boosting features:")
print(gbr_importance.head(15).round(4).to_string())

print("\nTop Ridge coefficients:")
print(ridge_coefficients.head(15).round(4).to_string())



# Step 13: Residual check

banner("STEP 13: RESIDUAL CHECK")

for name, predictions in [("Ridge", ridge_test_pred), ("Gradient Boosting", gbr_test_pred)]:
    residuals = y_test.values - predictions
    print(f"\n{name}")
    print(f"Mean residual: {residuals.mean():.4f}")
    print(f"Residual standard deviation: {residuals.std():.4f}")



# Step 14: Plots for the report

banner("STEP 14: SAVE REGRESSION PLOTS")

os.makedirs(PLOT_DIR, exist_ok=True)

save_actual_vs_predicted_plot(
    y_test, ridge_test_pred,
    "Ridge Regression: Actual vs Predicted",
    "ridge_actual_vs_predicted.png"
)

save_actual_vs_predicted_plot(
    y_test, gbr_test_pred,
    "Gradient Boosting: Actual vs Predicted",
    "gbr_actual_vs_predicted.png"
)

save_residual_plot(
    y_test, ridge_test_pred,
    "Ridge Regression: Residual Plot",
    "ridge_residual_plot.png"
)

save_residual_plot(
    y_test, gbr_test_pred,
    "Gradient Boosting: Residual Plot",
    "gbr_residual_plot.png"
)

# This is useful for the report because it shows which features were most useful.
top_gbr = gbr_importance.sort_values(ascending=True).tail(15)
plt.figure(figsize=(9, 7))
plt.barh(top_gbr.index, top_gbr.values)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Features from Gradient Boosting")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "gbr_feature_importance.png"), dpi=300)
plt.close()

print(f"Plots saved in: {PLOT_DIR}")



# Save model objects

banner("STEP 15: SAVE FILES")

model_files = {
    "ridge_model": ridge_model,
    "gradient_boosting_model": gbr_model,
    "feature_selector": rf_selector,
    "scaler": scaler,
    "selected_features": top_features,
    "all_feature_names": feature_names,
    "metrics": results_df,
    "feature_importance": importances_sorted,
}

with open("milestone1_models.pkl", "wb") as file:
    pickle.dump(model_files, file)

results_df.to_csv("milestone1_model_results.csv", index=False)

print("Saved: milestone1_models.pkl")
print("Saved: milestone1_model_results.csv")
print("Script finished.")
