import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
PLOT_DIR     = "milestone2_plots"
MODEL_FILE   = "knn_model.pkl"


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)



# Preprocessing


banner("STEP 1: PREPROCESSING")

df = pd.read_csv("train_data.csv")

drop_cols = [
    'QueryID', 'ResponseID', 'QueryName', 'ResponseName', 'ReleaseDate',
    'PriceCurrency', 'SupportEmail', 'SupportURL', 'AboutText', 'Background',
    'ShortDescrip', 'DetailedDescrip', 'DRMNotice', 'ExtUserAcctNotice',
    'HeaderImage', 'LegalNotice',
    'Reviews', 'SupportedLanguages', 'Website',
    'PCMinReqsText', 'PCRecReqsText', 'LinuxMinReqsText', 'LinuxRecReqsText',
    'MacMinReqsText', 'MacRecReqsText'
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

X = df.drop(columns=['GamePopularity']).select_dtypes(include=[np.number])
y = df['GamePopularity']

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "bool"]).columns

X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X[cat_cols] = X[cat_cols].fillna("Unknown")


col_medians = X[num_cols].median()

print(f"Features: {X.shape[1]}  |  Samples: {X.shape[0]}")
print(f"Class distribution:\n{y.value_counts()}\n")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

print("Preprocessing complete.")
print("X_train shape:", X_train.shape)
print("X_val shape  :", X_val.shape)
print("y_train shape:", y_train.shape)
print("y_val shape  :", y_val.shape)



# Hyperparameter Tuning


banner("STEP 2: HYPERPARAMETER TUNING")


print("\n--- Varying n_neighbors  (weights fixed = uniform) ---")
k_values  = [3, 7, 15]
k_results = []
for k in k_values:
    m = KNeighborsClassifier(n_neighbors=k, weights="uniform")
    m.fit(X_train, y_train)
    acc = accuracy_score(y_val, m.predict(X_val))
    k_results.append({"n_neighbors": k, "Accuracy": round(acc, 4)})
    print(f"  n_neighbors = {k:<4}  ->  accuracy = {acc:.4f}")


print("\n--- Varying weights  (n_neighbors fixed = 7) ---")
weight_values  = ["uniform", "distance"]
weight_results = []
for w in weight_values:
    m = KNeighborsClassifier(n_neighbors=7, weights=w)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_val, m.predict(X_val))
    weight_results.append({"weights": w, "Accuracy": round(acc, 4)})
    print(f"  weights = {w:<12}  ->  accuracy = {acc:.4f}")



# Train final model with best hyperparameters


banner("STEP 3: TRAIN FINAL MODEL")

best_k      = max(k_results,      key=lambda x: x["Accuracy"])["n_neighbors"]
best_weight = max(weight_results, key=lambda x: x["Accuracy"])["weights"]
print(f"Best n_neighbors : {best_k}")
print(f"Best weights     : {best_weight}")

t0 = time.time()
final_model = KNeighborsClassifier(n_neighbors=best_k, weights=best_weight)
final_model.fit(X_train, y_train)
train_time = time.time() - t0

t0 = time.time()
y_pred = final_model.predict(X_val)
test_time = time.time() - t0

accuracy = accuracy_score(y_val, y_pred)
print(f"\nFinal accuracy : {accuracy:.4f}")
print(f"Training time  : {train_time:.4f}s")
print(f"Test time      : {test_time:.4f}s")



# Evaluation


banner("STEP 4: EVALUATION")

print(classification_report(y_val, y_pred))

cm = confusion_matrix(y_val, y_pred, labels=["High", "Low", "Medium"])
print("Confusion matrix (rows = actual, cols = predicted):")
print(pd.DataFrame(cm, index=["High", "Low", "Medium"], columns=["High", "Low", "Medium"]))



# Save plots


banner("STEP 5: SAVE PLOTS")

os.makedirs(PLOT_DIR, exist_ok=True)

# Plot 1: Accuracy vs n_neighbors
plt.figure(figsize=(7, 4))
plt.bar([str(r["n_neighbors"]) for r in k_results], [r["Accuracy"] for r in k_results],
        color=["#4c78a8", "#72b7b2", "#f58518"])
for i, r in enumerate(k_results):
    plt.text(i, r["Accuracy"] + 0.005, f'{r["Accuracy"]:.4f}', ha="center")
plt.xlabel("n_neighbors")
plt.ylabel("Accuracy")
plt.title("KNN: Accuracy vs n_neighbors  (weights = uniform)")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "knn_accuracy_vs_k.png"), dpi=300)
plt.show()
plt.close()

# Plot 2: Accuracy vs weights
plt.figure(figsize=(7, 4))
plt.bar([r["weights"] for r in weight_results], [r["Accuracy"] for r in weight_results],
        color=["#4c78a8", "#72b7b2"])
for i, r in enumerate(weight_results):
    plt.text(i, r["Accuracy"] + 0.005, f'{r["Accuracy"]:.4f}', ha="center")
plt.xlabel("Weights")
plt.ylabel("Accuracy")
plt.title("KNN: Accuracy vs Weights  (n_neighbors = 7)")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "knn_accuracy_vs_weights.png"), dpi=300)
plt.show()
plt.close()

# Plot 3: Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im, ax=ax)
ax.set(xticks=range(3), yticks=range(3),
       xticklabels=["High", "Low", "Medium"],
       yticklabels=["High", "Low", "Medium"],
       xlabel="Predicted", ylabel="Actual",
       title="Confusion Matrix — KNN")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "knn_confusion_matrix.png"), dpi=300)
plt.show()
plt.close()

print(f"Plots saved in: {PLOT_DIR}/")



# Save model with pickle


banner("STEP 6: SAVE MODEL")

save_bundle = {
    "model":         final_model,
    "scaler":        scaler,
    "col_medians":   col_medians,
    "feature_names": list(X.columns),
    "metrics": {
        "accuracy":   accuracy,
        "train_time": train_time,
        "test_time":  test_time,
    },
}

with open(MODEL_FILE, "wb") as f:
    pickle.dump(save_bundle, f)

print(f"Saved: {MODEL_FILE}")
print(f"\n  Accuracy      : {accuracy:.4f}")
print(f"  Training time : {train_time:.4f}s")
print(f"  Test time     : {test_time:.4f}s")