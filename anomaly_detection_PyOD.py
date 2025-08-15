import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.pca import PCA
from pyod.models.auto_encoder import AutoEncoder
from pyod.utils.utility import standardizer
import matplotlib.pyplot as plt
from helpers import *
from paths_and_stuff import * 


# --- 1. Load data ---
newFolderPath = create_new_folder(folderPath, "2025-08-15")
df = read_csv_custom(filePath, nrows=10000)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# --- 2. Feature selection ---

# Numerical features
numeric_features = ["Amount Received", "Amount Paid"]

# Categorical features for one-hot encoding
categorical_features = ["Payment Format"]

# One-hot encode selected categorical features
df_encoded = pd.get_dummies(df[categorical_features], drop_first=True)

# Combine into feature matrix
X_raw = pd.concat([df[numeric_features], df_encoded], axis=1)

# --- 3. Standardize feature matrix ---
X_scaled = StandardScaler().fit_transform(X_raw)

# --- 4. Define multiple models for comparison ---
models = {
    "IsolationForest": IForest(contamination=0.1, random_state=42),
    "LocalOutlierFactor": LOF(contamination=0.1),
    "PCA": PCA(contamination=0.1),
   #"AutoEncoder": AutoEncoder(hidden_neurons=[8, 4, 4, 8], epochs=30, contamination=0.1, verbose=0)
}

# --- 5. Fit and compare models ---
results = {}

for name, model in models.items():
    model.fit(X_scaled)
    labels = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)
    results[name] = {
        "labels": labels,
        "scores": scores
    }
    print(f"🔍 {name}: {sum(labels)} anomalies detected")

# --- 6. Choose one model to visualize (e.g. Isolation Forest) ---
model_to_use = "IsolationForest"
df["Anomaly"] = results[model_to_use]["labels"]
df["Anomaly_Score"] = results[model_to_use]["scores"]

# --- 7. Save anomalies ---
anomalies = df[df["Anomaly"] == 1]
anomalies.to_csv(newFolderPath + "flagged_anomalies.csv", index=False)

# --- 8. Plot anomaly scores over time ---
plt.figure(figsize=(12, 4))
plt.scatter(df["Timestamp"], df["Amount Paid"], c=df["Anomaly"], cmap="coolwarm", alpha=0.7)
plt.title(f"Anomaly Detection with {model_to_use}")
plt.xlabel("Timestamp")
plt.ylabel("Amount Paid")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- 9. Optional: Score histogram ---
plt.figure(figsize=(8, 4))
plt.hist(df["Anomaly_Score"], bins=50, color="gray", edgecolor="black")
plt.title(f"Anomaly Scores ({model_to_use})")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
