# run_feature_build.py
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

from helpers import *
from paths_and_stuff import *

from new_anomaly_detection_preprocessing import (
    apply_basic_casts_inplace,
    one_hot_encode_inplace,
    clean_numeric_inplace,
    pre_model_prep,
)
from new_anomaly_detection_feature_engineering import (
    engineer_tx_features,
    compute_account_features,
    compute_uniques_and_hhi,
    attach_sender_receiver_features,
)

csv_sep = ";"
output_dir = create_new_folder(folderPath, datetime.now().strftime("%Y-%m-%d"))

start = time.time()

df = read_csv_custom(filePath, nrows=5000)
df = df.sample(n=500)

apply_basic_casts_inplace(df)

# Tx features (semantic)
tx = engineer_tx_features(df)

# 👉 preprocessing responsibility
one_hot_encode_inplace(tx, "Payment Format", "PF")
one_hot_encode_inplace(tx, "Payment Currency", "PC")

# Account features
acc = compute_account_features(df)
acc = acc.merge(compute_uniques_and_hhi(df), on="Account", how="left")

clean_numeric_inplace(acc)
clean_numeric_inplace(tx)

tx_model = attach_sender_receiver_features(tx, acc)

if "Timestamp" in tx_model.columns:
    tx_model["Timestamp"] = pd.to_datetime(tx_model["Timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

X_acc, acc_feat_names, _ = pre_model_prep(acc)
X_tx, tx_feat_names, _ = pre_model_prep(tx_model)

today = datetime.now().strftime("%Y-%m-%d")
out_dir = Path(output_dir)

acc.to_csv(out_dir / f"account_features_{today}.csv", sep=csv_sep, index=False)
tx_model.to_csv(out_dir / f"tx_with_sender_receiver_features_{today}.csv", sep=csv_sep, index=False)

pd.DataFrame(X_acc, columns=acc_feat_names).to_csv(
    out_dir / f"acc_pre_model_{today}.csv", sep=csv_sep, index=False
)
pd.DataFrame(X_tx, columns=tx_feat_names).to_csv(
    out_dir / f"tx_pre_model_with_account_context_pre_model_{today}.csv",
    sep=csv_sep,
    index=False,
)

print("Execution time:", time.time() - start, "seconds")
