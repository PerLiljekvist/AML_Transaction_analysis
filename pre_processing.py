import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from paths_and_stuff import *
from helpers import *

# Load data (replace with your filename or StringIO for this sample)
new_folder = create_new_folder(folderPath, '2025-08-26_pre_processing')
df = read_csv_custom(filePath, nrows=10000)

# 1. Missing Value Treatment
# Numeric columns: impute with mean, fallback to median if desired
num_cols = df.select_dtypes(include=[np.number]).columns
imputer_mean = SimpleImputer(strategy='mean')
df[num_cols] = imputer_mean.fit_transform(df[num_cols])

# Alternatively, for each column if you want to mix mean/median:
# for col in num_cols:
#     if df[col].isnull().any():
#         # Use median for skewed columns, mean otherwise (custom logic)
#         df[col].fillna(df[col].median(), inplace=True)

# 2. Outlier Treatment (for numeric columns, winsorization at 1st and 99th percentiles)
for col in num_cols:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = np.clip(df[col], lower, upper)

# 3. Feature Scaling (standardize 'Amount Received' and 'Amount Paid')
scaler = StandardScaler()
df[['Amount Received', 'Amount Paid']] = scaler.fit_transform(df[['Amount Received', 'Amount Paid']])

# If you need MinMaxScaler:
# scaler = MinMaxScaler()
# df[['Amount Received', 'Amount Paid']] = scaler.fit_transform(df[['Amount Received', 'Amount Paid']])

# 4. Multicollinearity Treatment (dropping highly correlated features)
corr_matrix = df[num_cols].corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
df.drop(columns=to_drop, inplace=True)

# 5. Feature Encoding
# Label Encoding for binary categorical features (e.g. 'Is Laundering')
if df['Is Laundering'].dtype == 'object':
    le = LabelEncoder()
    df['Is Laundering'] = le.fit_transform(df['Is Laundering'])

# One Hot Encoding for 'Payment Format', 'Receiving Currency', 'Payment Currency', etc.
categorical_features = ['Payment Format', 'Receiving Currency', 'Payment Currency']
df = pd.get_dummies(df, columns=categorical_features)

# 6. Parse Timestamps if not already
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
# (Optional) Create new features from datetime:
df['hour'] = df['Timestamp'].dt.hour
df['day_of_week'] = df['Timestamp'].dt.dayofweek

save_df_to_csv(df,'pre_processed.csv', new_folder)

print(df.head())

