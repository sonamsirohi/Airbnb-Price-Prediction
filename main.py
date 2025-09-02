# 📦 Imports
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# ✅ File paths
DATA_FILE = "airbnb.csv"
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

# ✅ Columns to keep (features + target)
columns_to_keep = [
    'bedrooms', 'bathrooms', 'guests', 'beds', 'country',
    'rating', 'reviews', 'checkin', 'checkout'
]

# ✅ Step 1: Load data
df = pd.read_csv(DATA_FILE)

# ✅ Step 2: Preprocess
# Remove rows where beds = 0 to avoid division errors
df = df[df['beds'] != 0]

# Create target column if not present
if 'price_per_bed' not in df.columns and 'price' in df.columns:
    df['price_per_bed'] = df['price'] / df['beds']

# Drop any rows where target is missing
df.dropna(subset=['price_per_bed'], inplace=True)

# ✅ Step 3: Keep only the relevant features
df = df[columns_to_keep + ['price_per_bed']]

# ✅ Step 4: Separate features and target
X = df.drop('price_per_bed', axis=1)
y = df['price_per_bed']

# ✅ Step 5: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 6: Identify types of features
num_attr = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_attr = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# ✅ Step 7: Build pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='median')),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("onehot", OneHotEncoder(handle_unknown='ignore'))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attr),
    ("cat", cat_pipeline, cat_attr)
])

# ✅ Step 8: Prepare the data
X_train_prepared = full_pipeline.fit_transform(X_train)
X_test_prepared = full_pipeline.transform(X_test)

# ✅ Step 9: Train model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train_prepared, y_train)

# ✅ Step 10: Evaluate
y_pred = model.predict(X_test_prepared)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"✅ Model trained! Test RMSE: {rmse:.2f}")

# Show first 10 predictions and actual values side-by-side
for pred, actual in zip(y_pred[:10], y_test[:10]):
    print(f"Predicted: {pred:.2f} | Actual: {actual:.2f}")

# ✅ Step 11: Save model and pipeline
joblib.dump(model, MODEL_FILE)
joblib.dump(full_pipeline, PIPELINE_FILE)
print("✅ Model and pipeline saved successfully.")
