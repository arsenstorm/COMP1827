import pandas as pd
import numpy as np

# Sklearn
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

# Utils
from utils.model import save_model, load_model, generate_model_filename, calculate_success_score
from utils.data import load_dataset

SETTINGS = {
    "training": {
        "test_size": 0.2,
    },
    "creation": {
        "n_estimators": 300,
        "min_samples_split": 2,
        "max_depth": None,
        "bootstrap": True
    },
    "model": "random_forest",
}

# Load dataset
data = load_dataset("housing_data.csv", file_type="csv")

# Preprocess data
data["total_bedrooms"] = data["total_bedrooms"].fillna(
    data["total_bedrooms"].median())

# Feature engineering
data["rooms_per_household"] = data["total_rooms"] / data["households"]
data["population_per_household"] = data["population"] / data["households"]
data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]

# Drop redundant columns
data.drop(["total_rooms", "total_bedrooms", "population",
          "households"], axis=1, inplace=True)

# Encode categorical variable
data = pd.get_dummies(data, columns=["ocean_proximity"], drop_first=True)

# Separate features and target
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]  # No log transform for now

# Feature scaling
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, **SETTINGS["training"])

# Load or train model
model_filename = generate_model_filename(SETTINGS)
model = load_model(model_filename)

if model is None:
    model = RandomForestRegressor(
        random_state=42,
        min_samples_leaf=1,
        max_features="sqrt",
        **SETTINGS["creation"]
    )
    model.fit(X_train, y_train)
    save_model(model, model_filename)
    print("[INFO] Model trained and saved.")
else:
    print("[INFO] Model loaded from file.")

# Evaluate model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# Cross-validation
y_cv_pred = cross_val_predict(model, X, y, cv=5)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
cv_rmse = root_mean_squared_error(y, y_cv_pred)

# Calculate mean target value
mean_target = y.mean()
train_error_percentage = (np.sqrt(train_mse) / mean_target) * 100
test_error_percentage = (np.sqrt(test_mse) / mean_target) * 100
success_score = calculate_success_score(
    train_rmse, test_rmse, cv_rmse, mean_target)

# Print results
print(f"[INFO] Mean Target Value: {mean_target:.2f}")
print(f"[INFO] Cross-validated RMSE: {cv_rmse:.2f}")
print(f"[INFO] Training Error: {train_error_percentage:.2f}%")
print(f"[INFO] Testing Error: {test_error_percentage:.2f}%")
print(f"[INFO] Success Score: {success_score:.2f}%")
