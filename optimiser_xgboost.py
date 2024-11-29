import pandas as pd

# Sklearn and XGBoost
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

# Utils
from utils.data import load_dataset

SETTINGS = {
    "training": {
        "test_size": 0.2,
    },
    "model": "xgboost",
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
data["household_density"] = data["population"] / data["total_rooms"]

# Drop redundant columns
data.drop(["total_rooms", "total_bedrooms", "population",
          "households"], axis=1, inplace=True)

# Encode categorical variable
data = pd.get_dummies(data, columns=["ocean_proximity"], drop_first=True)

# Separate features and target
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

# Feature scaling
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, **SETTINGS["training"])

# Perform hyperparameter tuning
param_grid = {
    "n_estimators": [500, 1000],
    "learning_rate": [0.01, 0.03, 0.05],
    "max_depth": [8, 10, 12, 15, 20, 25, 30],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2],
    "gamma": [0, 1, 5, 10, 15, 20],
    "reg_alpha": [0, 2, 5, 10, 15, 20],
    "reg_lambda": [1, 3, 5, 10, 15, 20],
}

search = RandomizedSearchCV(
    XGBRegressor(random_state=42),
    param_distributions=param_grid,
    scoring="neg_root_mean_squared_error",
    n_iter=20,
    cv=3,
    verbose=1,
    random_state=42
)

search.fit(X_train, y_train)

# Retrieve best parameters
best_params = search.best_params_
print("[INFO] Best Parameters:", best_params)

# Get feature importance
best_model = search.best_estimator_
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
})
feature_importance = feature_importance.sort_values(
    by="Importance", ascending=False)

print("\n[INFO] Feature Importance:")
print(feature_importance)
