import pandas as pd

# Sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Utils
from utils.data import load_dataset

SETTINGS = {
    "training": {
        "test_size": 0.2,
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

# Define parameter grid for Random Forest
param_grid = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [10, 20, 30, 40, 50, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False]
}

# Perform hyperparameter tuning
search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_grid,
    scoring="neg_root_mean_squared_error",
    n_iter=20,
    cv=3,
    verbose=1,
    random_state=42
)

search.fit(X_train, y_train)

# Retrieve and print best parameters
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
