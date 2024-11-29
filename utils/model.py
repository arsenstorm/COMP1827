import os
import joblib

os.makedirs("models", exist_ok=True)


def save_model(model, filename):
    filepath = os.path.join("models", filename)
    print(f"[INFO] Saving model to {filepath}")
    joblib.dump(model, filepath)


def load_model(filename):
    filepath = os.path.join("models", filename)
    if os.path.exists(filepath):
        print(f"[INFO] Loading model from {filepath}")
        return joblib.load(filepath)
    return None


def generate_model_filename(settings):
    flat_params = []
    model_param = None

    for category, params in settings.items():
        if category == "model":
            model_param = f"model_{params}"
            continue

        if isinstance(params, dict):
            for key, value in params.items():
                flat_params.append(f"{key}_{value}")
        else:
            flat_params.append(f"{category}_{params}")

    if model_param:
        flat_params = [model_param] + flat_params

    return f"{'-'.join(flat_params)}.pkl"


def calculate_success_score(train_rmse, test_rmse, cv_rmse, mean_target):
    """
    Calculate an overall success score factoring in training, testing, 
    and cross-validation RMSE relative to the mean target value.

    Weights:
        - 40% for Cross-Validation RMSE (most important for generalization)
        - 30% for Testing RMSE
        - 30% for Training RMSE
    """
    train_score = (1 - (train_rmse / mean_target)) * 100
    test_score = (1 - (test_rmse / mean_target)) * 100
    cv_score = (1 - (cv_rmse / mean_target)) * 100

    # Weighted average
    success_score = (0.4 * cv_score) + (0.3 * test_score) + (0.3 * train_score)
    return success_score
