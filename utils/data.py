from typing import Any
import pandas as pd
import os


def load_dataset(file_path: str, file_type: str = 'csv', base_dir: str = "data") -> Any:
    """
    Load a dataset from a file.

    Args:
        file_path: Path to the dataset file
        file_type: Type of file ('csv', 'json', 'parquet', etc.)
        base_dir: Base directory for the file path

    Returns:
        Loaded dataset (typically a pandas DataFrame)
    """
    try:
        if file_type.lower() == 'csv':
            return pd.read_csv(os.path.join(base_dir, file_path))
        elif file_type.lower() == 'json':
            return pd.read_json(os.path.join(base_dir, file_path))
        elif file_type.lower() == 'parquet':
            return pd.read_parquet(os.path.join(base_dir, file_path))
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    except (FileNotFoundError, PermissionError) as e:
        raise IOError(f"Error accessing file: {str(e)}")
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"Dataset is empty: {str(e)}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing dataset: {str(e)}")
