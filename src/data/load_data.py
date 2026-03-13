import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads CSV data into a pandas DataFrame and ensures numeric consistency.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset with corrected data types.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        
        df['TotalCharges'] = df['TotalCharges'].fillna(0)

    return df