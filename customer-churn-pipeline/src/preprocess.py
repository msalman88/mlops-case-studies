import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    """
    Preprocesses the Telco Customer Churn dataset.
    
    Args:
        df (pd.DataFrame): The raw input DataFrame.
        target_column (str): The name of the target variable column.
        test_size (float): The proportion of the dataset to allocate to the test split.
        random_state (int): The seed for reproducibility.
        
    Returns:
        A tuple containing:
        - X_train (pd.DataFrame): Training features.
        - X_test (pd.DataFrame): Testing features.
        - y_train (pd.Series): Training target.
        - y_test (pd.Series): Testing target.
    """
    # Drop the customerID column as it's just an identifier
    df = df.drop('customerID', axis=1)

    # Convert 'TotalCharges' to numeric, coercing errors to NaN
    # The raw data has empty strings for new customers
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Fill any resulting NaNs (for new customers) with 0
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # Convert target variable 'Churn' to binary (1 for 'Yes', 0 for 'No')
    df[target_column] = df[target_column].apply(lambda x: 1 if x == 'Yes' else 0)

    # Separate features (X) and target (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test