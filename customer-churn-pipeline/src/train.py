import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import os

# Import our preprocessing function
from preprocess import preprocess_data

# --- 1. Load and Preprocess Data ---
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'telco_customer_churn.csv')
df = pd.read_csv(DATA_PATH)

# Preprocess the data using the function from preprocess.py
X_train, X_test, y_train, y_test = preprocess_data(df, target_column='Churn')


# --- 2. Define Preprocessing for Model ---
# Identify categorical and numerical features
categorical_features = X_train.select_dtypes(include=['object']).columns
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns

# Create preprocessing pipelines for both numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


# --- 3. Define and Train the Model ---
# Create the full pipeline including the preprocessor and the model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

# Train the model
print("Training the model...")
model_pipeline.fit(X_train, y_train)


# --- 4. Evaluate the Model ---
print("Evaluating the model...")
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")


# --- 5. Save the Model ---
# Create an 'artifacts' directory if it doesn't exist
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Save the entire pipeline
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'model_pipeline.joblib')
joblib.dump(model_pipeline, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")