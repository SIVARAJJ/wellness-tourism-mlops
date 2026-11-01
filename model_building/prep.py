
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Locate the project and dataset
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "tourism.csv"

# Load the registered dataset
df = pd.read_csv(DATA_PATH)

# Remove columns that are not useful for prediction
df = df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")

# Separate features (X) and target (y)
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save the prepared datasets in the project root directory
X_train.to_csv(PROJECT_DIR / "Xtrain.csv", index=False)
X_test.to_csv(PROJECT_DIR / "Xtest.csv", index=False)
y_train.to_csv(PROJECT_DIR / "ytrain.csv", index=False)
y_test.to_csv(PROJECT_DIR / "ytest.csv", index=False)

# Print a summary
print("DATA PREPARATION SUCCESSFUL")
print("-" * 40)
print(f"Original dataset shape: {pd.read_csv(DATA_PATH).shape}")
print(f"Features retained: {X.shape[1]}")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"Training target distribution:\n{y_train.value_counts()}")
print(f"Testing target distribution:\n{y_test.value_counts()}")
print("\nSaved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
