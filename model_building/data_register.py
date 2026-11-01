
from pathlib import Path
import pandas as pd

# 1. Define the expected dataset columns
EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "ProductPitched",
    "NumberOfFollowups",
    "DurationOfPitch"
]

# 2. Locate the dataset relative to this script
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "tourism.csv"

# 3. Check that the dataset file exists
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}"
    )

# 4. Load the dataset
df = pd.read_csv(DATA_PATH)

# Remove accidental whitespace from column names
df.columns = df.columns.str.strip()

# 5. Validate the expected columns
missing_columns = [
    col for col in EXPECTED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing expected columns: {missing_columns}"
    )

# 6. Print the registration summary
print("DATA REGISTRATION SUCCESSFUL")
print("-" * 40)
print(f"Dataset location: {DATA_PATH}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print(f"Expected columns validated: {len(EXPECTED_COLUMNS)}")

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values in columns with missing data:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

print("\nTarget variable distribution (ProdTaken):")
print(df["ProdTaken"].value_counts(dropna=False))
