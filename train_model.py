import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Load dataset
df = pd.read_csv(
    r"C:\Users\balaj\Documents\DataProjects\Fraud-Detection-Project\Data\fraudTest.csv"
)

# 2. Feature Engineering
# Convert transaction time to datetime
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

# Create time-based features
df["hour"] = df["trans_date_trans_time"].dt.hour
df["day"] = df["trans_date_trans_time"].dt.day
df["weekday"] = df["trans_date_trans_time"].dt.weekday
df["month"] = df["trans_date_trans_time"].dt.month

# Drop irrelevant or high-cardinality columns
df = df.drop(columns=["Unnamed: 0", "trans_date_trans_time", "dob", "merchant", "cc_num"])

# 3. Define Features & Target
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# 4. Preprocessing Setup
categorical = ["category", "gender", "city"]   # categorical cols
numerical = ["amt", "lat", "long", "city_pop", "hour", "day", "weekday", "month"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", StandardScaler(), numerical)
])

# 5. Pipeline (Preprocessing + Model)
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42, n_estimators=200, max_depth=12))
])

# 6. Train-Test Split & Fit
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

# 7. Save the Pipeline

with open("fraud_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Training complete. Model saved as fraud_model.pkl")
