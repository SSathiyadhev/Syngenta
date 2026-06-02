# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data (adjust paths as needed)
growers = pd.read_csv("data/growers.csv")
whatsapp = pd.read_csv("data/whatsapp_campaign.csv")
# Merge
df = pd.merge(whatsapp, growers, on="grower_id", how="inner")

# Features
features = ['state', 'district', 'language', 'device_type', 'grower_age', 'grower_farm_size', 'campaign_crop']
df = df.dropna(subset=['clicked_status'])

# Fill missing numeric
df['grower_age'] = df['grower_age'].fillna(df['grower_age'].median())
df['grower_farm_size'] = df['grower_farm_size'].fillna(df['grower_farm_size'].median())

# Encode categorical
categorical_cols = ['state', 'district', 'language', 'device_type', 'campaign_crop']
encoders = {}
for col in categorical_cols:
    df[col] = df[col].fillna('Unknown').astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df[features]
y = df['clicked_status'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"✅ Model accuracy: {model.score(X_test, y_test)*100:.2f}%")

joblib.dump(model, "receptivity_model.pkl")
joblib.dump(encoders, "label_encoders.pkl")
print("✅ Saved model and encoders.")