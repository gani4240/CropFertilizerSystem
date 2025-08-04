import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load dataset
fertilizer = pd.read_csv("dataset/Fertilizer Prediction.csv")

# Encode categorical features
fert_dict = {'Urea': 1, 'DAP': 2, '14-35-14': 3, '28-28': 4, '17-17-17': 5, '20-20': 6, '10-26-26': 7}
fertilizer['fert_no'] = fertilizer['Fertilizer Name'].map(fert_dict)
fertilizer.drop('Fertilizer Name', axis=1, inplace=True)

lb = LabelEncoder()
fertilizer["Soil Type"] = lb.fit_transform(fertilizer['Soil Type'])
fertilizer['Crop Type'] = lb.fit_transform(fertilizer['Crop Type'])

# Split dataset
X = fertilizer.drop('fert_no', axis=1)
y = fertilizer['fert_no']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
fert_model = DecisionTreeClassifier()
fert_model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(fert_model, "fertilizer_model.pkl")
joblib.dump(scaler, "scaler_fertilizer.pkl")

print("Fertilizer recommendation model and scaler saved successfully!")
