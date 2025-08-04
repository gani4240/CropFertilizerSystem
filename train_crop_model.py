import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Load dataset
crop = pd.read_csv("dataset/Crop_recommendation.csv")

# Encode crop labels
crop_dict = {
    'rice': 0, 'maize': 1, 'chickpea': 2, 'kidneybeans': 3, 'pigeonpeas': 4,
    'mothbeans': 5, 'mungbean': 6, 'blackgram': 7, 'lentil': 8, 'pomegranate': 9,
    'banana': 10, 'mango': 11, 'grapes': 12, 'watermelon': 13, 'muskmelon': 14,
    'apple': 15, 'orange': 16, 'papaya': 17, 'coconut': 18, 'cotton': 19,
    'jute': 20, 'coffee': 21
}
crop['crop_no'] = crop['label'].map(crop_dict)
crop.drop('label', axis=1, inplace=True)

# Split dataset
X = crop.drop('crop_no', axis=1)
y = crop['crop_no']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
dtc = DecisionTreeClassifier()
dtc.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(dtc, "crop_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Crop recommendation model and scaler saved successfully!")
