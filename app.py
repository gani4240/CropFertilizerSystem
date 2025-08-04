import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')
# Loading the dataset
crop = pd.read_csv("dataset/Crop_recommendation.csv")
features = crop.columns.to_list()
features.remove('label')
print(features)
crop_dict = {
    'rice': 0,
    'maize': 1,
    'chickpea': 2,
    'kidneybeans': 3,
    'pigeonpeas':4,
    'mothbeans':5,
    'mungbean': 6,
    'blackgram': 7,
    'lentil': 8,
    'pomegranate': 9,
    'banana': 10,
    'mango': 11,
    'grapes': 12,
    'watermelon': 13,
    'muskmelon': 14,
    'apple': 15,
    'orange': 16,
    'papaya': 17,
    'coconut': 18,
    'cotton': 19,
    'jute': 20,
    'coffee': 21
}
crop['crop_no'] = crop['label'].map(crop_dict)
crop.drop('label', axis=1, inplace=True)
X=crop.drop('crop_no', axis=1)
y=crop['crop_no']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
dtc=DecisionTreeClassifier()
dtc.fit(X_train_scaled, y_train)
y_pred=dtc.predict(X_test_scaled)
acs=accuracy_score(y_test,y_pred)
print("Accuracy score",acs)
def crop_rec(N,P,K,temp,hum,ph,rain):
    features=np.array([[N,P,K,temp,hum,ph,rain]])
    transformed_features=scaler.transform(features)
    prediction=dtc.predict(transformed_features).reshape(1,-1)
    crop_dict = {
    0: 'rice', 1: 'maize', 2: 'chickpea', 3: 'kidneybeans', 4: 'pigeonpeas',
        5: 'mothbeans', 6: 'mungbean', 7: 'blackgram', 8: 'lentil', 9: 'pomegranate',
        10: 'banana', 11: 'mango', 12: 'grapes', 13: 'watermelon', 14: 'muskmelon',
        15: 'apple', 16: 'orange', 17: 'papaya', 18: 'coconut', 19: 'cotton',
        20: 'jute', 21: 'coffee'}
    crop=[crop_dict[i] for i in prediction[0]]
    return f"{crop} is a best crop to grow in the farm"
N=83
P=57
K=19
temp=25
hum=70
ph=6.8
rain=98
crop_rec(N,P,K,temp,hum,ph,rain)


#fertilizer
fertilizer = pd.read_csv(r"D:\sap proj\crop and fertilizer\dataset\Fertilizer Prediction.csv")
fert_dict = {
'Urea':1,
'DAP':2,
'14-35-14':3,
'28-28':4,
'17-17-17':5,
'20-20':6,
'10-26-26':7,
}
fertilizer['fert_no'] = fertilizer['Fertilizer Name'].map(fert_dict)
fertilizer.drop('Fertilizer Name',axis=1,inplace=True)
lb = LabelEncoder()
fertilizer["Soil Type"]=lb.fit_transform(fertilizer['Soil Type'])
fertilizer['Crop Type']=lb.fit_transform(fertilizer['Crop Type'])
# split the dataset into features and target
x = fertilizer.drop('fert_no',axis=1)
y = fertilizer['fert_no']
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)
# Scale the features using StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
# evaluate the model on the test set and print the accuracy
accuracy = model.score(x_test, y_test)
print(f"The accuracy of the model is: {accuracy*100:.2f}%")
def recommend_fertilizer(Temparature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorous):
    features = np.array([[Temparature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorous]])
    transformed_features = sc.transform(features)
    prediction = model.predict(transformed_features).reshape(1,-1)
    fert_dict = {1: 'Urea', 2: 'DAP', 3: '14-35-14', 4: '28-28', 5: '17-17-17', 6: '20-20', 7: '10-26-26'}
    fertilizer = [fert_dict[i] for i in prediction[0]]
    
    return f"{fertilizer} is a best fertilizer for the given conditions" 
# Given input values
Temparature = 26
Humidity = 0.5
Moisture = 0.6
Soil_Type = 2
Crop_Type = 3
Nitrogen = 10
Potassium = 15
Phosphorous = 6
    
# Use the recommendation function to get a prediction
recommend_fertilizer(Temparature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorous)