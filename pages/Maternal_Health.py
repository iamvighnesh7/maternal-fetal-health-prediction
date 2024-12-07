# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings 
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import f_classif, SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import boxcox
from scipy.stats.mstats import winsorize
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv(r'C:\Users\Gautami\Desktop\maternal and fetal health\Expanded_Maternal_Health_Risk_Data.csv')
data.drop(columns=['SystolicBP'],inplace=True)
data.drop(index=data[data['HeartRate'] == 7].index, inplace=True)

data = data.sample(n=2000, random_state=42)
X = data[["Age", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]]
y = data["RiskLevel"]
y = y.map({"low risk": 0, "mid risk": 1, "high risk": 2})
accuracy_maternal = {}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
lr_model = LogisticRegression(
    C=0.01,
    intercept_scaling=1,
    max_iter=100,
    solver='liblinear',
    tol=0.0001,
    multi_class='ovr'
)
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
accuracy_maternal['LogisticRegression']= accuracy_score(y_test, y_pred)





knn_model = KNeighborsClassifier(
    leaf_size=1,
    n_neighbors=10,
    p=2,
    weights='distance'
)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
accuracy_maternal['KNN']= accuracy_score(y_test, y_pred)




rf_model = RandomForestClassifier(
    criterion='gini',
    max_depth=30,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_maternal['RandomForest']= accuracy_score(y_test, y_pred)



catboost_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass',
    random_seed=42,
    verbose=False
)
catboost_model.fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)
accuracy_maternal['Catboost']= accuracy_score(y_test, y_pred)

import joblib

# Save the Random Forest model
joblib.dump(rf_model, 'maternal_model.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler_maternal.pkl')

print("Model and scaler saved successfully.")

import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('maternal_model.pkl')
scaler = joblib.load('scaler_maternal.pkl')

# Streamlit UI for input fields
st.title("Maternal Health Risk Prediction")
st.write("This app predicts the health risk level (low, mid, high) based on given inputs.")

# Input fields for user data
age = st.number_input("Enter Age", min_value=10, max_value=70, value=25, step=1)
diastolic_bp = st.number_input("Enter DiastolicBP (mmHg)", min_value=49, max_value=100, value=80, step=1)
bs = st.number_input("Enter Blood Sugar Level (mmol/L)", min_value=6.0, max_value=19.0, value=8.0, step=0.1)
body_temp = st.number_input("Enter Body Temperature (°C)", min_value=98.0, max_value=103.0, value=98.6, step=0.1)
heart_rate = st.number_input("Enter Heart Rate (bpm)", min_value=60, max_value=90, value=75, step=1)

# Button to trigger prediction
if st.button("Predict Risk Level"):
    # Prepare input data
    input_data = np.array([[age, diastolic_bp, bs, body_temp, heart_rate]])
    
    # Scale the input data using the saved scaler
    input_data_scaled = scaler.transform(input_data)
    
    # Predict using the saved model
    prediction = model.predict(input_data_scaled)
    
    # Mapping class labels to risk levels
    risk_mapping = {0: "Low Risk", 1: "Mid Risk", 2: "High Risk"}
    predicted_risk = risk_mapping[prediction[0]]
    
    # Show the result
    st.success(f"The predicted maternal health risk level is: {predicted_risk}")
