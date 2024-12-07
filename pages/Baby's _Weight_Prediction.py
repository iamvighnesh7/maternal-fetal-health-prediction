
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
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
df = pd.read_csv(r'C:\Users\Gautami\Desktop\maternal and fetal health\baby-weights-dataset_1.csv')


df = df.sample(n=2500, random_state=42)
# Select only the relevant columns for training
X = df[['MAGE', 'VISITS', 'ECLAMP', 'HYPERPR', 'TOTALP', 'WEEKS']]  # Make sure these match your column names
y = df['BWEIGHT']  # The target variable

# Replace NaN in numerical columns with the median
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())

# Preprocess the data (Scaling)
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Train the RandomForest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X, y)

# Save the model and scaler
import joblib
joblib.dump(rf_model, 'babies_model.pkl')  # Save the new model
joblib.dump(scaler, 'scaler_babies.pkl')  # Save the new scaler

import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('babies_model.pkl')
scaler = joblib.load('scaler_babies.pkl')

# Streamlit UI for input fields
st.title("Baby's Weight Prediction")
st.write("This app predicts a baby's weight based on maternal factors.")

# Input fields for user data (5 features only)
mother_age = st.number_input("Enter Mother's Age", min_value=10, max_value=50, value=25, step=1)
number_of_visits = st.number_input("Enter Number of Visits", min_value=1, max_value=20, value=5, step=1)
eclampsia = st.number_input("Enter Eclampsia Status (1 for Yes, 0 for No)", min_value=0, max_value=1, value=0, step=1)
hypertension = st.number_input("Enter Hypertension Status (1 for Yes, 0 for No)", min_value=0, max_value=1, value=0, step=1)
total_pregnancies = st.number_input("Enter Total Pregnancies", min_value=1, max_value=10, value=2, step=1)
total_weeks = st.number_input("Enter Total Weeks", min_value=1, max_value=40, value=2, step=1)

# Button to trigger prediction
if st.button("Predict Baby's Weight"):
    # Prepare input data
    input_data = np.array([[mother_age, number_of_visits, eclampsia, hypertension, total_pregnancies, total_weeks]])
    
    # Scale the input data using the saved scaler
    input_data_scaled = scaler.transform(input_data)
    
    # Predict using the saved model
    predicted_weight = model.predict(input_data_scaled)
    
    # Display the result
    st.success(f"The predicted baby's weight is: {predicted_weight[0]:.2f} lbs")

