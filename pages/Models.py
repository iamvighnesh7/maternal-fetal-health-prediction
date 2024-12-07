# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings 
warnings.filterwarnings('ignore')

#Gautami
df = pd.read_csv(r"C:\Users\Gautami\Desktop\maternal and fetal health\fetal_health_gautami.csv")
# Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Splitting data into features (X) and target (y)
X = df.drop(columns=["fetal_health"])
y = df["fetal_health"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models to evaluate
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}


# Evaluating models and storing results
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy


# Sorting models by accuracy
sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

# Streamlit title and description
st.title("Fetal Health Prediction")
st.write("## Model Comparison by Accuracy")
st.write("This chart compares the accuracy of different models used for fetal health prediction.")

# Plotting the chart
fig, ax = plt.subplots(figsize=(10, 6))

# Create the bar plot
sns.barplot(x=list(sorted_results.keys()), y=list(sorted_results.values()), palette="viridis", ax=ax)

# Add titles and labels
ax.set_title("Model Comparison by Accuracy", fontsize=16)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_xlabel("Model", fontsize=12)
ax.set_xticklabels(list(sorted_results.keys()), rotation=45, fontsize=10)
ax.set_ylim(0, 1)  # Set y-axis from 0 to 1

# Add accuracy values on top of each bar
for i, value in enumerate(sorted_results.values()):
    ax.text(i, value + 0.02, f"{value:.2f}", ha='center', va='bottom', fontsize=10, color='black')

# Adjust layout
fig.tight_layout()

# Display the chart in Streamlit
st.pyplot(fig)

# Instantiate and fit the Random Forest model
best_model = RandomForestClassifier(random_state=42)
best_model.fit(X_train_scaled, y_train)

# Generating predictions
y_pred_rf = best_model.predict(X_test_scaled)

# Checking accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.2%}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generating the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)

# Generate the classification report
print("Classification Report - Random Forest")
print(classification_report(y_test, y_pred_rf, target_names=["Class 1", "Class 2", "Class 3"]))

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_scaled, y)
# Extract feature importance
feature_importance = rf_model.feature_importances_
features = X.columns

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importance}).sort_values(by="Importance", ascending=False)

# Streamlit UI
st.title("Random Forest Feature Importance")

# Display the feature importance DataFrame
st.subheader("Feature Importance Scores")
st.write(importance_df)

# Plot the top 10 features
top_features = importance_df.head(10)

# Create a bar plot for top 10 feature importances
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=top_features, palette="coolwarm", ax=ax)
ax.set_title("Top 10 Feature Importances - Random Forest", fontsize=16)
ax.set_xlabel("Importance", fontsize=12)
ax.set_ylabel("Features", fontsize=12)
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)


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

#Anurag
df = pd.read_csv(r'C:\Users\Gautami\Desktop\maternal and fetal health\baby-weights-dataset_1.csv')
df_copy_1 = df.copy()
cols_to_remove = ['ID', 'MARITAL', 'FAGE', 'FEDUC', 'BDEAD', 'ACLUNG', 
                  'RENAL', 'RACEMOM', 'RACEDAD', 'HISPMOM', 'HISPDAD', 'TERMS']
df_copy_1 = df_copy_1.drop(columns=cols_to_remove)
df_copy_1.dropna(inplace=True)
df_copy_1.drop(index=df_copy_1[df_copy_1['SEX']==9].index,inplace=True)
df_copy_1['Gestation_Range'] = pd.cut(df_copy_1['WEEKS'], bins=[20, 30, 35, 37, 40, 45],
                                      labels=['Very Preterm', 'Moderate Preterm', 'Late Preterm', 'Full Term', 'Post Term'])
df_copy_1['Weight_Gain_Category'] = pd.cut(df_copy_1['GAINED'], bins=[0, 15, 25, 35, 50, 100],
                                           labels=['Very Low', 'Low', 'Normal', 'High', 'Very High'])
df_copy_1.drop(columns=['WEEKS', 'GAINED','LOUTCOME'], inplace=True)
df_sample = df_copy_1.sample(n=50000, random_state=42)
def replace_outliers_with_whiskers(column):
    Q1 = column.quantile(0.25)  
    Q3 = column.quantile(0.75)  
    IQR = Q3 - Q1           
    lower_whisker = Q1 - 1.5 * IQR
    upper_whisker = Q3 + 1.5 * IQR
    
    # Replace values below or above the whiskers
    column = column.clip(lower=lower_whisker, upper=upper_whisker)
    return column

columns_to_replace = ['MAGE'] 
for column in columns_to_replace:
    df_sample[column] = replace_outliers_with_whiskers(df_sample[column])
def winsorize_column(column, limits=(0.03, 0.03)):  
    return winsorize(column, limits=limits)

columns_to_winsorize = ['TOTALP']  
for column in columns_to_winsorize:
    df_sample[column] = winsorize_column(df_sample[column])

df = df_sample.copy()
X = df.drop(columns=['BWEIGHT']) 
y = df['BWEIGHT']       
categorical_cols = X.select_dtypes(include=['category']).columns
encoder = LabelEncoder()
for col in categorical_cols:
    X[col] = encoder.fit_transform(X[col])
y = pd.to_numeric(y, errors='coerce')
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

f_test = SelectKBest(score_func=f_classif, k='all')  
f_test.fit(X[numerical_cols], y)
anova_scores = pd.DataFrame({'Feature': numerical_cols, 'F-Score': f_test.scores_})
if len(categorical_cols) > 0:
    chi2_test = SelectKBest(score_func=chi2, k='all')
    chi2_test.fit(X[categorical_cols], y.astype('int')) 
    chi2_scores = pd.DataFrame({'Feature': categorical_cols, 'Chi-Square': chi2_test.scores_})
else:
    chi2_scores = pd.DataFrame({'Feature': [], 'Chi-Square': []})

feature_scores = pd.concat([anova_scores, chi2_scores], axis=0, ignore_index=True)


rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X, y)

rf_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit UI
st.title("Baby's Weight Analysis")
st.title("Feature Importance Analysis")

# Create subplots for displaying feature importance plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Chi-Square Feature Scores Plot
feature_scores_sorted_chi = feature_scores.sort_values(by='Chi-Square', ascending=False).head(2)
sns.barplot(x='Chi-Square', y='Feature', data=feature_scores_sorted_chi, palette='magma', ax=axes[0])
axes[0].set_title('Top Features by Chi-Square', fontsize=14)
axes[0].set_xlabel('Chi-Square', fontsize=12)
axes[0].set_ylabel('Features', fontsize=12)

# Plot 2: F-Score Feature Scores Plot
feature_scores_sorted_f = feature_scores.sort_values(by='F-Score', ascending=False).head(5)
sns.barplot(x='F-Score', y='Feature', data=feature_scores_sorted_f, palette='magma', ax=axes[1])
axes[1].set_title('Top Features by F-Score', fontsize=14)
axes[1].set_xlabel('F-Score', fontsize=12)
axes[1].set_ylabel('Features', fontsize=12)

# Plot 3: Random Forest Feature Importances Plot
rf_importances_sorted = rf_importances.sort_values(by='Importance', ascending=False).head(5)
sns.barplot(x='Importance', y='Feature', data=rf_importances_sorted, palette='viridis', ax=axes[2])
axes[2].set_title('Top Features by Random Forest Importance', fontsize=14)
axes[2].set_xlabel('Importance Score', fontsize=12)
axes[2].set_ylabel('Features', fontsize=12)

# Adjust layout for better display
plt.tight_layout()

# Display plots in Streamlit
st.pyplot(fig)

# Accuracy scores for different models
accuracy_scores = {
    'Naive Bayes': 0.8079038500712976,
    'Stacking Classifier': 0.5102872275412508,
    'Random Forest': 0.4737217355876961,
    'Voting Classifier': 0.8499694438785903
}

# Create a figure for the plot
plt.figure(figsize=(10, 6))
plt.bar(accuracy_scores.keys(), accuracy_scores.values(), color=['skyblue', 'salmon', 'lightgreen', 'orange'])

# Labeling the plot
plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Accuracies Comparison', fontsize=14)
plt.xticks(rotation=45, fontsize=10)

# Tight layout for better spacing
plt.tight_layout()

# Display the plot in Streamlit
st.title("Model Accuracy Comparison")
st.pyplot(plt)


#Nandini
data = pd.read_csv(r'C:\Users\Gautami\Desktop\maternal and fetal health\Expanded_Maternal_Health_Risk_Data.csv')
data.drop(columns=['SystolicBP'],inplace=True)
data.drop(index=data[data['HeartRate'] == 7].index, inplace=True)

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
    n_estimators=400,
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

# Create a figure for the plot
plt.figure(figsize=(10, 6))
plt.bar(accuracy_maternal.keys(), accuracy_maternal.values())

# Labeling the plot
plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Accuracies Comparison', fontsize=14)
plt.xticks(rotation=45, fontsize=10)

# Tight layout for better spacing
plt.tight_layout()

# Display the plot in Streamlit
st.title("Maternal Health")
st.title("Model Accuracy Comparison")
st.pyplot(plt)  # Display the plot on Streamlit








