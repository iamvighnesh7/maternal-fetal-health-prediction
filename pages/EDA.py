# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.max_columns = None


#fetal_health
df1 = pd.read_csv(r"C:\Users\Gautami\Desktop\maternal and fetal health\fetal_health_gautami.csv")
# Page setup
st.title("Top Correlations with Fetal Health")
st.markdown("""
Understanding which factors are most strongly correlated with fetal health can help healthcare providers focus on critical metrics.
""")
# Calculate the correlation matrix
correlation_matrix = df1.corr()
correlation_with_target = (
    correlation_matrix["fetal_health"]
    .drop("fetal_health")
    .sort_values(key=abs, ascending=False)
)

# Create the visualization
fig = px.bar(
    correlation_with_target.head(5),
    text=correlation_with_target.head(5).round(2),
    title="Top 5 Correlations with Fetal Health",
)
fig.update_layout(
    width=1000,
    height=500,
    margin=dict(l=10, r=10, t=30, b=100),
    xaxis_tickangle=-45,
)
fig.update_traces(textposition="inside")

# Display the visualization
st.plotly_chart(fig, use_container_width=True)

# Summary
st.subheader("Summary")
st.markdown("""
- **Positive Correlations**:  
  1. Abnormal Short-Term Variability (+0.47)  
  2. Percentage of Time with Abnormal Long-Term Variability (+0.43)  

- **Negative Correlations**:  
  1. Histogram Mode (-0.25)  
  2. Accelerations (-0.25)  
  3. Mean Value of Long-Term Variability (-0.24)
""")

# Page setup
st.title("Violin Plots: Important Features by Fetal Health Class")
st.markdown("""
Violin plots provide a detailed view of the distribution of important features across different fetal health classes.
""")

correlation_matrix = df1.corr()
# Correlation of features with the target variable
correlation_with_target = correlation_matrix["fetal_health"].drop("fetal_health").sort_values(key=abs, ascending=False)

# Select top correlated features
important_features = correlation_with_target.head(5).index.tolist()

# Display important features and their correlation values
print(important_features, correlation_with_target[important_features])
# Create the violin plots
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(len(important_features), 1, figsize=(14, len(important_features) * 4), constrained_layout=True)

for i, col in enumerate(important_features):
    sns.violinplot(
        x="fetal_health", 
        y=col, 
        data=df1, 
        palette="viridis", 
        ax=axes[i]
    )
    axes[i].set_title(
        f"Violin Plot of {col} by Fetal Health Class", 
        fontsize=14, 
        fontweight='bold', 
        color='darkblue'
    )
    axes[i].set_xlabel("Fetal Health Class", fontsize=12)
    axes[i].set_ylabel(col, fontsize=12)
    axes[i].grid(visible=True, which='major', linestyle='--', linewidth=0.7, alpha=0.7)

# Display the plot in Streamlit
st.pyplot(fig)
# Analysis Summary
st.subheader("Interpretation of Violin Plots")
st.markdown("""
### **1. Abnormal Short-Term Variability**
- **Class 1**: Higher concentration around lower values with a wide spread.
- **Class 2**: Uniform distribution.
- **Class 3**: Peaks at intermediate values, indicating moderate variability.

### **2. Percentage of Time with Abnormal Long-Term Variability**
- **Class 1**: Concentrated near zero, indicating minimal abnormality.
- **Class 2**: Broader spread with higher values than Class 1.
- **Class 3**: Peaks at higher percentages, suggesting this feature is significant.

### **3. Histogram Mode**
- **Class 1**: Narrow distribution centered around intermediate values.
- **Class 2**: Broader distribution but similar center.
- **Class 3**: Wider spread with frequent intermediate values.

### **4. Accelerations**
- **Class 1**: High peak at very low values, suggesting low accelerations dominate.
- **Class 2**: Extremely narrow distribution with minimal accelerations.
- **Class 3**: Slightly broader spread than Class 2, still centered at low values.

### **5. Mean Value of Long-Term Variability**
- **Class 1**: Concentrated at lower values with a narrow spread.
- **Class 2**: Slightly wider spread with intermediate values.
- **Class 3**: Widest spread with peaks at higher values.
""")

# Page setup
st.title("Density Plots: Feature Distributions by Fetal Health")
st.markdown("""
Density plots illustrate the distribution of key features, segmented by fetal health classes. These plots help in identifying overlaps and separations between health categories.
""")

# Create density plots
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots((len(important_features) + 1) // 2, 2, figsize=(18, len(important_features) * 3), constrained_layout=True)
axes = axes.flatten()

palette = "viridis"
for i, feature in enumerate(important_features):
    sns.kdeplot(
        data=df1,
        x=feature,
        hue="fetal_health",
        fill=True,
        common_norm=False,
        palette=palette,
        alpha=0.7,
        ax=axes[i]
    )
    axes[i].set_title(
        f"Density Plot of {feature} by Fetal Health",
        fontsize=16,
        fontweight="bold",
        color="black"
    )
    axes[i].set_xlabel(feature, fontsize=14)
    axes[i].set_ylabel("Density", fontsize=14)
    axes[i].legend(
        title="Fetal Health",
        fontsize=12,
        title_fontsize="13",
        loc="upper right",
        fancybox=True,
        framealpha=0.5
    )

# Hide any unused subplot axes
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

# Display the plots in Streamlit
st.pyplot(fig)

# Analysis Summary
st.subheader("Interpretation of Density Plots")
st.markdown("""
### **1. Abnormal Short-Term Variability**
- **Class 1**: Peaks at ~40–60 (lower variability).
- **Class 2**: Shifts slightly higher.
- **Class 3**: Peaks at ~60–80 (higher variability).

### **2. Percentage of Time with Abnormal Long-Term Variability**
- **Class 1**: Concentrated near zero.
- **Class 2**: Broader, with density around ~20–60.
- **Class 3**: Higher values (~40–80) dominate.

### **3. Histogram Mode**
- **Class 1**: Narrow peak at ~110–130.
- **Class 2**: Slightly broader, same range.
- **Class 3**: Wider spread, peaks ~130–150.

### **4. Accelerations**
- **Class 1**: High density near 0–0.005.
- **Class 2**: Similar, even lower density.
- **Class 3**: Broader, still concentrated at low values.

### **5. Mean Value of Long-Term Variability**
- **Class 1**: Peaks at ~5–10.
- **Class 2**: Slightly higher than Class 1.
- **Class 3**: Peaks at ~10–20 (greater variability).
""")



#Anurag
st.title("Birth Weight by Gestation Range")
st.markdown("""
This visualization highlights the relationship between gestation range and average birth weight, including the standard deviation for each range. This helps to understand how gestational age impacts fetal development.
""")
df2 = pd.read_csv(r'C:\Users\Gautami\Desktop\maternal and fetal health\baby-weights-dataset_1.csv')

df_copy_1 = df2.copy()
cols_to_remove = ['ID', 'MARITAL', 'FAGE', 'FEDUC', 'BDEAD', 'ACLUNG', 
                  'RENAL', 'RACEMOM', 'RACEDAD', 'HISPMOM', 'HISPDAD', 'TERMS']
df_copy_1 = df_copy_1.drop(columns=cols_to_remove)
df_copy_1.dropna(inplace=True)
df_copy_1.drop(index=df_copy_1[df_copy_1['SEX']==9].index,inplace=True)

sex_grouped = df_copy_1.groupby('SEX')['BWEIGHT'].mean().reset_index()

# Create a new gestation range column
df_copy_1['Gestation_Range'] = pd.cut(
    df_copy_1['WEEKS'], 
    bins=[20, 30, 35, 37, 40, 45], 
    labels=['Very Preterm', 'Moderate Preterm', 'Late Preterm', 'Full Term', 'Post Term']
)

# Calculate statistics
gestation_stats = df_copy_1.groupby('Gestation_Range')['BWEIGHT'].agg(['mean', 'std']).reset_index()

# Create the bar chart
fig = px.bar(
    gestation_stats,
    x='Gestation_Range',
    y='mean',
    error_y='std',
    labels={'Gestation_Range': 'Gestation Range', 'mean': 'Mean Birth Weight (lbs)'},
    title='Birth Weight by Gestation Range (Mean and Std Dev)',
)

# Display the visualization
st.plotly_chart(fig, use_container_width=True)
# Interpretation
st.write("### Interpretation")
st.write("Birth weight increases steadily with longer gestational periods, with post-term babies having the highest mean weight.")

# Page setup
st.title("Probability of Low Birth Weight by Maternal Weight Gain")
st.markdown("""
This chart explores the relationship between maternal weight gain during pregnancy and the probability of delivering a baby with low birth weight. It highlights how appropriate weight gain impacts fetal health.
""")
# Define weight gain categories
df_copy_1['Weight_Gain_Category'] = pd.cut(
    df_copy_1['GAINED'], 
    bins=[0, 15, 25, 35, 50, 100],
    labels=['Very Low', 'Low', 'Normal', 'High', 'Very High']
)

# Define low birth weight threshold and calculate probabilities
low_birth_weight = 5.5  # lbs
weight_gain_stats = df_copy_1.groupby('Weight_Gain_Category').apply(
    lambda x: (x['BWEIGHT'] < low_birth_weight).mean()
).reset_index()

weight_gain_stats.columns = ['Weight_Gain_Category', 'Low_Birth_Weight_Probability']

# Create the bar chart
fig = px.bar(
    weight_gain_stats,
    x='Weight_Gain_Category',
    y='Low_Birth_Weight_Probability',
    labels={
        'Weight_Gain_Category': 'Weight Gain Category',
        'Low_Birth_Weight_Probability': 'Probability of Low Birth Weight'
    },
    title='Probability of Low Birth Weight by Maternal Weight Gain'
)

# Display the visualization
st.plotly_chart(fig, use_container_width=True)
# Interpretation
st.write("### Interpretation")
st.write("""The probability of low birth weight tends to be higher in mothers with very low or very high weight gain during pregnancy. Maintaining a "Normal" weight gain reduces the likelihood of low birth weight.
""")

# Page setup
st.title("Impact of Maternal Habits on Baby's Birth Weight")
st.markdown("""
This section visualizes the relationship between maternal habits like cigarette and alcohol consumption during pregnancy and the baby's birth weight. These scatter plots include trendlines to indicate the correlation strength and direction.
""")
# Cigarette Consumption vs. Baby's Birth Weight
st.subheader("Cigarette Consumption vs. Birth Weight")
fig_cigarettes = px.scatter(
    df_copy_1,
    x='CIGNUM',
    y='BWEIGHT',
    trendline='ols',
    labels={'CIGNUM': 'Cigarette Consumption (Count)', 'BWEIGHT': 'Birth Weight (lbs)'},
    title='Cigarette Consumption vs. Birth Weight'
)
st.plotly_chart(fig_cigarettes, use_container_width=True)

st.write("### Interpretation")
st.markdown(""" Higher cigarette consumption during pregnancy is associated with lower birth weights. This suggests a negative impact of smoking on fetal health.
""")

# Alcohol Consumption vs. Baby's Birth Weight
st.subheader("Alcohol Consumption vs. Birth Weight")
fig_alcohol = px.scatter(
    df_copy_1,
    x='DRINKNUM',
    y='BWEIGHT',
    trendline='ols',
    labels={'DRINKNUM': 'Alcohol Consumption (Count)', 'BWEIGHT': 'Birth Weight (lbs)'},
    title='Alcohol Consumption vs. Birth Weight'
)
st.plotly_chart(fig_alcohol, use_container_width=True)

st.write("### Interpretation")
st.markdown(""" Similarly, higher alcohol consumption during pregnancy shows a trend toward lower birth weights, emphasizing the risks associated with alcohol use during pregnancy.
""")

# Nandini
data = pd.read_csv(r'C:\Users\Gautami\Desktop\maternal and fetal health\Expanded_Maternal_Health_Risk_Data.csv')
# Page setup
st.title("Distribution of Numerical Features")
st.markdown("""
This section visualizes the distribution of numerical features in the dataset using Kernel Density Estimation (KDE) plots. These plots help understand the distribution and density of values for each numerical feature.
""")
# Get the numerical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns

# Create subplots for each numerical column
st.subheader("Distribution of Numerical Columns")
fig, axes = plt.subplots(len(numerical_cols), 1, figsize=(15, len(numerical_cols) * 3))

# Plot KDE for each numerical column
sns.set(style="darkgrid")
for i, col in enumerate(numerical_cols):
    sns.kdeplot(data=data, x=col, shade=True, color='teal', ax=axes[i])
    axes[i].set_title(f'Distribution of {col}', fontsize=14)
    axes[i].set_xlabel(col, fontsize=12)
    axes[i].set_ylabel('Density')

plt.tight_layout()
st.pyplot(fig)

st.write("### Interpretation")
st.write("""
- **Age**: Shows the age distribution across the population.
- **SystolicBP**: Displays the range of systolic blood pressure values.
- **DiastolicBP**: Represents diastolic blood pressure distribution.
- **BS (Blood Sugar)**: Indicates the variability in blood sugar levels.
- **BodyTemp**: Captures the typical body temperature range.
- **HeartRate**: Highlights the spread of heart rate values.

""")

# Page setup
st.title("Density Plots of Features by Risk Level")
st.markdown("""
This section visualizes the distribution of various features based on different risk levels. Each plot shows the density distribution for numerical columns, grouped by their respective 'RiskLevel'.
""")
col = data.columns
col = col[:-1]  # Exclude last column

# Create subplots for each numerical column with risk level
st.subheader("Density Plot by Risk Level")
fig, axes = plt.subplots(3, 2, figsize=(18, 15))

# Plot KDE for each feature grouped by 'RiskLevel'
sns.set(style="darkgrid")
for i, j in enumerate(col, 1):
    ax = axes[(i-1)//2, (i-1)%2]  # Adjust the position of each subplot
    sns.kdeplot(data=data, x=j, hue='RiskLevel', fill=True, common_norm=False, palette='viridis', alpha=0.7, ax=ax)
    ax.set_title(f'Density Plot of {j} by Risk Level', fontsize=14, fontweight='bold', color='black')
    ax.set_xlabel(j, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(title='Risk Level', fontsize=10, title_fontsize='11', loc='upper right', fancybox=True, framealpha=0.5)

plt.tight_layout(h_pad=3)
st.pyplot(fig)
st.write("### Interpretation")
st.write("""
- **Age**: Risk levels are distributed across varying age groups, indicating trends in risk accumulation with age.
- **Blood Pressure**: Both systolic and diastolic blood pressure levels show clear variations between risk categories.
- **Blood Sugar**: Elevated blood sugar levels align more with higher risk groups.
- **Body Temperature & Heart Rate**: Subtle patterns in body temperature and heart rate highlight their potential role in risk stratification.
""")


#Vighnesh
data = pd.read_csv(r'C:\Users\Gautami\Desktop\maternal and fetal health\Percentage_Women_complication_Pregnancy_delivery.csv')
renamed_columns = {
    "Percentage of Women who had - Any Pregnancy complication": "Pregnancy_Complication",
    "Percentage of Women who had - Any Delivery complication": "Delivery_Complication",
    "Percentage of Women who had - Any Post-delivery complication": "PostDelivery_Complication",
    "Percentage of Women who had - Problem of vaginal discharge during last three months": "Vaginal_Discharge",
    "Percentage of Women who had - Menstrual related problems during last three months": "Menstrual_Problems"
}
data_renamed = data.rename(columns=renamed_columns)

# Page setup
st.title("Average Percentage of Women Facing Health Issues")
st.markdown("""
This chart visualizes the average percentage of women facing different health issues. The bar plot below highlights the top categories based on the mean percentage.
""")
# Select numerical columns
numerical_cols = data_renamed.select_dtypes('number').columns

# Calculate mean values for each category
mean_values = data_renamed[numerical_cols].mean().reset_index()
mean_values.columns = ['Category', 'Average Percentage']

# Plotting the bar plot
sns.set_style(style='darkgrid')
plt.figure(figsize=(15, 6))
sns.barplot(data=mean_values.head(3), x='Category', y='Average Percentage', palette='viridis')
plt.title('Average Percentage of Women Facing Health Issues')
plt.ylabel('Average Percentage')
plt.xlabel('Health Issue Categories')

# Rotate x-axis labels
plt.xticks(rotation=30, ha='right')
plt.tight_layout()

# Display plot in Streamlit
st.pyplot(plt)

st.write("### Interpretation")
st.write(
    """
    - The chart indicates that:
        1. Pregnancy complications have the highest occurrence among women.
        2. Delivery complications are moderately prevalent.
        3. Post-delivery complications are the least common.
    """
)
# Page setup
st.title("Top States and Districts with Highest Average Complications")
st.markdown("""
This chart visualizes the top 5 states and districts with the highest average complications based on maternal health data. The bar plots show the average complication percentage across different states and districts.
""")
# Select numerical columns
numerical_cols = data_renamed.select_dtypes('number').columns

# Calculate average complication for each row
data_renamed['Average Complication'] = data_renamed[numerical_cols].mean(axis=1)

# Get top 5 states and districts with highest average complications
top_states = data_renamed.groupby('States')['Average Complication'].mean().sort_values(ascending=False).head(5)
top_districts = data_renamed[['Districts', 'Average Complication']].sort_values(by='Average Complication', ascending=False).head(5)

# Create figure for subplots
plt.figure(figsize=(15, 6))

# Bar plot for top states
plt.subplot(1, 2, 1)
sns.barplot(x=top_states.values, y=top_states.index, palette='viridis')
plt.title('Top 5 States with Highest Average Complications', fontsize=15)
plt.xlabel('Average Complication (%)', fontsize=12)
plt.ylabel('States', fontsize=12)

# Bar plot for top districts
plt.subplot(1, 2, 2)
sns.barplot(x=top_districts['Average Complication'], y=top_districts['Districts'], palette='plasma')
plt.title('Top 5 Districts with Highest Average Complications', fontsize=15)
plt.xlabel('Average Complication (%)', fontsize=12)
plt.ylabel('Districts', fontsize=12)

# Adjust layout and display the plots
plt.tight_layout()

# Display plot in Streamlit
st.pyplot(plt)

st.write("### Interpretation")
st.write(
    """
    - **Top States**:
        - West Bengal has the highest average complication percentage.
        - Tripura, Himachal Pradesh, Andaman and Nicobar Islands, and Haryana follow in ranking.
    - **Top Districts**:
        - Hamirpur exhibits the highest complication rates among districts, followed by Kinnaur, Maldah, Pudukkottai, and Jalpaiguri.
    - The chart underscores the need for targeted interventions in these regions.
    """
)




