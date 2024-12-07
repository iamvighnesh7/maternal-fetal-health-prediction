import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Main dashboard theme
st.set_page_config(
    page_title="Healthcare Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Healthcare Analytics Dashboard")
st.sidebar.title("Navigation")
st.markdown("""
This dashboard provides insights into healthcare data through various visualizations. Use the sidebar to navigate between pages.
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

