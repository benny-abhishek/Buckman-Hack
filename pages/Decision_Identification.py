import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency

# Read data
df = pd.read_excel("Sample Data for shortlisting.xlsx")
st.title(":blue[Best Investment Decision Identification]")
# Define function to extract numeric value
def extract_numeric_value(value):
    if value == "Negative Return":
        return -1
    elif value == "1 to 4":
        return 1
    elif value == "5 to 6":
        return 2
    elif value == "7 to 9":
        return 3
    elif value == "10 to 12":
        return 4

# Apply function to 'Return Earned' column
df["Return Earned"] = df["Return Earned"].apply(extract_numeric_value)

# Define categorical columns
categorical_columns = ['City', 'Gender', 'Marital Status', 'Age', 'Education', 'Role', 
                       'Number of investors in family', 'Household Income', 'Percentage of Investment', 
                       'Source of Awareness about Investment', 'Knowledge level about different investment product', 
                       'Knowledge level about sharemarket', 'Knowledge about Govt. Schemes', 'Investment Influencer', 
                       'Investment Experience', 'Risk Level', 'Reason for Investment']

demographic_columns = ['City', 'Gender', 'Marital Status', 'Age', 'Education']
employment_columns = ['Role', 'Number of investors in family', 'Household Income']
behavioral_columns = ['Percentage of Investment', 'Source of Awareness about Investment',
                      'Knowledge level about different investment product',
                      'Knowledge level about sharemarket', 'Knowledge about Govt. Schemes',
                      'Investment Influencer', 'Investment Experience', 'Risk Level']
other_columns = ['Reason for Investment']

target_variable = 'Return Earned'

# Perform chi-square test for each feature
data_chi2 = df.drop(columns=['S. No.', 'Return Earned'])
p_values = []
for column in data_chi2.columns:
    contingency_table = pd.crosstab(data_chi2[column], df['Return Earned'])
    _, p, _, _ = chi2_contingency(contingency_table)
    p_values.append(p)

# Plot p-values using Plotly
p_values_fig = go.Figure(go.Bar(
    x=p_values,
    y=data_chi2.columns,
    orientation='h',
    marker_color='skyblue'
))
p_values_fig.update_layout(
    title='p-values from Chi-square Test for Predicting Return Earned',
    xaxis_title='p-value',
    yaxis_title='Feature',
    bargap=0.2,
    plot_bgcolor='rgba(0,0,0,0)',
    height=600,
    width=900
)

# Display plot in Streamlit
st.subheader("Chi-Square Test to find factors contributing to decision")
st.plotly_chart(p_values_fig)
st.markdown("---")

# Group data by each categorical column and calculate average return earned
data = df.copy()
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

average_returns_by_category = {}
decoded_category_names = {}
for column in categorical_columns:
    grouped_data = data.groupby(column)[target_variable].mean()
    average_returns_by_category[column] = grouped_data
    decoded_category_names[column] = [label_encoders[column].inverse_transform([index])[0] for index in grouped_data.index]

# Display categorical names and plots in a table format using Plotly
st.subheader("Plots to show how demographic, employment, and behavioral characteristics correlate with successful investment outcomes (computing average returns)")
columns_per_row = 2
num_rows = (len(categorical_columns) + columns_per_row - 1) // columns_per_row
for i in range(num_rows):
    cols = st.columns(columns_per_row)
    for j in range(columns_per_row):
        index = i * columns_per_row + j
        if index < len(categorical_columns):
            column = categorical_columns[index]
            cols[j].write(f"\n**{column}**")
            sorted_categories = average_returns_by_category[column].sort_values(ascending=False)
            sorted_names = [decoded_category_names[column][index] for index in sorted_categories.index]
            df_cat = pd.DataFrame({'Category': sorted_names, 'Average Return': sorted_categories}).reset_index()
            # Plot the average return using Plotly
            fig = go.Figure(go.Bar(
                x=df_cat['Average Return'],
                y=df_cat['Category'],
                orientation='h',
                marker_color='lightgreen' if j % 2 == 0 else 'lightblue'
            ))
            fig.update_layout(
                title=f'',
                xaxis_title='Average Return',
                yaxis_title='Category',
                bargap=0.2,
                plot_bgcolor='rgba(0,0,0,0)',
                height=400,
                width=500,
                margin=dict(l=50, r=50, t=50, b=50)  # Add margin for spacing
            )
            cols[j].plotly_chart(fig)
st.markdown("---")
