import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.subplots as ps


df=pd.read_excel("Sample Data for shortlisting.xlsx")

# Gender distribution
gender_distribution = df['Gender'].value_counts().reset_index()
gender_distribution.columns = ['Gender', 'Count']
gender_fig = go.Bar(x=gender_distribution['Gender'], y=gender_distribution['Count'], name='Gender Distribution')

# Marital status distribution
marital_distribution = df['Marital Status'].value_counts().reset_index()
marital_distribution.columns = ['Marital Status', 'Count']
marital_fig = go.Bar(x=marital_distribution['Marital Status'], y=marital_distribution['Count'], name='Marital Status Distribution')

# Age distribution
age_fig = go.Histogram(x=df['Age'], nbinsx=20, name='Age Distribution')

# City distribution
city_distribution = df['City'].value_counts().reset_index()
city_distribution.columns = ['City', 'Count']
city_fig = go.Bar(x=city_distribution['City'], y=city_distribution['Count'], name='City Distribution')

# Create subplots
fig = ps.make_subplots(rows=2, cols=2, subplot_titles=("Gender Distribution", "Marital Status Distribution",
                                                      "Age Distribution", "City Distribution"))

fig.add_trace(gender_fig, row=1, col=1)
fig.add_trace(marital_fig, row=1, col=2)
fig.add_trace(age_fig, row=2, col=1)
fig.add_trace(city_fig, row=2, col=2)

# Update layout
fig.update_layout(height=800, width=1200, title_text="Demographic Distributions")
fig.update_xaxes(title_text="Gender", row=1, col=1)
fig.update_xaxes(title_text="Marital Status", row=1, col=2)
fig.update_xaxes(title_text="Age", row=2, col=1)
fig.update_xaxes(title_text="City", row=2, col=2)
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_yaxes(title_text="Count", row=1, col=2)
fig.update_yaxes(title_text="Count", row=2, col=1)
fig.update_yaxes(title_text="Count", row=2, col=2)

# Show plot
fig.show()

education_distribution = df['Education'].value_counts().reset_index()
education_distribution.columns = ['Education', 'Count']
education_fig = go.Bar(x=education_distribution['Education'], y=education_distribution['Count'], name='Education Distribution')

# Role distribution
role_distribution = df['Role'].value_counts().reset_index()
role_distribution.columns = ['Role', 'Count']
role_fig = go.Bar(x=role_distribution['Role'], y=role_distribution['Count'], name='Role Distribution')

# Income distribution
income_distribution = df['Household Income'].value_counts().reset_index()
income_distribution.columns = ['Household Income', 'Count']
income_fig = go.Bar(x=income_distribution['Household Income'], y=income_distribution['Count'], name='Income Distribution')

# Create subplots
fig = ps.make_subplots(rows=3, cols=1, subplot_titles=("Education Distribution", "Income Distribution",
                                                      "Role Distribution"))

fig.add_trace(education_fig, row=1, col=1)
fig.add_trace(role_fig, row=3, col=1)
fig.add_trace(income_fig, row=2, col=1)

# Update layout
fig.update_layout(height=1100, width=900, title_text="Exploration of Employment Details")
fig.update_xaxes(title_text="Education", row=1, col=1)
fig.update_xaxes(title_text="Role", row=3, col=1)
fig.update_xaxes(title_text="Household Income", row=2, col=1)
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_yaxes(title_text="Count", row=3, col=1)
fig.update_yaxes(title_text="Count", row=2, col=1)

# Show plot
fig.show()
"""
"""
description=df.describe()


fig = ps.make_subplots(rows=3, cols=2, subplot_titles=("Percentage of Household Income Invested",
                                                      "Sources of Awareness",
                                                      "Knowledge Levels",
                                                      "Influencers",
                                                      "Risk Levels",
                                                      "Reasons for Investment"))

# Percentage of Household Income Invested
fig.add_trace(go.Histogram(x=df['Percentage of Investment'], name='Household Income Invested'), row=1, col=1)

# Sources of Awareness about Investments
source_counts = df['Source of Awareness about Investment'].value_counts()
fig.add_trace(go.Bar(x=source_counts.index, y=source_counts.values, name='Sources of Awareness'), row=1, col=2)

# Knowledge Levels
knowledge_counts = df['Knowledge level about different investment product'].value_counts()
fig.add_trace(go.Bar(x=knowledge_counts.index, y=knowledge_counts.values, name='Knowledge Levels'), row=2, col=1)

# Influencers
influencer_counts = df['Investment Influencer'].value_counts()
fig.add_trace(go.Bar(x=influencer_counts.index, y=influencer_counts.values, name='Influencers'), row=2, col=2)

# Risk Levels
risk_counts = df['Risk Level'].value_counts()
fig.add_trace(go.Bar(x=risk_counts.index, y=risk_counts.values, name='Risk Levels'), row=3, col=1)

# Reasons for Investment
reason_counts = df['Reason for Investment'].value_counts()
fig.add_trace(go.Bar(x=reason_counts.index, y=reason_counts.values, name='Reasons for Investment'), row=3, col=2)

# Update layout
fig.update_layout(height=1000, width=1200, title_text="Investment Behavior Insights")

# Show plot
fig.show()

import re    
def extract_numeric_value1(value):
    if 'to' in value:
        # Extract numbers from the string using regular expression
        numbers = re.findall(r'\d+', value)
        # Convert the numbers to integers and calculate the midpoint
        midpoint = (int(numbers[0]) + int(numbers[1])) / 2
        # Scale the midpoint to a score between 0 and 10
        return (midpoint / 13) * 10
    elif value == 'Negative Return':
        return -1  # Assigning a negative value for 'Negative Return'
    elif value == 'More than 13':
        return 10  # Assuming 'More than 13' represents 10 on a scale of 0 to 10
    else:
        # Convert the value to an integer
        numeric_value = int(value.split()[0])
        # Scale the numeric value to a score between 0 and 10
        return (numeric_value / 13) * 10
    
def extract_numeric_value(value):
    if value=="Negative Return":
        return -1
    elif value== "1 to 4":
        return 1
    elif value== "5 to 6":
        return 2
    elif value== "7 to 9":
        return 3
    elif value== "10 to 12":
        return 4
    
    
df["Return Earned"]=df["Return Earned"].apply(extract_numeric_value)
    
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
data = df.copy()

categorical_columns = ['City', 'Gender', 'Marital Status', 'Age', 'Education', 'Role', 
                       'Number of investors in family', 'Household Income', 'Percentage of Investment', 
                       'Source of Awareness about Investment', 'Knowledge level about different investment product', 
                       'Knowledge level about sharemarket', 'Knowledge about Govt. Schemes', 'Investment Influencer', 
                       'Investment Experience', 'Risk Level', 'Reason for Investment']
target_variable = 'Return Earned'

from scipy.stats import chi2_contingency

#CHI SQUARE TEST
data_chi2 = df.drop(columns=['S. No.', 'Return Earned'])

# Perform chi-square test for each feature
p_values = []
for column in data_chi2.columns:
    contingency_table = pd.crosstab(data_chi2[column], df['Return Earned'])
    _, p, _, _ = chi2_contingency(contingency_table)
    p_values.append(p)

# Plot p-values
plt.figure(figsize=(14, 10))  # Increase figure height
plt.barh(data_chi2.columns, p_values, color='skyblue')
plt.xlabel('p-value')
plt.ylabel('Feature')
plt.title('p-values from Chi-square Test for Predicting Return Earned')
plt.axvline(x=0.5, color='red', linestyle='--', label='Significance level (0.5)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()  # Adjust layout to prevent cropping of labels
plt.show()


demographic_columns = ['City', 'Gender', 'Marital Status', 'Age', 'Education']
employment_columns = ['Role', 'Number of investors in family', 'Household Income']
behavioral_columns = ['Percentage of Investment', 'Source of Awareness about Investment',
                      'Knowledge level about different investment product',
                      'Knowledge level about sharemarket', 'Knowledge about Govt. Schemes',
                      'Investment Influencer', 'Investment Experience', 'Risk Level']
other_columns = ['Reason for Investment']

# Initialize label encoders for categorical columns
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Group data by each categorical column and calculate average return earned
average_returns_by_category = {}
decoded_category_names = {}
for column in categorical_columns:
    grouped_data = data.groupby(column)[target_variable].mean()
    average_returns_by_category[column] = grouped_data
    decoded_category_names[column] = [label_encoders[column].inverse_transform([index])[0] for index in grouped_data.index]

# Function to display category names with their average returns sorted by average return
def display_category_names_sorted(column):
    sorted_categories = average_returns_by_category[column].sort_values(ascending=False)
    print(f"\n{column}:")
    sorted_names = [decoded_category_names[column][index] for index in sorted_categories.index]
    for category_name, avg_return in zip(sorted_names, sorted_categories):
        print(f"{category_name}: {avg_return:.2f}")

# Display categories for each category sorted by average return
print("Demographic Characteristics:")
for column in demographic_columns:
    display_category_names_sorted(column)

print("\nEmployment Characteristics:")
for column in employment_columns:
    display_category_names_sorted(column)

print("\nBehavioral Characteristics:")
for column in behavioral_columns:
    display_category_names_sorted(column)

print("\nOther Characteristics:")
for column in other_columns:
    display_category_names_sorted(column)