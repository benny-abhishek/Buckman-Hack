import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as ps

# Read the data
df = pd.read_excel("Sample Data for shortlisting.xlsx")

st.title(":violet[Data Exploration]")
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

# Education distribution
education_distribution = df['Education'].value_counts().reset_index()
education_distribution.columns = ['Education', 'Count']
education_fig = go.Bar(x=education_distribution['Education'], y=education_distribution['Count'], name='Education Distribution')

# Role distribution
role_distribution = df['Role'].value_counts().reset_index()
role_distribution.columns = ['Role', 'Count']
role_fig = go.Bar(x=role_distribution['Role'], y=role_distribution['Count'], name='Role Distribution')

# Household Income distribution
income_distribution = df['Household Income'].value_counts().reset_index()
income_distribution.columns = ['Household Income', 'Count']
income_fig = go.Bar(x=income_distribution['Household Income'], y=income_distribution['Count'], name='Income Distribution')

# Create subplots for demographics
demographics_fig = ps.make_subplots(rows=2, cols=2, subplot_titles=("Gender Distribution", "Marital Status Distribution",
                                                                    "Age Distribution", "City Distribution"))
demographics_fig.add_trace(gender_fig, row=1, col=1)
demographics_fig.add_trace(marital_fig, row=1, col=2)
demographics_fig.add_trace(age_fig, row=2, col=1)
demographics_fig.add_trace(city_fig, row=2, col=2)

# Create subplots for employment details
employment_fig = ps.make_subplots(rows=3, cols=1, subplot_titles=("Education Distribution", "Income Distribution",
                                                                  "Role Distribution"), vertical_spacing=0.2)
employment_fig.add_trace(education_fig, row=1, col=1)
employment_fig.add_trace(role_fig, row=3, col=1)
employment_fig.add_trace(income_fig, row=2, col=1)

# Percentage of Household Income Invested
percentage_invested_fig = go.Histogram(x=df['Percentage of Investment'], name='Household Income Invested')

# Sources of Awareness about Investments
source_counts = df['Source of Awareness about Investment'].value_counts()
sources_fig = go.Bar(x=source_counts.index, y=source_counts.values, name='Sources of Awareness')

# Knowledge Levels
knowledge_counts = df['Knowledge level about different investment product'].value_counts()
knowledge_fig = go.Bar(x=knowledge_counts.index, y=knowledge_counts.values, name='Knowledge Levels')

# Influencers
influencer_counts = df['Investment Influencer'].value_counts()
influencers_fig = go.Bar(x=influencer_counts.index, y=influencer_counts.values, name='Influencers')

# Risk Levels
risk_counts = df['Risk Level'].value_counts()
risk_fig = go.Bar(x=risk_counts.index, y=risk_counts.values, name='Risk Levels')

# Reasons for Investment
reason_counts = df['Reason for Investment'].value_counts()
reasons_fig = go.Bar(x=reason_counts.index, y=reason_counts.values, name='Reasons for Investment')

# Create subplots for investment behavior insights
insights_fig = ps.make_subplots(rows=3, cols=2, subplot_titles=("Percentage of Household Income Invested",
                                                                 "Sources of Awareness",
                                                                 "Knowledge Levels",
                                                                 "Influencers",
                                                                 "Risk Levels",
                                                                 "Reasons for Investment"),vertical_spacing=0.2)
insights_fig.add_trace(percentage_invested_fig, row=1, col=1)
insights_fig.add_trace(sources_fig, row=1, col=2)
insights_fig.add_trace(knowledge_fig, row=2, col=1)
insights_fig.add_trace(influencers_fig, row=2, col=2)
insights_fig.add_trace(risk_fig, row=3, col=1)
insights_fig.add_trace(reasons_fig, row=3, col=2)

# Update layout for demographics plot
demographics_fig.update_layout(height=700, width=900, title_text="")

# Update layout for employment details plot
employment_fig.update_layout(height=800, width=900, title_text="")

# Update layout for investment behavior insights plot
insights_fig.update_layout(height=1000, width=1200, title_text="")

# Display in Streamlit
st.subheader("Demographic Distributions")
st.plotly_chart(demographics_fig, use_container_width=True)
st.markdown("---")

st.subheader("Exploration of Employment Details")
st.plotly_chart(employment_fig, use_container_width=True)
st.markdown("---")

st.subheader("Investment Behavior Insights")
st.plotly_chart(insights_fig, use_container_width=True)
st.markdown("---")

st.subheader("Description of Dataset")
description=df.describe()
st.write(description)
st.markdown("---")

