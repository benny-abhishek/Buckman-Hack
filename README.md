# InvestiGuide: Your Trusted Investment Companion
An Investment Decision Recommendation System
# 21PD03- Benny Abhishek M

# NOTE: 
* To run the app, run command: streamlit run app.py (or) python -m streamlit run app.py from the directory having the file 'app.py'
* The pages can be navigated through the sidebar in the app.
* The 'main.py' contains only the backend code. Do run the app.py file or visit the .py files in the 'pages' folder to view the organized code with straamlit.
* The images in the readme and documentation are saved using plotly for documentation purpose. [ALL THE OUTPUT SCREENSHOTS from the app are provided in the PPT- "Investment Recommendation System".]
* The 'Documentation.docx' file has the documentation.
* Anlysis using PowerBI is done and the .pbix is uploaded. Output screenshots are provided in PPT.

  # Libraries required
  Pandas, streamlit, sklearn, plotly

  # Data Preprocessing:
1)	The data is pre-processed by finding null values and outliers if any.
2)	The “Return Earned” column of the dataset is converted to a numeric score range from 0-4 and -1 for negative returns. This makes it easier for the upcoming tasks.

   # Data Exploration:
1)	Plots for analysing demographic distribution are done for Gender, Marital status, Age, City.
   
![image](https://github.com/benny-abhishek/Buckman-Hack/assets/121245162/3c45ba75-312b-4923-8191-10a94e92fc08)

2) Plots for employment details are done using Plotly.

![image](https://github.com/benny-abhishek/Buckman-Hack/assets/121245162/889187dc-7bb5-4896-bdd9-250b30d1237f)

3) Plots for investment behaviour insights including the percentage of household income invested, sources of awareness about investments, knowledge levels, influencers, risk levels, and reasons for investment are done.

![image](https://github.com/benny-abhishek/Buckman-Hack/assets/121245162/0471f232-3801-4f55-959b-b6af950cb7dc)

4) Also other simple EDA has been done.

   # Best Investment Decision Identification
1)	Since all the columns are categorical, The 
Chi-Square test is used to find the factor variables contributing to the best investment results. Here we have taken the significance level as 0.5

![image](https://github.com/benny-abhishek/Buckman-Hack/assets/121245162/42deb2a5-6296-46ce-89df-f5cefc36f3e3)

2) Then the demographic, employment, and behavioral characteristics which correlate successful investment outcomes is found by calculating the average returns for each categorical value under every column.
(The average can be found because the Return Earned column is transformed to a Numerical Score) . These results are also plotted for easy visualization

![image](https://github.com/benny-abhishek/Buckman-Hack/assets/121245162/c668def8-5e92-4671-85eb-75ddfded6c6e)

![image](https://github.com/benny-abhishek/Buckman-Hack/assets/121245162/b057192e-d203-4108-bee1-25c3f3c026ed)
# Note: Refer Presentation for all plots

# Model Selection & Evaluation
1)Different models like DecisionTree, Random Forest, Multi-layer perceptron(Neural network) and Support Vector Machine was applied to the categorical data and the performance was measure using performance metrics like Accuracy, precision, recall and F1-Score and it was found to give a reasonable performance even for a small dataset.(OUTPUT SCREENSHOT IN PPT)
2)Finally after tuning and validation, the best model is chosen to predict the returns based on inputs from the user for all the input features.

![image](https://github.com/benny-abhishek/Buckman-Hack/assets/121245162/7f1d4207-b2e3-4b01-81c1-4b94923b4e0f)

![image](https://github.com/benny-abhishek/Buckman-Hack/assets/121245162/264e5a2c-d9ce-4f09-9bd5-ccdd1d41c1dd)

# POWERBI ANALYSIS

![image](https://github.com/benny-abhishek/Buckman-Hack/assets/121245162/8c14051d-debc-47f6-abe2-97751aaadd7c)

# PS: REFER PPT FOR ALL PLOTS & INFERENCE














      
