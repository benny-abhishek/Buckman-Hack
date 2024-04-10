import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier


# Define options for each feature
options = {
    "City": ['New York', 'Seattle', 'San Francisco', 'Memphis', 'Houston'],
    "Gender": ['Men', 'Women'],
    "Marital Status": ['Never Married', 'Married'],
    "Age": ['Early Working', 'Elderly', 'Prime Working', 'Mature Working', 'Children'],
    "Education": ['Secondary', 'Middle', 'Teritary', 'Uneducated', 'Primary'],
    "Role": ['Marketing and Sales Executive', 'Advertising and Promotion Executive',
             'Training and Development Executive', 'Computer and Information System Executive', 'Top Executives'],
    "Number of investors in family": [2, 5, 3, 1, 4],
    "Household Income": ['US$ 2736 to US$ 8205', 'US$ 19146 to US$ 24615', 'US$ 13676 to US$ 19145',
                         'Above US$ 30086', 'US$ 24616 to US$ 30085', 'US$ 2735', 'US$ 8206 to US$ 13675'],
    "Percentage of Investment": ["Don't Want to Reveal", 'Above 26%', '16% to 20%', '21% to 25%', 'Upto 5%',
                                  '11% to 15%', '6% to 10%'],
    "Source of Awareness about Investment": ['Television', 'Workers', 'Family', 'Magazine', 'Others', 'Flash Board',
                                             'Friends'],
    "Knowledge level about different investment product": [7, 4, 5, 6, 3, 2, 10, 9, 1, 8],
    "Knowledge level about sharemarket": [9, 5, 3, 1, 10, 4, 2, 7, 6, 8],
    "Knowledge about Govt. Schemes": [4, 2, 3, 5, 1, 6],
    "Investment Influencer": ['Family Reference', 'Workers Reference', 'Friends Reference', 'Institutions',
                               'Others', 'Social Media', 'Intermediaries'],
    "Investment Experience": ['Less Than 1 Year', 'Above 9 Years', '7 Years to 9 Years',
                               '1 Year to 3 Years', '4 Years to 6 Years'],
    "Risk Level": ['Low', 'High', 'Medium'],
    "Reason for Investment": ['Tax', 'Others', 'Return', 'Fun and Exitement', 'Status', 'Inflation']
}

# Streamlit app
st.title(":orange[Recommendation System]")
st.title("Enter Input Data")

# Ask for input values and store them
input_values = {}
for feature, option_list in options.items():
    input_values[feature] = st.selectbox(f"Select {feature}", option_list)


govt_schemes = input_values["Knowledge about Govt. Schemes"]
knowledge_investment = input_values["Knowledge level about different investment product"]
percentage_of_investment = input_values["Percentage of Investment"]
household_income = input_values["Household Income"]
reason_for_investment = input_values["Reason for Investment"]



data = pd.read_excel("Sample Data for shortlisting.xlsx")
data = data[["Knowledge about Govt. Schemes", "Knowledge level about different investment product",
             "Percentage of Investment", "Household Income", "Return Earned", "Reason for Investment"]]

# Convert categorical columns to numerical using LabelEncoder
int_columns = data.select_dtypes(include='int').columns
data[int_columns] = data[int_columns].astype(str)

# Convert categorical columns to numerical using LabelEncoder
le_dict = {}
encodings_dict = {}

data_encoded = pd.DataFrame()

for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data_encoded[column] = le.fit_transform(data[column])
        le_dict[column] = le
        encodings_dict[column] = dict(zip(le.classes_, le.transform(le.classes_)))
    else:
        data_encoded[column] = data[column]


# Split data into features and target variable
X = data_encoded.drop(columns=["Return Earned"])
y = data_encoded["Return Earned"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Ensure consistency in columns between training and test data
X_test = X_test[X_train.columns]

# Define classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=10),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(max_iter=10),
    "Multilayer Perceptron": MLPClassifier(hidden_layer_sizes=(100,100,100,100,200),
                                           max_iter=100000, activation="relu")
}

# Train and evaluate classifiers
results = {"Classifier": [], "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": []}

for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results["Classifier"].append(clf_name)
    results["Accuracy"].append(accuracy)
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1 Score"].append(f1)

# Prepare test data
input_data = pd.DataFrame({
    "Knowledge about Govt. Schemes": [int(govt_schemes)],
    "Knowledge level about different investment product": [int(knowledge_investment)],
    "Percentage of Investment": [percentage_of_investment],
    "Household Income": [household_income],
    "Reason for Investment": [reason_for_investment]
})

int_columns = input_data.select_dtypes(include='int').columns
input_data[int_columns] = input_data[int_columns].astype(str)

# Encode categorical variables in input data using the LabelEncoder objects from le_dict
input_data_encoded = pd.DataFrame()

for column, le in le_dict.items():
    if column in input_data.columns:
        input_data_encoded[column] = le.transform(input_data[column])

predictions = {}
for clf_name, clf in classifiers.items():
    y_pred = clf.predict(input_data_encoded)
    predictions[clf_name] = int(y_pred[0])

# Decode predictions using encodings_dict
# Decode predictions for Return Earned column using encodings_dict
return_predictions = {}
for clf_name, pred in predictions.items():
    return_label = list(encodings_dict['Return Earned'].keys())[pred]
    return_predictions[clf_name] = return_label

# Display return predictions
st.markdown("---")
st.subheader(":red[Your result will be : ]"+str(return_predictions["Multilayer Perceptron"]))
