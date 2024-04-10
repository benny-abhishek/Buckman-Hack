import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier

# Load the dataset
data = pd.read_excel("Sample Data for shortlisting.xlsx")

# Drop irrelevant columns for prediction
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
    "Multilayer Perceptron": MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 200),
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

# Display results in a table
results_df = pd.DataFrame(results)

# Display results in Streamlit
st.title(":blue[Model Selection & Evaluation]")
st.write(results_df)

# Display accuracy percentage for each model
st.title(":blue[Accuracy Percentage]")
for clf_name, accuracy in zip(results_df["Classifier"], results_df["Accuracy"]):
    st.write(f"{clf_name}: {accuracy * 100:.2f}%")
