import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the credit card dataset
credit_card_data = pd.read_csv('creditcard.csv')

# Display basic information about the dataset
print(credit_card_data.info())

# Explore data statistics and missing values
print(credit_card_data.describe())
print(credit_card_data.isnull().sum())

# Explore data grouped by 'Class' column
print(credit_card_data.groupby('Class').mean())

# Separate legitimate and fraudulent transactions
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# Display shapes of legitimate and fraudulent data
print(legit.shape)
print(fraud.shape)

# Describe 'Amount' column for both classes
print(legit.Amount.describe())
print(fraud.Amount.describe())

# Sample legitimate transactions to balance the dataset
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Display class distribution in the new dataset
print(new_dataset['Class'].value_counts())

# Separate features (X) and target (Y)
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Initialize and train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Make predictions on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)

# Make predictions on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy)
