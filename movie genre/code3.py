import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Read and preprocess training data
train_file_path = 'train_data.txt'
train_data = []
with open(train_file_path, 'r', encoding='unicode_escape') as file:
    for line in file:
        line = line.strip() 
        if line:
            parts = line.split(':::')
            train_data.append(parts)

train_columns = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']
train_df = pd.DataFrame(train_data, columns=train_columns)
train_df['TEXT'] = train_df['TITLE'] + ' ' + train_df['DESCRIPTION']

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train = tfidf_vectorizer.fit_transform(train_df['TEXT'])
y_train = train_df['GENRE']

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train a Logistic Regression classifier

classifier = LogisticRegression(max_iter=1000)   
print('Training Model....')
classifier.fit(X_train_split, y_train_split) 

# Read and preprocess test data
test_file_path = 'test_data.txt'
test_data = []
with open(test_file_path, 'r', encoding='unicode_escape') as file:
    for line in file:
        line = line.strip()
        if line:
            parts = line.split(':::')
            test_data.append(parts)

test_columns = ['ID', 'TITLE', 'DESCRIPTION']
test_df = pd.DataFrame(test_data, columns=test_columns)
test_df['TEXT'] = test_df['TITLE'] + ' ' + test_df['DESCRIPTION']

X_test = tfidf_vectorizer.transform(test_df['TEXT']) 

print('Predicting Data....')
predictions = classifier.predict(X_test) 

for prediction in predictions:
    print(prediction)

# Calculate accuracy on the test set
# Replace 'actual_labels_for_test_data' with actual ground truth labels for the test data
actual_labels_for_test_data = test_df['GENRE']  # Assuming the 'GENRE' column contains the ground truth labels
accuracy = classifier.score(X_test, actual_labels_for_test_data)
print("Accuracy on the test set:", accuracy)
