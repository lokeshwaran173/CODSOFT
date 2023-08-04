import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Load the data
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v2', 'v1']]
df = df.rename(columns={'v2': 'messages', 'v1': 'label'})

# Check for null values
df.isnull().sum()

# Define stopwords
STOPWORDS = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^0-9a-zA-Z]', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = " ".join(word for word in text.split() if word not in STOPWORDS)  # Remove stopwords
    return text

# Clean the messages
df['clean_text'] = df['messages'].apply(clean_text)

# Prepare data for training
X = df['clean_text']
y = df['label']

# Create a TfidfVectorizer and LogisticRegression classifier pipeline
pipeline_lr = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Create a TfidfVectorizer and MultinomialNB classifier pipeline
pipeline_nb = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Create a TfidfVectorizer and SVC classifier pipeline
pipeline_svc = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', SVC())
])

# Create a TfidfVectorizer and RandomForestClassifier classifier pipeline
pipeline_rf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier())
])

# Function for classification
def classify(model, X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True, stratify=y)
    model.fit(x_train, y_train)
    
    print('Accuracy:', model.score(x_test, y_test) * 100)
    
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))

# Perform classification for each model
print("Logistic Regression:")
classify(pipeline_lr, X, y)

print("\nMultinomial Naive Bayes:")
classify(pipeline_nb, X, y)

print("\nSupport Vector Machine:")
classify(pipeline_svc, X, y)

print("\nRandom Forest Classifier:")
classify(pipeline_rf, X, y)
