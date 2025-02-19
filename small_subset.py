import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load IMDb dataset 
df = pd.read_csv("C:/Users/Mario Rus/Desktop/Proiect Cercetare/IMDB-Dataset.csv")  

# Convert sentiment labels to numeric (positive = 1, negative = 0)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Select a balanced subset of 5,000 reviews (2,500 positive, 2,500 negative)
df_positive = df[df['sentiment'] == 1].sample(n=2500, random_state=42)
df_negative = df[df['sentiment'] == 0].sample(n=2500, random_state=42)
df_subset = pd.concat([df_positive, df_negative]).sample(frac=1, random_state=42)  # Shuffle

# Text cleaning function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    words = text.split()
    return ' '.join(words)

# Apply preprocessing
df_subset['cleaned_review'] = df_subset['review'].apply(preprocess_text)

# Split subset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_subset['cleaned_review'], df_subset['sentiment'], test_size=0.2, random_state=42)


########## BoW

# Convert text data into BoW feature vectors
vectorizer_bow = CountVectorizer()
X_train_bow = vectorizer_bow.fit_transform(X_train)
X_test_bow = vectorizer_bow.transform(X_test)

# Train Naïve Bayes on BoW
nb_bow = MultinomialNB()
nb_bow.fit(X_train_bow, y_train)

# Predict on test data
y_pred_bow = nb_bow.predict(X_test_bow)

# Evaluate performance
accuracy_bow = accuracy_score(y_test, y_pred_bow)
print("=== Naïve Bayes with BoW (Subset) ===")
print(f"Accuracy: {accuracy_bow:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_bow))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bow))


print("\n\n\n")

########## TF-IDF

# Convert text data into TF-IDF feature vectors
vectorizer_tfidf = TfidfVectorizer()
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

# Train Naïve Bayes on TF-IDF
nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, y_train)

# Predict on test data
y_pred_tfidf = nb_tfidf.predict(X_test_tfidf)

# Evaluate performance
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
print("=== Naïve Bayes with TF-IDF (Subset) ===")
print(f"Accuracy: {accuracy_tfidf:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_tfidf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tfidf))
