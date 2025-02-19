import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# # Download stopwords if not already available
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))

# Load IMDb dataset (Ensure you have a CSV file with 'review' and 'sentiment' columns)
df = pd.read_csv("C:/Users/Mario Rus/Desktop/Proiect Cercetare/IMDB-Dataset.csv")  

# Convert sentiment labels to numeric (positive = 1, negative = 0)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Text cleaning function
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    words = text.split()
    # words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

# Apply preprocessing
df['cleaned_review'] = df['review'].apply(preprocess_text)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42)

