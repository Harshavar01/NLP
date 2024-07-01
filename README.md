# NLP
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources if not already downloaded
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load the dataset
df = pd.read_csv(r"C:\Users\Tq28\Downloads\blogs.csv")

# Display basic information
print(df.head())
print(df.info())
print(df['Labels'].value_counts())

# Text preprocessing
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

df['Cleaned_Text'] = df['Data'].apply(clean_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)  # Limiting to top 1000 features
X = vectorizer.fit_transform(df['Cleaned_Text']).toarray()
y = df['Labels']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Naive Bayes classifier
nb_classifier = MultinomialNB()

# Train the classifier
nb_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = nb_classifier.predict(X_test)

# Evaluate the classifier
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Initialize VADER SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Perform sentiment analysis on each blog post
df['Sentiment'] = df['Data'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Categorize sentiment
df['Sentiment_Label'] = df['Sentiment'].apply(lambda x: 'positive' if x >= 0.05 else 'negative' if x <= -0.05 else 'neutral')

# Analyze sentiments across categories
sentiment_summary = df.groupby('Labels')['Sentiment_Label'].value_counts(normalize=True).unstack()
print("Sentiment Distribution across Categories:\n", sentiment_summary)

