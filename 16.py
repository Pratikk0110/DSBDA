import pandas as pd
import string
import nltk
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# Load the dataset (update the path to your file)
df = pd.read_csv('amazon_alexa_reviews.csv')

# Display the first few rows of the dataset
print(df.head())
# Function to remove punctuation from text
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

df['no_punct_text'] = df['verified_reviews'].apply(remove_punctuation)
# Tokenize the text into words
df['tokens'] = df['no_punct_text'].apply(word_tokenize)
# Define the stopwords
stop_words = set(stopwords.words('english'))

# Function to remove stopwords from tokenized text
def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in stop_words]

df['filtered_tokens'] = df['tokens'].apply(remove_stopwords)
# Initialize the stemmer
stemmer = PorterStemmer()

# Function to perform stemming
def stem_words(tokens):
    return [stemmer.stem(word) for word in tokens]

df['stemmed_tokens'] = df['filtered_tokens'].apply(stem_words)
# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to perform lemmatization
def lemmatize_words(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

df['lemmatized_tokens'] = df['filtered_tokens'].apply(lemmatize_words)
# Initialize the CountVectorizer for Bag of Words technique
vectorizer = CountVectorizer()

# Apply the vectorizer to the 'no_punct_text' column to get the Bag of Words
X_bow = vectorizer.fit_transform(df['no_punct_text'])

# Convert the output to a DataFrame for readability
bow_df = pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out())

# Display the first few rows of the Bag of Words representation
print(bow_df.head())
# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Apply the TF-IDF vectorizer to the review text
X_tfidf = tfidf_vectorizer.fit_transform(df['no_punct_text'])

# Convert the output to a DataFrame for readability
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Display the first few rows of the TF-IDF representation
print(tfidf_df.head())
