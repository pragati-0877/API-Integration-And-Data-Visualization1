# AI Chatbot with NLP in Python
# -----------------------------
# Install requirements first:
# pip install nltk scikit-learn

import nltk
import random
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# -----------------------------
# Preprocessing Functions
# -----------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Lowercase
    text = text.lower()
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove punctuation & stopwords, then lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens 
              if word not in string.punctuation and word not in stop_words]
    return ' '.join(tokens)

# -----------------------------
# Sample Knowledge Base
# -----------------------------
corpus = [
    "Hello! How can I help you today?",
    "I am an AI chatbot created to assist you with questions.",
    "My creator made me using Python and NLP techniques.",
    "Natural Language Processing (NLP) is the ability of machines to understand human language.",
    "Python is a popular programming language for AI and Data Science.",
    "Goodbye! Have a great day!"
]

# Preprocess the corpus
processed_corpus = [preprocess(sentence) for sentence in corpus]

# -----------------------------
# Chatbot Response Logic
# -----------------------------
def chatbot_response(user_input):
    user_input_processed = preprocess(user_input)
    all_sentences = processed_corpus + [user_input_processed]

    # Vectorize with TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(all_sentences)

    # Compute cosine similarity
    similarity_scores = cosine_similarity(tfidf[-1], tfidf[:-1])
    index = similarity_scores.argsort()[0][-1]
    score = similarity_scores[0][index]

    # If similarity is too low, return fallback
    if score < 0.2:
        return "I'm not sure I understand. Can you rephrase?"
    else:
        return corpus[index]

# -----------------------------
# Main Chat Loop
# -----------------------------
print("AI Chatbot: Hello! Type 'bye' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['bye', 'exit', 'quit']:
        print("AI Chatbot: Goodbye! 👋")
        break
    print("AI Chatbot:", chatbot_response(user_input))
