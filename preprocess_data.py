import numpy as np
import pandas as pd
import re
import string
import math
import os
from collections import Counter

def load_data(file_path):
    """Load the dataset from CSV file."""
    df = pd.read_csv(file_path)
    df.columns = ['text', 'spam']
    return df.dropna()

def clean_text(text):
    """Clean and preprocess the text data."""
    # Convert to lowercase and string type
    text = str(text).lower()
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def compute_tf(text_tokens):
    """Compute Term Frequency for a document."""
    word_counts = Counter(text_tokens)
    total_words = len(text_tokens)
    return {word: count / total_words for word, count in word_counts.items()}

def compute_idf(texts):
    """Compute Inverse Document Frequency across all documents."""
    doc_count = len(texts)
    word_doc_count = Counter()
    
    for text in texts:
        words = set(text.split())  # Use set to count unique words per document
        word_doc_count.update(words)
    
    return {word: math.log(doc_count / (1 + count)) for word, count in word_doc_count.items()}

def compute_tf_idf(text, idf_values):
    """Compute TF-IDF scores for a document."""
    tokens = text.split()
    tf = compute_tf(tokens)
    return {word: tf[word] * idf_values.get(word, 0) for word in tf}

def create_vocabulary(texts, min_freq=2, max_freq_ratio=0.95):
    """Create vocabulary from texts with frequency filtering."""
    # Count all words
    word_counts = Counter()
    total_docs = len(texts)
    
    for text in texts:
        words = set(text.split())  # Use set to count document frequency
        word_counts.update(words)
    
    # Filter words based on frequency
    vocabulary = {}
    idx = 0
    for word, count in word_counts.items():
        if (count >= min_freq and  # Remove rare words
            count/total_docs <= max_freq_ratio):  # Remove too frequent words
            vocabulary[word] = idx
            idx += 1
    
    return vocabulary

def text_to_tfidf_vector(text, vocabulary, idf_values):
    """Convert text to TF-IDF vector representation."""
    vector = np.zeros(len(vocabulary))
    tf_idf_scores = compute_tf_idf(text, idf_values)
    
    for word, score in tf_idf_scores.items():
        if word in vocabulary:
            vector[vocabulary[word]] = score
    return vector

def normalize_features(X):
    """Normalize features to have zero mean and unit variance."""
    # Calculate mean and std for each feature
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    # Handle zero standard deviation
    std[std == 0] = 1
    
    # Normalize
    X_normalized = (X - mean) / std
    return X_normalized, mean, std

def train_test_split(X, y, test_size=0.2, random_state=None):
    """Split the data into training and testing sets."""
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate random indices
    indices = np.random.permutation(len(X))
    test_size = int(test_size * len(X))
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    return (X[train_indices], X[test_indices], 
            y[train_indices], y[test_indices])

def preprocess_data(file_path, test_size=0.2, random_state=42):
    # Load data
    print("Loading data...")
    df = load_data(file_path)
    
    # Clean texts
    print("Cleaning texts...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Create vocabulary
    print("Creating vocabulary...")
    vocabulary = create_vocabulary(df['cleaned_text'])
    
    # Compute IDF values
    print("Computing IDF values...")
    idf_values = compute_idf(df['cleaned_text'])
    
    # Convert texts to TF-IDF
    print("Converting texts to TF-IDF features...")
    X = np.array([text_to_tfidf_vector(text, vocabulary, idf_values) 
                for text in df['cleaned_text']])
    y = df['spam'].values
    
    # Split data
    print("Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Normalize features
    print("Normalizing features...")
    X_train_norm, mean, std = normalize_features(X_train)
    X_test_norm = (X_test - mean) / std
    
    # Create a dictionary with preprocessed data
    preprocessed_data = {
        'X_train': X_train_norm,
        'X_test': X_test_norm,
        'y_train': y_train,
        'y_test': y_test,
        'vocabulary': vocabulary,
        'idf_values': idf_values,
        'feature_mean': mean,
        'feature_std': std
    }
    
    # Print some information about the preprocessed data
    print("\nPreprocessing complete!")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Training set shape: {X_train_norm.shape}")
    print(f"Testing set shape: {X_test_norm.shape}")
    
    return preprocessed_data

if __name__ == "__main__":
    file_name = "emails.csv"
    working_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(working_dir, file_name)
    file_path = os.path.join(data_dir, file_name)
    preprocessed_data = preprocess_data(file_path)
    