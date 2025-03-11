import os
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

from preprocess_data import preprocess_data

def train_svm(X_train, y_train, max_iter=100000, C=1.0, verbose=False):

    # Create and train SVM model
    start_time = time.time()
    
    # Linear SVM chosen for spam classification due to efficiency with high-dimensional, sparse text data
    # Provides good generalization while allowing interpretability of word importance weights
    model = svm.LinearSVC(C=C, dual=False, max_iter=max_iter, tol=1e-4, verbose=1 if verbose else 0)
    
    model.fit(X_train, y_train)
    
    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
    
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):

    # Make predictions on training and testing data
    y_train_prediction = model.predict(X_train)
    y_test_prediction = model.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_prediction)
    test_accuracy = accuracy_score(y_test, y_test_prediction)
    
    precision = precision_score(y_test, y_test_prediction)
    recall = recall_score(y_test, y_test_prediction)
    f1 = f1_score(y_test, y_test_prediction)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_test_prediction)
    
    # Return metrics
    metrics = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }
    
    return metrics

def run_svm():
    
    file_name = "emails.csv"
    working_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(working_dir, file_name)
    file_path = os.path.join(data_dir, file_name)
    
    
    print("-" * 50)
    print("SPAM CLASSIFICATION USING SCIKIT-LEARN SVM")
    print("-" * 50)
    
    # Preprocess the data
    print("\nPreprocessing the data...")
    start_time = time.time()
    preprocessed_data = preprocess_data(file_path)
    preprocess_time = time.time() - start_time
    print(f"Preprocessing completed in {preprocess_time:.2f} seconds")
    
    X_train = preprocessed_data['X_train']
    X_test = preprocessed_data['X_test']
    y_train = preprocessed_data['y_train']
    y_test = preprocessed_data['y_test']
    vocabulary = preprocessed_data['vocabulary']
    
    print(f"\nDataset Information:")
    print(f"Number of features (vocabulary size): {len(vocabulary)}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    print(f"Spam ratio in training set: {np.mean(y_train):.2f}")
    
    # Train the SVM model
    print("\nTraining the SVM model...")

    svm_model = train_svm(X_train, y_train, C=1.0)
    
    # Evaluate the model
    print("\nEvaluating the model...")

    metrics = evaluate_model(svm_model, X_train, y_train, X_test, y_test  )

    print(f"\nEvaluation Results:")
    print(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Testing Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Confusion Matrix:")
    print(metrics['confusion_matrix'])
    
if __name__ == "__main__":
    run_svm()