import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

from preprocess_data import preprocess_data
from svm import train_svm, evaluate_model as evaluate_svm
from logistic import LogisticRegression, evaluate_model as evaluate_logistic

def train_random_forest(X_train, y_train, n_estimators=100, verbose=False):
    """Train a Random Forest model."""
    start_time = time.time()
    
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        verbose=1 if verbose else 0
    )
    
    rf_model.fit(X_train, y_train)
    print(f"\nRandom Forest training completed in {time.time() - start_time:.2f} seconds")
    
    return rf_model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model performance using various metrics."""
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

def run_ensemble():
    """Run the ensemble model combining SVM, Logistic Regression, and Random Forest."""
    file_name = "emails.csv"
    working_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(working_dir, file_name)
    file_path = os.path.join(data_dir, file_name)
    
    print("-" * 50)
    print("ENSEMBLE SPAM CLASSIFICATION USING SVM, LOGISTIC REGRESSION, AND RANDOM FOREST")
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
    
    # Train SVM model
    print("\nTraining the SVM model...")
    svm_model = train_svm(X_train, y_train, C=1.0)
    svm_train_pred = svm_model.decision_function(X_train)  # Get decision function scores instead
    svm_test_pred = svm_model.decision_function(X_test)
    
    # Train Logistic Regression model
    print("\nTraining the Logistic Regression model...")
    logistic_model = LogisticRegression(learning_rate=0.1, max_iterations=1000)
    logistic_model.fit(X_train, y_train)
    logistic_train_probs = logistic_model.predict_prob(X_train)
    logistic_test_probs = logistic_model.predict_prob(X_test)
    
    # Combine predictions for Random Forest
    print("\nPreparing features for Random Forest...")
    X_train_ensemble = np.column_stack([
        svm_train_pred,        # SVM decision function scores
        logistic_train_probs   # Logistic Regression probabilities
    ])
    
    X_test_ensemble = np.column_stack([
        svm_test_pred,         # SVM decision function scores
        logistic_test_probs    # Logistic Regression probabilities
    ])
    
    # Train Random Forest
    print("\nTraining the Random Forest model...")
    rf_model = train_random_forest(X_train_ensemble, y_train)
    
    # Evaluate the ensemble model
    print("\nEvaluating the ensemble model...")
    metrics = evaluate_model(rf_model, X_train_ensemble, y_train, X_test_ensemble, y_test)
    
    print(f"\nEnsemble Model Evaluation Results:")
    print(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Testing Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Confusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Print individual model performances for comparison
    print("\nIndividual Model Performances:")
    print("\nSVM Model:")
    svm_metrics = evaluate_svm(svm_model, X_train, y_train, X_test, y_test)
    print(f"Testing Accuracy: {svm_metrics['test_accuracy']:.4f}")
    
    print("\nLogistic Regression Model:")
    logistic_metrics = evaluate_logistic(y_test, logistic_model.predict(X_test))
    print(f"Testing Accuracy: {logistic_metrics['accuracy']:.4f}")

if __name__ == "__main__":
    run_ensemble() 