#This file should use preprocess_data.py to preprocess the data and then train a logistic regression model on the preprocessed data

import preprocess_data
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tol=1e-4):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tol = tol
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        """Apply sigmoid function."""
        # Clip z to avoid overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """Train the logistic regression model."""
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check for convergence
            if i > 0 and np.all(np.abs(self.learning_rate * dw) < self.tol):
                print(f"Converged after {i} iterations")
                break
                
        return self
    
    def predict_prob(self, X):
        """Predict probability of class 1."""
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        """Predict class labels."""
        return (self.predict_prob(X) >= threshold).astype(int)

def evaluate_model(y_true, y_pred):
    """Calculate various metrics for model evaluation."""
    accuracy = np.mean(y_true == y_pred)
    precision = np.sum((y_true == 1) & (y_pred == 1)) / (np.sum(y_pred == 1) + 1e-10)
    recall = np.sum((y_true == 1) & (y_pred == 1)) / (np.sum(y_true == 1) + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
if __name__ == "__main__":
    # Load and preprocess the data
    file_path = "emails.csv/emails.csv"
    preprocessed_data = preprocess_data.preprocess_data(file_path)

    # Extract training and testing data
    X_train = preprocessed_data['X_train']
    X_test = preprocessed_data['X_test']
    y_train = preprocessed_data['y_train']
    y_test = preprocessed_data['y_test']

    # Initialize and train the model
    print("\nTraining logistic regression model...")
    model = LogisticRegression(learning_rate=0.1, max_iterations=1000)
    model.fit(X_train, y_train)

    # Make predictions
    print("\nMaking predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_prob = model.predict_prob(X_train)

    # Evaluate the model
    print("\nEvaluating model performance...")
    train_metrics = evaluate_model(y_train, y_train_pred)
    test_metrics = evaluate_model(y_test, y_test_pred)

    print("\nTraining Set Metrics:")
    for metric, value in train_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    print("\nTest Set Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")