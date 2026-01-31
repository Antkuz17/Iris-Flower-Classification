import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Returns a tuple (X, y) where X is the flower data and y is the target labels  
iris = load_iris()

# Separating features and target labels
X = iris.data
y = iris.target

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)


# One-hot encode the labels
def one_hot_encode(y, num_classes=3):
    n_samples = len(y)
    one_hot = np.zeros((n_samples, num_classes))
    one_hot[np.arange(n_samples), y] = 1
    return one_hot

y_train_encoded = one_hot_encode(y_train)
y_test_encoded = one_hot_encode(y_test)

# 4 features in the Iris dataset
input_size = 4      

# 5 neurons in hidden layer
hidden_size = 5

# 3 output classes
output_size = 3

# Initialize weights and biases with predictable random values
np.random.seed(42)  # For reproducibility

# Weights between input and hidden layer 
W1 = np.random.randn(input_size, hidden_size) * 0.01

# Biases for hidden layer 
b1 = np.zeros((1, hidden_size))

# Weights between hidden and output layer
W2 = np.random.randn(hidden_size, output_size) * 0.01

# Biases for output layer
b2 = np.zeros((1, output_size))

# Normalize the features to range 
X_train_normalized = (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0))
X_test_normalized = (X_test - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0))

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    # Input to hidden layer
    Z1 = X @ W1 + b1        
    A1 = sigmoid(Z1)     
    
    # Hidden to output layer
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)     
    
    # Return all values
    return Z1, A1, Z2, A2

# Test it with your training data
Z1, A1, Z2, A2 = forward_propagation(X_train_normalized, W1, b1, W2, b2)

# Cost function 
def compute_cost(A2, Y):
    m = Y.shape[0]  # Number of samples
    
    cost = np.sum((A2 - Y) ** 2) / (2 * m)
    return cost

# Backpropagation
def backpropagation(X, Y, Z1, A1, Z2, A2, W1, W2):
    m = X.shape[0]  # Number of samples
    
    # Output layer error
    dZ2 = A2 - Y
    
    dW2 = (1/m) * (A1.T @ dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
    
    # Hidden layer error (propagate error backwards)
    dZ1 = (dZ2 @ W2.T) * A1 * (1 - A1)
    
    # Gradients for W1 and b1
    dW1 = (1/m) * (X.T @ dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
    
    return dW1, db1, dW2, db2

# Test backpropagation
dW1, db1, dW2, db2 = backpropagation(X_train_normalized, y_train_encoded, Z1, A1, Z2, A2, W1, W2)

print("dW1 shape:", dW1.shape)  # Should be (4, 5)
print("db1 shape:", db1.shape)  # Should be (1, 5)
print("dW2 shape:", dW2.shape)  # Should be (5, 3)
print("db2 shape:", db2.shape)  # Should be (1, 3)


# Training function
def train(X, Y, W1, b1, W2, b2, learning_rate=0.5, epochs=1000):
    costs = []  # To track cost over time
    
    for epoch in range(epochs):
        # Forward propagation
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        
        # Compute cost
        cost = compute_cost(A2, Y)
        costs.append(cost)
        
        # Backpropagation
        dW1, db1, dW2, db2 = backpropagation(X, Y, Z1, A1, Z2, A2, W1, W2)
        
        # Update weights and biases
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        
        # Print progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Cost: {cost:.4f}")
    
    return W1, b1, W2, b2, costs

# Calling the train function
W1_trained, b1_trained, W2_trained, b2_trained, costs = train(
    X_train_normalized, 
    y_train_encoded, 
    W1, b1, W2, b2, 
    learning_rate=0.5, 
    epochs=1000
)

# Make predictions on test set
def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    # Convert probabilities to class predictions 
    predictions = np.argmax(A2, axis=1)
    return predictions

# Calculate accuracy
def calculate_accuracy(predictions, true_labels):
    accuracy = np.mean(predictions == true_labels) * 100
    return accuracy


# Test on test data
test_predictions = predict(X_test_normalized, W1_trained, b1_trained, W2_trained, b2_trained)
test_accuracy = calculate_accuracy(test_predictions, y_test)

