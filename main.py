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

print("Original label:", y_train[0])
print("One-hot encoded:", y_train_encoded[0])