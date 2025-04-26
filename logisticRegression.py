
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class RegularizedLogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000, lambda_=0.1):
        self.lr = lr
        self.n_iters = n_iters
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None
        
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear_pred)
            
            # Regularized gradient
            dw = (1/n_samples) * (np.dot(X.T, (predictions - y)) + (self.lambda_ * self.weights)
            db = (1/n_samples) * np.sum(predictions - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X, threshold=0.5):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_pred)
        return (y_pred >= threshold).astype(int)

# Load your data
data = np.array([
        [0,3,0,1,1,1,0,1,0,1,0], [0,3,0,1,1,2,1,1,0,1,0],
        [0,3,0,0,1,2,2,1,0,2,0], [0,3,2,2,1,1,2,1,1,1,0],
        [0,1,2,3,1,1,2,1,5,1,0], [0,3,0,2,0,1,2,1,5,2,0],
        [0,1,0,3,1,1,1,1,1,1,1], [1,3,0,1,1,1,0,1,1,1,0],
        [0,2,0,2,1,1,1,1,3,1,1], [0,3,0,3,0,1,1,1,1,0,1],
        [1,1,0,3,1,2,1,1,1,1,1], [1,2,0,1,1,1,2,1,3,1,1],
        [0,3,0,2,1,2,2,1,1,1,0], [0,3,0,3,1,1,2,1,3,1,1],
        [0,3,0,2,1,1,2,1,2,1,0], [1,3,0,1,1,1,1,1,3,2,1],
        [0,2,0,2,0,2,2,1,4,2,1], [0,3,0,3,1,1,1,1,5,1,1],
        [0,3,0,3,1,1,2,1,1,2,1], [0,0,0,3,0,1,2,1,4,1,1],
        [1,3,0,3,1,1,2,1,3,1,1], [0,3,0,2,1,1,1,1,4,1,0],
        [1,3,0,3,1,2,2,1,4,1,0], [0,3,0,2,1,1,2,1,1,1,1],
        [0,2,0,3,1,2,2,1,1,1,0], [0,3,0,3,1,1,2,1,0,1,0],
        [1,2,0,2,1,1,1,0,1,1,0], [1,2,0,2,1,1,1,1,1,1,0],
        [0,3,0,2,1,1,2,1,5,1,0], [0,3,0,3,1,1,1,1,1,1,0],
        [0,3,0,3,1,1,1,1,0,1,1], [0,3,0,1,1,2,1,1,0,1,1],
        [0,3,0,0,1,2,2,1,0,2,0], [0,3,1,2,1,1,2,1,1,1,0],
        [0,1,2,3,1,1,2,1,5,1,0], [0,3,0,2,0,1,2,1,5,2,0],
        [0,1,0,3,1,1,1,1,1,1,1], [1,3,0,1,1,1,0,1,1,1,0],
        [0,2,0,2,1,1,1,1,4,1,1], [0,3,0,3,0,1,1,1,1,0,1],
        [1,1,0,3,1,2,1,1,1,1,1], [1,2,0,1,1,1,2,1,3,1,1],
        [0,3,0,2,1,2,2,1,1,1,0], [0,3,0,3,1,1,2,1,1,1,0],
        [0,3,0,2,1,1,2,1,2,1,0], [1,3,0,1,1,1,1,1,3,2,1],
        [0,2,0,2,0,2,2,1,4,2,1], [0,3,0,3,1,1,1,1,5,1,1],
        [0,3,0,3,1,1,2,1,1,2,1],  [0,3,0,1,1,2,1,1,0,1,0]
    ])

X = data[:, :-1]
y = data[:, -1]

# 1. Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Feature selection
selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X_scaled, y)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train model
model = RegularizedLogisticRegression(lr=0.1, lambda_=0.01)
model.fit(X_train, y_train)

# 5. Evaluate
train_acc = np.mean(model.predict(X_train) == y_train)
test_acc = np.mean(model.predict(X_test) == y_test)
print(f"Train Accuracy: {train_acc:.2%}")
print(f"Test Accuracy: {test_acc:.2%}")

# 6. Feature importance visualization
plt.bar(range(len(model.weights)), np.abs(model.weights))
plt.title("Feature Importance (Absolute Weight Values)")
plt.xlabel("Feature Index")
plt.ylabel("Weight Magnitude")
plt.show()