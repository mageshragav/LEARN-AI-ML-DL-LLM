# Day 12 – Classification Models

## What are Classification Models and Why are They Important for Deep Learning?

Classification models are a type of supervised learning algorithm that categorizes input data into predefined classes or labels. Unlike regression models that predict continuous values, classification models predict discrete outcomes. In deep learning, classification is one of the most fundamental tasks, serving as the backbone for many real-world applications.

Classification models are essential because:
- They enable decision-making in automated systems
- They form the basis for many complex AI applications
- They help in understanding patterns and relationships in data
- They provide probabilistic predictions that can be used for risk assessment
- They're applicable across numerous domains from healthcare to finance

## Essential Subtopics

### 1. Logistic Regression

**What is logistic regression?**
Logistic regression is a statistical model that uses a logistic function to model a binary dependent variable. Despite its name, it's a classification algorithm, not a regression one. It estimates the probability that an instance belongs to a particular class.

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a synthetic binary classification dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).view(-1, 1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).view(-1, 1)

# Define logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))

# Initialize model, loss function, and optimizer
model = LogisticRegression(input_dim=10)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
with torch.no_grad():
    model.eval()
    test_outputs = model(X_test)
    test_preds = (test_outputs > 0.5).float()
    accuracy = accuracy_score(y_test, test_preds)
    print(f'Accuracy: {accuracy:.4f}')
```

**When and where we use logistic regression:**
- Binary classification problems (spam detection, disease diagnosis)
- As a baseline model for more complex algorithms
- When we need probabilistic interpretations of predictions
- In scenarios where model interpretability is important

**Real-world examples:**
- Email spam detection (spam vs. not spam)
- Medical diagnosis (disease present vs. absent)
- Credit scoring (loan default risk)
- Customer churn prediction (will a customer leave or stay)

### 2. K-Nearest Neighbors (KNN)

**What is K-Nearest Neighbors?**
K-Nearest Neighbors is a non-parametric classification algorithm that classifies a data point based on how its neighbors are classified. It stores all available cases and classifies new cases based on a similarity measure (e.g., distance functions).

**PyTorch Implementation:**
```python
import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# KNN implementation in PyTorch
class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        # Calculate distances between test and training instances
        distances = torch.cdist(X, self.X_train)
        
        # Get indices of k nearest samples
        _, indices = torch.topk(distances, self.k, dim=1, largest=False)
        
        # Get labels of k nearest samples
        knn_labels = self.y_train[indices]
        
        # Predict the most common label
        predictions = torch.mode(knn_labels, dim=1).values
        return predictions

# Initialize and train KNN model
knn = KNN(k=5)
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)
accuracy = accuracy_score(y_test.numpy(), predictions.numpy())
print(f'Accuracy: {accuracy:.4f}')
```

**When and where we use KNN:**
- When the decision boundary is irregular
- For multi-class classification problems
- When training data is large and representative
- In recommendation systems (finding similar items)
- In pattern recognition applications

**Real-world examples:**
- Image recognition (classifying images based on similar features)
- Recommendation systems (finding similar products or users)
- Handwritten digit recognition
- Customer segmentation based on behavior

### 3. Support Vector Machines (SVM)

**What is a Support Vector Machine?**
A Support Vector Machine is a supervised learning model that analyzes data for classification and regression analysis. SVM constructs a hyperplane or set of hyperplanes in a high-dimensional space, which can be used for classification. The best hyperplane is the one that represents the largest separation or margin between the two classes.

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# SVM implementation in PyTorch
class SVM(nn.Module):
    def __init__(self, input_dim):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

# Hinge loss function for SVM
class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
    
    def forward(self, outputs, labels):
        # Convert labels to -1 and 1
        labels = 2 * labels - 1
        # Calculate hinge loss
        loss = torch.mean(torch.clamp(1 - outputs * labels, min=0))
        return loss

# Initialize model, loss function, and optimizer
model = SVM(input_dim=10)
criterion = HingeLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)  # L2 regularization

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train.view(-1, 1))
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
with torch.no_grad():
    model.eval()
    test_outputs = model(X_test)
    test_preds = (test_outputs > 0).float().view(-1)
    accuracy = accuracy_score(y_test, test_preds)
    print(f'Accuracy: {accuracy:.4f}')
```

**When and where we use SVM:**
- High-dimensional classification problems
- When the number of dimensions is greater than the number of samples
- For non-linear classification using kernel tricks
- In text classification and image recognition
- When clear margin of separation is needed

**Real-world examples:**
- Text categorization (spam detection, sentiment analysis)
- Image classification (handwritten digit recognition)
- Bioinformatics (protein classification, cancer classification)
- Handwriting recognition

### 4. Decision Trees for Classification

**What are Decision Trees?**
Decision Trees are a non-parametric supervised learning method used for classification and regression. The model predicts the value of a target variable by learning simple decision rules inferred from the data features. Trees can be visualized, making them easy to interpret.

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# Decision Tree implementation in PyTorch
class DecisionNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # Index of feature to split on
        self.threshold = threshold      # Threshold value for the split
        self.left = left                # Left subtree
        self.right = right              # Right subtree
        self.value = value              # Class value if leaf node

class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None
    
    def fit(self, X, y):
        self.n_classes = len(torch.unique(y))
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        
        # Stopping criteria
        if depth >= self.max_depth or len(torch.unique(y)) == 1:
            leaf_value = self._most_common_label(y)
            return DecisionNode(value=leaf_value)
        
        # Find best split
        best_feature_idx, best_threshold = self._best_split(X, y)
        
        # Split data
        left_idx = X[:, best_feature_idx] <= best_threshold
        right_idx = ~left_idx
        
        # Grow left and right subtrees
        left_subtree = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        
        return DecisionNode(
            feature_idx=best_feature_idx,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )
    
    def _best_split(self, X, y):
        m = y.size(0)
        if m <= 1:
            return None, None
        
        # Count of each class in the current node
        num_parent = [torch.sum(y == c).item() for c in range(self.n_classes)]
        
        # Gini of parent node
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_feature_idx, best_threshold = None, None
        
        # Loop through all features
        for feature_idx in range(self.n_features):
            thresholds, classes = zip(*sorted(zip(X[:, feature_idx], y)))
            num_left = [0] * self.n_classes
            num_right = num_parent.copy()
            
            # Loop through all possible thresholds
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                
                # Skip if thresholds are equal
                if thresholds[i] == thresholds[i - 1]:
                    continue
                
                # Calculate Gini index
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes))
                
                # Weighted average of Gini indices
                gini = (i * gini_left + (m - i) * gini_right) / m
                
                # Update best split if this one is better
                if gini < best_gini:
                    best_gini = gini
                    best_feature_idx = feature_idx
                    best_threshold = (thresholds[i] + thresholds[i - 1]) / 2
        
        return best_feature_idx, best_threshold
    
    def _most_common_label(self, y):
        return torch.mode(y).values.item()
    
    def predict(self, X):
        return torch.tensor([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

# Initialize and train Decision Tree model
tree = DecisionTree(max_depth=5)
tree.fit(X_train, y_train)

# Make predictions
predictions = tree.predict(X_test)
accuracy = accuracy_score(y_test.numpy(), predictions.numpy())
print(f'Accuracy: {accuracy:.4f}')
```

**When and where we use Decision Trees:**
- When model interpretability is important
- For both numerical and categorical data
- When dealing with non-linear relationships
- In ensemble methods like Random Forests and Gradient Boosting
- When we want to capture feature interactions

**Real-world examples:**
- Credit risk assessment (loan approval)
- Medical diagnosis (disease prediction)
- Customer segmentation
- Fault detection in industrial systems

## Applications in Deep Learning

### 1. Multi-class Classification with Neural Networks

While traditional classification models are useful, deep learning extends these concepts to more complex problems:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a simple neural network for multi-class classification
class NeuralNetClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Initialize model, loss function, and optimizer
model = NeuralNetClassifier(input_size=784, hidden_size=128, num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Reshape data from (batch_size, 1, 28, 28) to (batch_size, 784)
        data = data.view(-1, 784)
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

# Evaluation
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data = data.view(-1, 784)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
accuracy = 100. * correct / len(test_loader.dataset)
print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
```

### 2. Binary Classification with Imbalanced Data

Real-world classification problems often have imbalanced classes, which requires special handling:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Create an imbalanced dataset
np.random.seed(42)
X = np.random.randn(1000, 10)
y = np.array([0] * 900 + [1] * 100)  # 90% class 0, 10% class 1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# Define a simple neural network
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Method 1: Use weighted loss function
model = BinaryClassifier(input_size=10)

# Calculate weights for the loss function
pos_weight = torch.tensor([len(y_train) / (2 * sum(y_train))])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test).squeeze()
    test_preds = (test_outputs > 0.5).float()
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_preds))
    
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds))
```

### 3. Multi-label Classification

Some classification problems require assigning multiple labels to each instance:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import hamming_loss, accuracy_score
from sklearn.datasets import make_multilabel_classification

# Create a multi-label dataset
X, y = make_multilabel_classification(n_samples=1000, n_features=20, n_classes=5, n_labels=2, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# Define a neural network for multi-label classification
class MultiLabelClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultiLabelClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Initialize model, loss function, and optimizer
model = MultiLabelClassifier(input_size=20, hidden_size=64, num_classes=5)
criterion = nn.BCELoss()  # Binary Cross Entropy for each label
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_preds = (test_outputs > 0.5).float()
    
    # Calculate metrics
    hamming = hamming_loss(y_test, test_preds)
    subset_accuracy = accuracy_score(y_test, test_preds)
    
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Subset Accuracy: {subset_accuracy:.4f}")
```

## Common Problems and Advancements

### 1. Problem: Class Imbalance

**Issue:**
In many real-world classification problems, the classes are imbalanced, with one class having significantly more samples than the other. This can lead to models that are biased toward the majority class and perform poorly on the minority class.

**How advancements solved this:**
- **Resampling techniques**: Oversampling the minority class or undersampling the majority class
- **Synthetic data generation**: Using techniques like SMOTE to create synthetic samples of the minority class
- **Cost-sensitive learning**: Assigning higher misclassification costs to the minority class
- **Ensemble methods**: Combining multiple models trained on balanced subsets of data

```python
# Problem code: Training on imbalanced data without special handling
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
import numpy as np

# Create imbalanced dataset
np.random.seed(42)
X = np.random.randn(1000, 10)
y = np.array([0] * 900 + [1] * 100)  # 90% class 0, 10% class 1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# Define a simple neural network
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Initialize model, loss function, and optimizer
model = BinaryClassifier(input_size=10)
criterion = nn.BCELoss()  # Standard loss without handling imbalance
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test).squeeze()
    test_preds = (test_outputs > 0.5).float()
    
    print("Without handling imbalance:")
    print(classification_report(y_test, test_preds))

# Enhancement code: Using weighted loss function and oversampling
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

# Create a dataset
train_dataset = TensorDataset(X_train, y_train)

# Calculate weights for sampling
class_counts = [sum(y_train == 0), sum(y_train == 1)]
class_weights = [1.0 / class_counts[0], 1.0 / class_counts[1]]
sample_weights = [class_weights[int(y)] for y in y_train]

# Create a weighted sampler
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# Create a dataloader with the sampler
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)

# Initialize a new model
model = BinaryClassifier(input_size=10)

# Use weighted loss
pos_weight = torch.tensor([class_counts[0] / class_counts[1]])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with the balanced dataloader
num_epochs = 50
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        # Forward pass
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test).squeeze()
    test_preds = (test_outputs > 0.5).float()
    
    print("\nWith handling imbalance:")
    print(classification_report(y_test, test_preds))
```

### 2. Problem: Overfitting

**Issue:**
Classification models, especially complex ones like deep neural networks, can overfit to the training data. This means they learn the training examples too well, including noise and outliers, and fail to generalize to new, unseen data.

**How advancements solved this:**
- **Regularization techniques**: L1 and L2 regularization to penalize complex models
- **Dropout**: Randomly dropping units during training to prevent co-adaptation
- **Early stopping**: Monitoring validation loss and stopping training when it starts to increase
- **Data augmentation**: Artificially increasing the size and diversity of the training data
- **Batch normalization**: Normalizing the inputs of each layer to reduce internal covariate shift

```python
# Problem code: Model without regularization techniques
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Create a synthetic dataset
np.random.seed(42)
X = np.random.randn(1000, 20)
y = np.random.randint(0, 2, 1000)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# Define a neural network without regularization
class OverfittingNet(nn.Module):
    def __init__(self, input_size):
        super(OverfittingNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

# Initialize model, loss function, and optimizer
model = OverfittingNet(input_size=20)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  # High learning rate

# Training loop
num_epochs = 100
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)
    train_losses.append(loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val).squeeze()
        val_loss = criterion(val_outputs, y_val)
        val_losses.append(val_loss.item())
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss (Without Regularization)')
plt.legend()
plt.show()

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test).squeeze()
    test_preds = (test_outputs > 0.5).float()
    test_accuracy = accuracy_score(y_test, test_preds)
    print(f'Test Accuracy: {test_accuracy:.4f}')

# Enhancement code: Model with regularization techniques
# Define a neural network with dropout and batch normalization
class RegularizedNet(nn.Module):
    def __init__(self, input_size):
        super(RegularizedNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc4 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

# Initialize model, loss function, and optimizer
model = RegularizedNet(input_size=20)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Lower learning rate with L2 regularization

# Training loop with early stopping
num_epochs = 100
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 10
counter = 0

for epoch in range(num_epochs):
    # Training
    model.train()
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)
    train_losses.append(loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val).squeeze()
        val_loss = criterion(val_outputs, y_val)
        val_losses.append(val_loss.item())
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        # Save the best model
        best_model_state = model.state_dict()
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Load the best model
model.load_state_dict(best_model_state)

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss (With Regularization)')
plt.legend()
plt.show()

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test).squeeze()
    test_preds = (test_outputs > 0.5).float()
    test_accuracy = accuracy_score(y_test, test_preds)
    print(f'Test Accuracy: {test_accuracy:.4f}')
```

### 3. Problem: High Dimensionality

**Issue:**
In many classification problems, especially in text analysis and image recognition, the input data has a very high number of features (high dimensionality). This can lead to the "curse of dimensionality," where the performance of classification algorithms degrades as the dimensionality increases.

**How advancements solved this:**
- **Dimensionality reduction**: Techniques like PCA, t-SNE, and autoencoders to reduce the number of features
- **Feature selection**: Selecting the most relevant features based on statistical tests or model-based importance
- **Regularization**: L1 regularization (Lasso) that encourages sparsity in the model weights
- **Deep learning approaches**: Using convolutional layers for images or embedding layers for categorical data

```python
# Problem code: Classification with high-dimensional data without dimensionality reduction
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import time

# Create a high-dimensional dataset
X, y = make_classification(n_samples=1000, n_features=1000, n_informative=50, n_redundant=50, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# Define a neural network for high-dimensional data
class HighDimNet(nn.Module):
    def __init__(self, input_size):
        super(HighDimNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Initialize model, loss function, and optimizer
model = HighDimNet(input_size=1000)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
start_time = time.time()
num_epochs = 50
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

training_time = time.time() - start_time

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test).squeeze()
    test_preds = (test_outputs > 0.5).float()
    test_accuracy = accuracy_score(y_test, test_preds)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Training Time: {training_time:.2f} seconds')

# Enhancement code: Using PCA for dimensionality reduction before classification
from sklearn.decomposition import PCA

# Apply PCA to reduce dimensionality
pca = PCA(n_components=100)  # Reduce from 1000 to 100 features
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Convert to PyTorch tensors
X_train_pca = torch.FloatTensor(X_train_pca)
X_test_pca = torch.FloatTensor(X_test_pca)

# Define a neural network for reduced-dimensional data
class ReducedDimNet(nn.Module):
    def __init__(self, input_size):
        super(ReducedDimNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Initialize model, loss function, and optimizer
model = ReducedDimNet(input_size=100)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
start_time = time.time()
num_epochs = 50
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_pca).squeeze()
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

training_time = time.time() - start_time

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_pca).squeeze()
    test_preds = (test_outputs > 0.5).float()
    test_accuracy = accuracy_score(y_test, test_preds)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Training Time: {training_time:.2f} seconds')
    print(f'Explained Variance Ratio: {sum(pca.explained_variance_ratio_):.4f}')
```

## Project: Heart Disease Detection

Now, let's implement a heart disease detection system using classification models:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Heart Disease UCI dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 
                'oldpeak', 'slope', 'ca', 'thal', 'target']
data = pd.read_csv(url, names=column_names, na_values='?')

# Data preprocessing
# Handle missing values
data = data.dropna()

# Convert target to binary (0 = no disease, 1 = has disease)
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

# Split features and target
X = data.drop('target', axis=1).values
y = data['target'].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# Define the neural network model
class HeartDiseaseNet(nn.Module):
    def __init__(self, input_size):
        super(HeartDiseaseNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Initialize model, loss function, and optimizer
model = HeartDiseaseNet(input_size=13)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)
    train_losses.append(loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test).squeeze()
        val_loss = criterion(val_outputs, y_test)
        val_losses.append(val_loss.item())
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test).squeeze()
    test_preds = (test_outputs > 0.5).float()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, test_preds)
    print(f'Accuracy: {accuracy:.4f}')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds))

# Feature importance analysis using permutation importance
def permutation_importance(model, X, y, n_repeats=10):
    model.eval()
    baseline_accuracy = accuracy_score(y, (model(X).squeeze() > 0.5).float())
    importances = []
    
    for i in range(X.shape[1]):
        X_permuted = X.clone()
        accuracies = []
        
        for _ in range(n_repeats):
            # Permute feature i
            X_permuted[:, i] = X_permuted[torch.randperm(X_permuted.size(0)), i]
            
            # Calculate accuracy with permuted feature
            with torch.no_grad():
                preds = (model(X_permuted).squeeze() > 0.5).float()
                acc = accuracy_score(y, preds)
                accuracies.append(acc)
        
        # Importance is decrease in accuracy
        importance = baseline_accuracy - np.mean(accuracies)
        importances.append(importance)
    
    return importances

# Calculate feature importance
importances = permutation_importance(model, X_test, y_test)

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.bar(column_names[:-1], importances)
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
```

This heart disease detection project demonstrates how classification models can be applied to real-world medical data. The model uses various techniques to handle potential issues like overfitting (dropout, batch normalization) and provides insights into which features are most important for making predictions.

## Summary

Classification models are fundamental to machine learning and deep learning, enabling us to categorize data into predefined classes. In this lesson, we've covered:

1. **Logistic Regression**: A simple yet powerful binary classification algorithm that estimates probabilities
2. **K-Nearest Neighbors (KNN)**: A non-parametric algorithm that classifies based on similarity to neighboring points
3. **Support Vector Machines (SVM)**: A method that finds the optimal hyperplane to separate classes
4. **Decision Trees**: A model that makes decisions based on a series of questions about the features

We've also explored how these concepts are applied in deep learning, common problems encountered, and how advancements in the field have addressed these challenges.

The heart disease detection project provides a practical application of classification models, demonstrating how to preprocess data, train a model, evaluate its performance, and analyze feature importance.

Understanding classification models is crucial for anyone looking to master deep learning, as they form the foundation for many more complex algorithms and applications in AI.