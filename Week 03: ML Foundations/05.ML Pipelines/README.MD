# Day 15 – ML Pipelines

## What are ML Pipelines and Why are They Important for Deep Learning?

ML Pipelines are sequences of data processing and modeling steps that automate the workflow of machine learning projects. They provide a structured framework for organizing the entire lifecycle of a machine learning model, from data ingestion and preprocessing to model training, evaluation, and deployment.

ML Pipelines are essential because:
- They ensure reproducibility of experiments and results
- They streamline the development and deployment process
- They reduce errors by automating repetitive tasks
- They make it easier to maintain and update models
- They facilitate collaboration among team members
- They enable scalability from prototyping to production

## Essential Subtopics

### 1. Data Preprocessing Pipeline

**What is a data preprocessing pipeline?**
A data preprocessing pipeline is a sequence of transformations applied to raw data to make it suitable for machine learning models. This includes handling missing values, encoding categorical variables, scaling numerical features, and feature engineering.

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Custom PyTorch transformer for data preprocessing
class PyTorchPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numerical_pipeline = None
        self.categorical_pipeline = None
        self.preprocessor = None
    
    def fit(self, X, y=None):
        # Identify numerical and categorical columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        # Create pipelines for numerical and categorical features
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine pipelines
        self.preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ])
        
        self.preprocessor.fit(X)
        return self
    
    def transform(self, X):
        # Transform the data
        X_processed = self.preprocessor.transform(X)
        
        # Convert to PyTorch tensor
        if hasattr(X_processed, "toarray"):  # For sparse matrices
            X_processed = X_processed.toarray()
        
        return torch.FloatTensor(X_processed)

# Example usage
# Load sample data
data = pd.DataFrame({
    'age': [25, 30, 35, np.nan, 40],
    'income': [50000, 60000, 75000, 80000, np.nan],
    'gender': ['M', 'F', 'M', 'F', 'M'],
    'education': ['High School', 'College', 'College', 'Graduate', 'High School'],
    'target': [0, 1, 1, 1, 0]
})

# Split features and target
X = data.drop('target', axis=1)
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit preprocessor
preprocessor = PyTorchPreprocessor()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"Processed training data shape: {X_train_processed.shape}")
print(f"Processed test data shape: {X_test_processed.shape}")
```

**When and where we use data preprocessing pipelines:**
- At the beginning of any machine learning project
- When dealing with heterogeneous data types (numerical, categorical, text)
- When we need to ensure consistent preprocessing between training and inference
- In production systems where data comes in raw format and needs to be transformed

**Real-world examples:**
- In credit scoring, preprocessing pipelines handle missing values and encode categorical variables like employment type
- In recommendation systems, preprocessing pipelines normalize user ratings and encode categorical features like product categories
- In healthcare analytics, preprocessing pipelines handle missing patient data and normalize vital signs

### 2. Feature Engineering Pipeline

**What is a feature engineering pipeline?**
A feature engineering pipeline is a sequence of operations that create new features from existing data to improve model performance. This includes feature extraction, feature selection, and creating interaction features.

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif

# Custom PyTorch feature engineering transformer
class PyTorchFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, k_best=10, degree=2):
        self.k_best = k_best
        self.degree = degree
        self.poly_features = None
        self.feature_selector = None
    
    def fit(self, X, y=None):
        # Create polynomial features
        self.poly_features = PolynomialFeatures(degree=self.degree, include_bias=False)
        X_poly = self.poly_features.fit_transform(X)
        
        # Select best features
        self.feature_selector = SelectKBest(score_func=f_classif, k=self.k_best)
        self.feature_selector.fit(X_poly, y)
        
        return self
    
    def transform(self, X):
        # Create polynomial features
        X_poly = self.poly_features.transform(X)
        
        # Select best features
        X_selected = self.feature_selector.transform(X_poly)
        
        # Convert to PyTorch tensor
        return torch.FloatTensor(X_selected)

# Example usage
# Create sample data
np.random.seed(42)
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = np.random.randint(0, 2, 100)  # Binary target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit feature engineer
feature_engineer = PyTorchFeatureEngineer(k_best=10, degree=2)
X_train_fe = feature_engineer.fit_transform(X_train, y_train)
X_test_fe = feature_engineer.transform(X_test)

print(f"Original training data shape: {X_train.shape}")
print(f"Feature engineered training data shape: {X_train_fe.shape}")
```

**When and where we use feature engineering pipelines:**
- When we need to extract more information from existing features
- When dealing with high-dimensional data and need to select the most relevant features
- When we want to create interaction features to capture non-linear relationships
- In domains where domain knowledge can be incorporated into feature creation

**Real-world examples:**
- In financial modeling, creating features like debt-to-income ratio from individual debt and income features
- In image processing, extracting features like edges, textures, and shapes from raw pixel data
- In time series analysis, creating features like moving averages, lagged values, and seasonality indicators

### 3. Model Training Pipeline

**What is a model training pipeline?**
A model training pipeline automates the process of training a machine learning model, including hyperparameter tuning, cross-validation, and model selection. It ensures that the training process is reproducible and efficient.

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np

# Define a simple neural network
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Model training pipeline
class ModelTrainingPipeline:
    def __init__(self, input_size, hidden_size=64, num_classes=2, learning_rate=0.001, 
                 batch_size=32, num_epochs=10, k_folds=5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.k_folds = k_folds
        self.best_model = None
        self.best_accuracy = 0
    
    def train(self, X, y):
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # K-fold cross-validation
        kfold = KFold(n_splits=self.k_folds, shuffle=True)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_tensor)):
            print(f"Fold {fold+1}/{self.k_folds}")
            
            # Split data
            X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
            y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
            
            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            # Initialize model
            model = SimpleClassifier(self.input_size, self.hidden_size, self.num_classes)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            
            # Training
            for epoch in range(self.num_epochs):
                model.train()
                for batch_X, batch_y in train_loader:
                    # Forward pass
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val)
                    _, val_preds = torch.max(val_outputs, 1)
                    val_accuracy = accuracy_score(y_val, val_preds)
                    
                    if (epoch+1) % 5 == 0:
                        print(f"Epoch {epoch+1}/{self.num_epochs}, Validation Accuracy: {val_accuracy:.4f}")
            
            # Save best model
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.best_model = model.state_dict()
        
        print(f"Best validation accuracy: {self.best_accuracy:.4f}")
        return self.best_model
    
    def predict(self, X):
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X)
        
        # Initialize model with best parameters
        model = SimpleClassifier(self.input_size, self.hidden_size, self.num_classes)
        model.load_state_dict(self.best_model)
        model.eval()
        
        # Make predictions
        with torch.no_grad():
            outputs = model(X_tensor)
            _, preds = torch.max(outputs, 1)
        
        return preds.numpy()

# Example usage
# Create sample data
np.random.seed(42)
X = np.random.rand(200, 10)  # 200 samples, 10 features
y = np.random.randint(0, 2, 200)  # Binary target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model pipeline
pipeline = ModelTrainingPipeline(input_size=10, hidden_size=32, num_epochs=20)
best_model_state = pipeline.train(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Test accuracy: {accuracy:.4f}")
```

**When and where we use model training pipelines:**
- When we need to train multiple models with different hyperparameters
- When we want to ensure consistent training across different experiments
- When we need to automate the model selection process
- In production systems where models need to be retrained periodically

**Real-world examples:**
- In e-commerce recommendation systems, where models need to be retrained with new user interaction data
- In financial fraud detection, where models need to adapt to new fraud patterns
- In healthcare analytics, where models need to be updated with new patient data

### 4. Model Evaluation Pipeline

**What is a model evaluation pipeline?**
A model evaluation pipeline automates the process of assessing model performance using various metrics and techniques. It provides a comprehensive view of how well a model is performing and helps identify areas for improvement.

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Model evaluation pipeline
class ModelEvaluationPipeline:
    def __init__(self, model):
        self.model = model
        self.metrics = {}
    
    def evaluate(self, X_test, y_test):
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_test)
        y_tensor = torch.LongTensor(y_test)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability of positive class
            _, preds = torch.max(outputs, 1)
        
        # Convert to numpy arrays
        y_true = y_tensor.numpy()
        y_pred = preds.numpy()
        y_prob = probabilities.numpy()
        
        # Calculate metrics
        self.metrics['accuracy'] = accuracy_score(y_true, y_pred)
        self.metrics['precision'] = precision_score(y_true, y_pred)
        self.metrics['recall'] = recall_score(y_true, y_pred)
        self.metrics['f1'] = f1_score(y_true, y_pred)
        self.metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.metrics['confusion_matrix'] = cm
        
        return self.metrics
    
    def plot_confusion_matrix(self):
        if 'confusion_matrix' not in self.metrics:
            raise ValueError("Run evaluate() first to generate confusion matrix.")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
    
    def print_metrics(self):
        if not self.metrics:
            raise ValueError("Run evaluate() first to calculate metrics.")
        
        print("Model Evaluation Metrics:")
        print(f"Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall: {self.metrics['recall']:.4f}")
        print(f"F1 Score: {self.metrics['f1']:.4f}")
        print(f"ROC AUC: {self.metrics['roc_auc']:.4f}")

# Example usage
# Create a simple model
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create sample data
np.random.seed(42)
X = np.random.rand(200, 10)  # 200 samples, 10 features
y = np.random.randint(0, 2, 200)  # Binary target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
model = SimpleClassifier(input_size=10, hidden_size=32, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)
    
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
evaluator = ModelEvaluationPipeline(model)
metrics = evaluator.evaluate(X_test, y_test)
evaluator.print_metrics()
evaluator.plot_confusion_matrix()
```

**When and where we use model evaluation pipelines:**
- After training a model to assess its performance
- When comparing multiple models to select the best one
- In production systems to monitor model performance over time
- When debugging models to understand their strengths and weaknesses

**Real-world examples:**
- In medical diagnosis systems, where both precision and recall are critical
- In credit scoring, where the cost of false positives and false negatives needs to be balanced
- In A/B testing frameworks, where different models are compared to select the best one

### 5. Model Deployment Pipeline

**What is a model deployment pipeline?**
A model deployment pipeline automates the process of deploying a trained model to a production environment, including model serialization, API creation, and scaling infrastructure. It ensures that models can be served efficiently and reliably.

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
from datetime import datetime

# Define a simple model
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Model deployment pipeline
class ModelDeploymentPipeline:
    def __init__(self, model, model_name="model"):
        self.model = model
        self.model_name = model_name
        self.model_path = f"{model_name}.pt"
        self.scaler_path = f"{model_name}_scaler.pkl"
        self.app = None
    
    def save_model(self, scaler=None):
        # Save the model
        torch.save(self.model.state_dict(), self.model_path)
        
        # Save the scaler if provided
        if scaler is not None:
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        print(f"Model saved to {self.model_path}")
        if scaler is not None:
            print(f"Scaler saved to {self.scaler_path}")
    
    def load_model(self):
        # Load the model
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        
        # Load the scaler if it exists
        scaler = None
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        
        print(f"Model loaded from {self.model_path}")
        if scaler is not None:
            print(f"Scaler loaded from {self.scaler_path}")
        
        return scaler
    
    def create_api(self, scaler=None):
        # Create Flask app
        self.app = Flask(__name__)
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            # Get data from request
            data = request.json
            if not data or 'features' not in data:
                return jsonify({'error': 'No features provided'}), 400
            
            # Convert to numpy array
            features = np.array(data['features']).reshape(1, -1)
            
            # Apply scaler if provided
            if scaler is not None:
                features = scaler.transform(features)
            
            # Convert to PyTorch tensor
            features_tensor = torch.FloatTensor(features)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(features_tensor)
                _, preds = torch.max(outputs, 1)
                probabilities = torch.softmax(outputs, dim=1)
            
            # Prepare response
            response = {
                'prediction': int(preds[0].item()),
                'probabilities': probabilities[0].tolist()
            }
            
            return jsonify(response)
        
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})
        
        return self.app
    
    def run_api(self, host='0.0.0.0', port=5000, debug=False):
        if self.app is None:
            raise ValueError("API not created. Call create_api() first.")
        
        print(f"Running API on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# Example usage
# Create and train a simple model
model = SimpleClassifier(input_size=10, hidden_size=32, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create sample data
np.random.seed(42)
X = np.random.rand(200, 10)  # 200 samples, 10 features
y = np.random.randint(0, 2, 200)  # Binary target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)
    
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Deploy the model
deployment_pipeline = ModelDeploymentPipeline(model, model_name="heart_disease_classifier")

# Save the model
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
deployment_pipeline.save_model(scaler)

# Load the model (simulating a deployment environment)
loaded_scaler = deployment_pipeline.load_model()

# Create API
app = deployment_pipeline.create_api(loaded_scaler)

# Run the API (commented out to prevent blocking)
# deployment_pipeline.run_api()

# Example of how to use the API with requests
"""
import requests

# Example request
data = {
    'features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}

response = requests.post('http://localhost:5000/predict', json=data)
print(response.json())
"""
```

**When and where we use model deployment pipelines:**
- When moving models from development to production
- When we need to serve predictions through an API
- When we need to update models in production without downtime
- In systems that require real-time predictions

**Real-world examples:**
- In e-commerce platforms, where product recommendation models need to be served to millions of users
- In financial services, where fraud detection models need to make real-time decisions
- In healthcare applications, where diagnostic models need to be integrated into clinical workflows

## Applications in Deep Learning

### 1. End-to-End Image Classification Pipeline

Let's create a complete pipeline for an image classification task:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Define the CNN model
class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# End-to-end pipeline
class ImageClassificationPipeline:
    def __init__(self, batch_size=64, learning_rate=0.001, num_epochs=10):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.transform = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_data(self):
        # Define transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load CIFAR-10 dataset
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=self.transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   download=True, transform=self.transform)
        
        # Split training data into train and validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Class names
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck')
    
    def train_model(self):
        # Initialize model
        self.model = ImageClassifier(num_classes=10).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            running_loss = 0.0
            
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            train_loss = running_loss / len(self.train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss = val_loss / len(self.val_loader)
            val_losses.append(val_loss)
            
            val_accuracy = 100 * correct / total
            val_accuracies.append(val_accuracy)
            
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        
        # Plot training and validation loss
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Validation Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self):
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate accuracy
        accuracy = 100 * accuracy_score(all_labels, all_preds)
        print(f'Test Accuracy: {accuracy:.2f}%')
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=self.classes))
        
        return all_preds, all_labels
    
    def visualize_predictions(self, num_samples=10):
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Get a batch of test images
        dataiter = iter(self.test_loader)
        images, labels = next(dataiter)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            images_gpu = images.to(self.device)
            outputs = self.model(images_gpu)
            _, predicted = torch.max(outputs, 1)
        
        # Display images with predictions
        plt.figure(figsize=(15, 7))
        for i in range(min(num_samples, len(images))):
            plt.subplot(1, num_samples, i+1)
            plt.imshow(np.transpose(images[i].numpy() / 2 + 0.5, (1, 2, 0)))
            plt.title(f"True: {self.classes[labels[i]]}\nPred: {self.classes[predicted[i]]}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# Run the pipeline
pipeline = ImageClassificationPipeline(batch_size=64, learning_rate=0.001, num_epochs=10)
pipeline.load_data()
pipeline.train_model()
all_preds, all_labels = pipeline.evaluate_model()
pipeline.visualize_predictions()
```

### 2. Text Classification Pipeline with BERT

Let's create a pipeline for text classification using a pre-trained BERT model:

```python
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm

# Text classification pipeline with BERT
class TextClassificationPipeline:
    def __init__(self, model_name='bert-base-uncased', num_classes=2, max_length=128, batch_size=32, learning_rate=2e-5, num_epochs=3):
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_data(self, texts, labels):
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        
        # Tokenize and encode sequences
        train_encodings = self.tokenizer(X_train, truncation=True, padding=True, max_length=self.max_length)
        val_encodings = self.tokenizer(X_val, truncation=True, padding=True, max_length=self.max_length)
        test_encodings = self.tokenizer(X_test, truncation=True, padding=True, max_length=self.max_length)
        
        # Convert to PyTorch tensors
        train_dataset = TensorDataset(
            torch.tensor(train_encodings['input_ids']),
            torch.tensor(train_encodings['attention_mask']),
            torch.tensor(y_train.values)
        )
        
        val_dataset = TensorDataset(
            torch.tensor(val_encodings['input_ids']),
            torch.tensor(val_encodings['attention_mask']),
            torch.tensor(y_val.values)
        )
        
        test_dataset = TensorDataset(
            torch.tensor(test_encodings['input_ids']),
            torch.tensor(test_encodings['attention_mask']),
            torch.tensor(y_test.values)
        )
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=self.batch_size)
        self.val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=self.batch_size)
        self.test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=self.batch_size)
        
        return X_test, y_test
    
    def train_model(self):
        # Initialize model
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_classes
        )
        self.model.to(self.device)
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} - Training"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, labels = batch
                
                optimizer.zero_grad()
                
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}/{self.num_epochs} - Average training loss: {avg_train_loss:.4f}")
            
            # Validation
            self.model.eval()
            val_loss = 0
            predictions, true_labels = [], []
            
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} - Validation"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, labels = batch
                
                with torch.no_grad():
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits
                
                val_loss += loss.item()
                
                logits = logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()
                
                predictions.extend(np.argmax(logits, axis=1).flatten())
                true_labels.extend(label_ids.flatten())
            
            avg_val_loss = val_loss / len(self.val_loader)
            val_accuracy = accuracy_score(true_labels, predictions)
            
            print(f"Epoch {epoch+1}/{self.num_epochs} - Validation loss: {avg_val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}")
    
    def evaluate_model(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        self.model.eval()
        predictions, true_labels = [], []
        
        for batch in tqdm(self.test_loader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, attention_mask, labels = batch
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            
            predictions.extend(np.argmax(logits, axis=1).flatten())
            true_labels.extend(label_ids.flatten())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions))
        
        return predictions
    
    def predict(self, texts):
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        self.model.eval()
        
        # Tokenize texts
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
        
        # Move to device
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Get predictions
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        
        return predictions

# Example usage
# Create sample data
texts = [
    "This movie was great! I really enjoyed it.",
    "The acting was terrible and the plot was boring.",
    "I loved the special effects in this film.",
    "The worst movie I have ever seen.",
    "An amazing cinematic experience.",
    "I would not recommend this movie to anyone."
]
labels = pd.Series([1, 0, 1, 0, 1, 0])  # 1: positive, 0: negative

# Create and run pipeline
pipeline = TextClassificationPipeline(
    model_name='bert-base-uncased',
    num_classes=2,
    max_length=128,
    batch_size=16,
    learning_rate=2e-5,
    num_epochs=3
)

X_test, y_test = pipeline.load_data(texts, labels)
pipeline.train_model()
predictions = pipeline.evaluate_model(X_test, y_test)

# Test with new texts
new_texts = [
    "This film was absolutely fantastic!",
    "I was really disappointed with this movie."
]
new_predictions = pipeline.predict(new_texts)
for text, pred in zip(new_texts, new_predictions):
    sentiment = "positive" if pred == 1 else "negative"
    print(f"Text: '{text}'\nPredicted sentiment: {sentiment}\n")
```

## Common Problems and Advancements

### 1. Problem: Data Leakage

**Issue:**
Data leakage occurs when information from outside the training dataset is used to create the model, leading to overly optimistic performance estimates. This can happen when preprocessing steps like imputation or scaling are applied before splitting the data into training and test sets.

**How advancements solved this:**
- **Pipeline-based preprocessing**: Ensuring that preprocessing steps are fitted only on the training data and then applied to the test data
- **Cross-validation within pipelines**: Performing preprocessing within each fold of cross-validation
- **Feature selection within pipelines**: Selecting features based only on the training data

```python
# Problem code: Data leakage due to preprocessing before train-test split
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Create sample data with missing values
np.random.seed(42)
X = np.random.rand(200, 5)
X[np.random.rand(200, 5) < 0.1] = np.nan  # Add missing values
y = np.random.randint(0, 2, 200)

# Preprocessing before train-test split (PROBLEM: Data leakage)
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
X_imputed = imputer.fit_transform(X)  # Fitting on entire dataset
X_scaled = scaler.fit_transform(X_imputed)  # Fitting on entire dataset

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a simple model
class SimpleClassifier(nn.Module):
    def __init__(self, input_size):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

# Train model
model = SimpleClassifier(input_size=5)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_preds = (test_outputs > 0.5).float()
    accuracy = accuracy_score(y_test_tensor, test_preds)
    print(f'Test Accuracy (with data leakage): {accuracy:.4f}')

# Enhancement code: Proper pipeline to prevent data leakage
# Create sample data with missing values
np.random.seed(42)
X = np.random.rand(200, 5)
X[np.random.rand(200, 5) < 0.1] = np.nan  # Add missing values
y = np.random.randint(0, 2, 200)

# Split data first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Fit on training data only
X_train_processed = preprocessing_pipeline.fit_transform(X_train)

# Transform test data using the fitted pipeline
X_test_processed = preprocessing_pipeline.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_processed)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_processed)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

# Train model
model = SimpleClassifier(input_size=5)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_preds = (test_outputs > 0.5).float()
    accuracy = accuracy_score(y_test_tensor, test_preds)
    print(f'Test Accuracy (without data leakage): {accuracy:.4f}')
```

### 2. Problem: Concept Drift

**Issue:**
Concept drift occurs when the statistical properties of the target variable or the relationships between variables change over time. This can lead to degraded model performance in production, as the model was trained on data that no longer represents the current situation.

**How advancements solved this:**
- **Monitoring model performance**: Tracking model performance metrics over time to detect degradation
- **Automated retraining pipelines**: Setting up systems to automatically retrain models when performance drops below a threshold
- **Online learning**: Using algorithms that can adapt to new data without full retraining
- **Ensemble methods**: Combining models trained on different time periods to handle changing patterns

```python
# Problem code: Static model without handling concept drift
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Generate data with concept drift
def generate_data(n_samples, concept_version=1):
    np.random.seed(42)
    X = np.random.randn(n_samples, 5)
    
    if concept_version == 1:
        # Initial concept: positive correlation with first feature
        y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + np.random.randn(n_samples) * 0.5) > 0
    else:
        # Changed concept: negative correlation with first feature
        y = (-X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2] + np.random.randn(n_samples) * 0.5) > 0
    
    return X, y.astype(int)

# Generate initial training data
X_train, y_train = generate_data(1000, concept_version=1)

# Define a simple classifier
class SimpleClassifier(nn.Module):
    def __init__(self, input_size):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)

# Train model
model = SimpleClassifier(input_size=5)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Simulate data streams over time with concept drift
time_periods = 10
samples_per_period = 100
accuracies = []

for period in range(time_periods):
    # Gradually introduce concept drift
    concept_version = 1 if period < 5 else 2
    drift_factor = min(1.0, max(0.0, (period - 4) / 2.0))  # Gradual drift
    
    # Generate test data with increasing concept drift
    X_test1, y_test1 = generate_data(samples_per_period, concept_version=1)
    X_test2, y_test2 = generate_data(samples_per_period, concept_version=2)
    
    # Mix data based on drift factor
    X_test = (1 - drift_factor) * X_test1 + drift_factor * X_test2
    y_test = np.where(np.random.rand(samples_per_period) < (1 - drift_factor), y_test1, y_test2)
    
    # Evaluate model
    X_test_tensor = torch.FloatTensor(X_test)
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_preds = (test_outputs > 0.5).float().numpy().flatten()
        accuracy = accuracy_score(y_test, test_preds)
        accuracies.append(accuracy)
        print(f"Period {period+1}, Drift Factor: {drift_factor:.2f}, Accuracy: {accuracy:.4f}")

# Plot accuracy over time
plt.figure(figsize=(10, 6))
plt.plot(range(1, time_periods+1), accuracies, marker='o')
plt.axvline(x=5, color='r', linestyle='--', label='Concept Drift Starts')
plt.xlabel('Time Period')
plt.ylabel('Accuracy')
plt.title('Model Performance Over Time with Concept Drift')
plt.legend()
plt.grid(True)
plt.show()

# Enhancement code: Adaptive pipeline to handle concept drift
class AdaptivePipeline:
    def __init__(self, input_size, threshold=0.7, window_size=3):
        self.input_size = input_size
        self.threshold = threshold  # Performance threshold for retraining
        self.window_size = window_size  # Window size for performance monitoring
        self.model = SimpleClassifier(input_size)
        self.performance_history = []
        self.retrain_count = 0
    
    def train(self, X, y):
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).view(-1, 1)
        
        # Train model
        self.model = SimpleClassifier(self.input_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        num_epochs = 50
        for epoch in range(num_epochs):
            # Forward pass
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        self.retrain_count += 1
        print(f"Model retrained (Retrain count: {self.retrain_count})")
    
    def evaluate(self, X, y):
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        
        # Evaluate model
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_tensor)
            test_preds = (test_outputs > 0.5).float().numpy().flatten()
            accuracy = accuracy_score(y, test_preds)
        
        # Update performance history
        self.performance_history.append(accuracy)
        
        # Keep only the most recent window_size performances
        if len(self.performance_history) > self.window_size:
            self.performance_history = self.performance_history[-self.window_size:]
        
        return accuracy
    
    def check_retrain(self):
        # Check if average performance over the window is below threshold
        if len(self.performance_history) >= self.window_size:
            avg_performance = sum(self.performance_history) / self.window_size
            return avg_performance < self.threshold
        return False
    
    def predict(self, X):
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            preds = (outputs > 0.5).float().numpy().flatten()
        
        return preds

# Generate initial training data
X_train, y_train = generate_data(1000, concept_version=1)

# Initialize adaptive pipeline
pipeline = AdaptivePipeline(input_size=5, threshold=0.7, window_size=3)

# Train initial model
pipeline.train(X_train, y_train)

# Simulate data streams over time with concept drift
time_periods = 10
samples_per_period = 100
adaptive_accuracies = []

for period in range(time_periods):
    # Gradually introduce concept drift
    concept_version = 1 if period < 5 else 2
    drift_factor = min(1.0, max(0.0, (period - 4) / 2.0))  # Gradual drift
    
    # Generate test data with increasing concept drift
    X_test1, y_test1 = generate_data(samples_per_period, concept_version=1)
    X_test2, y_test2 = generate_data(samples_per_period, concept_version=2)
    
    # Mix data based on drift factor
    X_test = (1 - drift_factor) * X_test1 + drift_factor * X_test2
    y_test = np.where(np.random.rand(samples_per_period) < (1 - drift_factor), y_test1, y_test2)
    
    # Evaluate model
    accuracy = pipeline.evaluate(X_test, y_test)
    adaptive_accuracies.append(accuracy)
    print(f"Period {period+1}, Drift Factor: {drift_factor:.2f}, Accuracy: {accuracy:.4f}")
    
    # Check if model needs retraining
    if pipeline.check_retrain():
        # Generate new training data with current concept
        X_new, y_new = generate_data(500, concept_version=concept_version)
        pipeline.train(X_new, y_new)

# Plot accuracy over time
plt.figure(figsize=(10, 6))
plt.plot(range(1, time_periods+1), accuracies, marker='o', label='Static Model')
plt.plot(range(1, time_periods+1), adaptive_accuracies, marker='s', label='Adaptive Model')
plt.axvline(x=5, color='r', linestyle='--', label='Concept Drift Starts')
plt.xlabel('Time Period')
plt.ylabel('Accuracy')
plt.title('Model Performance Over Time with Concept Drift')
plt.legend()
plt.grid(True)
plt.show()
```

### 3. Problem: Scalability and Performance

**Issue:**
As datasets grow larger and models become more complex, ML pipelines can face scalability and performance issues. Training time can become prohibitively long, and inference latency may not meet the requirements of real-time applications.

**How advancements solved this:**
- **Distributed training**: Using multiple GPUs or machines to train models in parallel
- **Model optimization techniques**: Quantization, pruning, and distillation to reduce model size and improve inference speed
- **Pipeline parallelism**: Overlapping different stages of the pipeline to improve throughput
- **Hardware acceleration**: Leveraging specialized hardware like TPUs and GPUs for faster computation

```python
# Problem code: Non-optimized pipeline with performance issues
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Create a large dataset
np.random.seed(42)
X = np.random.rand(10000, 100)  # 10,000 samples, 100 features
y = np.random.randint(0, 2, 10000)  # Binary target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a large model
class LargeModel(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(LargeModel, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

# Train model without optimization
model = LargeModel(input_size=100, hidden_sizes=[512, 256, 128, 64])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop without optimization
start_time = time.time()
num_epochs = 10
batch_size = 32  # Small batch size

for epoch in range(num_epochs):
    model.train()
    epoch_start_time = time.time()
    
    # Manual batching without DataLoader
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size]
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    epoch_time = time.time() - epoch_start_time
    print(f'Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s')

training_time = time.time() - start_time
print(f'Total training time: {training_time:.2f}s')

# Evaluate
model.eval()
with torch.no_grad():
    start_time = time.time()
    test_outputs = model(X_test_tensor)
    test_preds = (test_outputs > 0.5).float()
    inference_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test_tensor, test_preds)
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Inference time for {len(X_test_tensor)} samples: {inference_time:.4f}s')

# Enhancement code: Optimized pipeline with performance improvements
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Create a large dataset
np.random.seed(42)
X = np.random.rand(10000, 100)  # 10,000 samples, 100 features
y = np.random.randint(0, 2, 10000)  # Binary target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a more efficient model
class EfficientModel(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(EfficientModel, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # Add batch normalization
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # Add dropout for regularization
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Convert to PyTorch tensors and create datasets
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create data loaders with larger batch size and multiple workers
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

# Initialize model with mixed precision if available
model = EfficientModel(input_size=100, hidden_sizes=[256, 128, 64])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Use mixed precision training if available
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

# Training loop with optimizations
start_time = time.time()
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    epoch_start_time = time.time()
    
    for batch_X, batch_y in train_loader:
        # Move data to GPU if available
        if torch.cuda.is_available():
            batch_X = batch_X.cuda()
            batch_y = batch_y.cuda()
        
        optimizer.zero_grad()
        
        # Use mixed precision if available
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
    
    epoch_time = time.time() - epoch_start_time
    print(f'Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s')

training_time = time.time() - start_time
print(f'Total training time: {training_time:.2f}s')

# Evaluate with optimizations
model.eval()
all_preds = []
all_labels = []

start_time = time.time()
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        # Move data to GPU if available
        if torch.cuda.is_available():
            batch_X = batch_X.cuda()
        
        # Use mixed precision if available
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(batch_X)
        else:
            outputs = model(batch_X)
        
        preds = (outputs > 0.5).float()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch_y.numpy())

inference_time = time.time() - start_time

# Concatenate all batches
all_preds = np.concatenate(all_preds).flatten()
all_labels = np.concatenate(all_labels).flatten()

accuracy = accuracy_score(all_labels, all_preds)
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Inference time for {len(X_test_tensor)} samples: {inference_time:.4f}s')

# Model quantization for further optimization
print("\nApplying quantization for further optimization...")
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Evaluate quantized model
quantized_model.eval()
all_preds = []

start_time = time.time()
with torch.no_grad():
    for batch_X, _ in test_loader:
        # Move data to GPU if available
        if torch.cuda.is_available():
            batch_X = batch_X.cuda()
        
        outputs = quantized_model(batch_X)
        preds = (outputs > 0.5).float()
        all_preds.append(preds.cpu().numpy())

quantized_inference_time = time.time() - start_time

# Concatenate all batches
all_preds = np.concatenate(all_preds).flatten()

accuracy = accuracy_score(all_labels, all_preds)
print(f'Quantized Model Test Accuracy: {accuracy:.4f}')
print(f'Quantized Model Inference time for {len(X_test_tensor)} samples: {quantized_inference_time:.4f}s')
print(f'Speedup from quantization: {inference_time / quantized_inference_time:.2f}x')
```

## Project: Full ML Pipeline from Raw Data to Deployment

Now, let's implement a complete ML pipeline that takes raw data, processes it, trains a model, evaluates it, and prepares it for deployment:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify
import pickle
import os
from datetime import datetime

# Full ML Pipeline
class FullMLPipeline:
    def __init__(self, model_name="heart_disease_model"):
        self.model_name = model_name
        self.model_path = f"{model_name}.pt"
        self.preprocessor_path = f"{model_name}_preprocessor.pkl"
        self.model = None
        self.preprocessor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_data(self, url):
        """Load data from URL"""
        # Column names for the Heart Disease UCI dataset
        column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 
                        'oldpeak', 'slope', 'ca', 'thal', 'target']
        
        # Load data
        data = pd.read_csv(url, names=column_names, na_values='?')
        
        # Handle missing values
        data = data.dropna()
        
        # Convert target to binary (0 = no disease, 1 = has disease)
        data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)
        
        return data
    
    def preprocess_data(self, data):
        """Preprocess the data"""
        # Split features and target
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Identify numerical and categorical columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        # Create preprocessing pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine pipelines
        self.preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ])
        
        # Fit and transform the data
        X_processed = self.preprocessor.fit_transform(X)
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_processed)
        y_tensor = torch.FloatTensor(y.values)
        
        return X_tensor, y_tensor
    
    def define_model(self, input_size):
        """Define the neural network model"""
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
        
        self.model = HeartDiseaseNet(input_size).to(self.device)
        return self.model
    
    def train_model(self, X_train, y_train, X_val, y_val, num_epochs=100, learning_rate=0.001):
        """Train the model"""
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Move data to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 10
        counter = 0
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            outputs = self.model(X_train).squeeze()
            loss = criterion(outputs, y_train)
            train_losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val).squeeze()
                val_loss = criterion(val_outputs, y_val)
                val_losses.append(val_loss.item())
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                # Save the best model
                best_model_state = self.model.state_dict()
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
            
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
        
        # Load the best model
        self.model.load_state_dict(best_model_state)
        
        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model"""
        # Move data to device
        X_test = X_test.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            test_outputs = self.model(X_test).squeeze()
            test_preds = (test_outputs > 0.5).float()
        
        # Move predictions back to CPU for sklearn metrics
        test_preds = test_preds.cpu().numpy()
        y_test = y_test.cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, test_preds)
        print(f'Test Accuracy: {accuracy:.4f}')
        
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
        
        return accuracy
    
    def save_model(self):
        """Save the model and preprocessor"""
        # Save the model
        torch.save(self.model.state_dict(), self.model_path)
        
        # Save the preprocessor
        with open(self.preprocessor_path, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        
        print(f"Model saved to {self.model_path}")
        print(f"Preprocessor saved to {self.preprocessor_path}")
    
    def load_model(self, input_size):
        """Load the model and preprocessor"""
        # Define the model
        self.define_model(input_size)
        
        # Load the model
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        
        # Load the preprocessor
        with open(self.preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)
        
        print(f"Model loaded from {self.model_path}")
        print(f"Preprocessor loaded from {self.preprocessor_path}")
        
        return self.model, self.preprocessor
    
    def create_api(self):
        """Create a Flask API for the model"""
        app = Flask(__name__)
        
        @app.route('/predict', methods=['POST'])
        def predict():
            # Get data from request
            data = request.json
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Convert to DataFrame
            df = pd.DataFrame(data, index=[0])
            
            # Preprocess the data
            X_processed = self.preprocessor.transform(df)
            X_tensor = torch.FloatTensor(X_processed).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(X_tensor).squeeze()
                prediction = (output > 0.5).float().item()
                probability = output.item()
            
            # Prepare response
            response = {
                'prediction': int(prediction),
                'probability': probability,
                'interpretation': 'Heart disease detected' if prediction == 1 else 'No heart disease detected'
            }
            
            return jsonify(response)
        
        @app.route('/health', methods=['GET'])
        def health():
            return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})
        
        return app
    
    def run_full_pipeline(self, url):
        """Run the full pipeline from data loading to API creation"""
        # Load data
        print("Loading data...")
        data = self.load_data(url)
        print(f"Data loaded with shape: {data.shape}")
        
        # Preprocess data
        print("\nPreprocessing data...")
        X, y = self.preprocess_data(data)
        print(f"Data preprocessed with shape: {X.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
        
        print(f"\nData split into:")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Define model
        print("\nDefining model...")
        model = self.define_model(input_size=X_train.shape[1])
        print(f"Model defined with input size: {X_train.shape[1]}")
        
        # Train model
        print("\nTraining model...")
        self.train_model(X_train, y_train, X_val, y_val, num_epochs=100, learning_rate=0.001)
        
        # Evaluate model
        print("\nEvaluating model...")
        accuracy = self.evaluate_model(X_test, y_test)
        
        # Save model
        print("\nSaving model...")
        self.save_model()
        
        # Create API
        print("\nCreating API...")
        app = self.create_api()
        
        print("\nFull pipeline completed successfully!")
        print(f"Model accuracy: {accuracy:.4f}")
        print("API is ready to run.")
        
        return app

# Run the full pipeline
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
pipeline = FullMLPipeline(model_name="heart_disease_classifier")
app = pipeline.run_full_pipeline(url)

# Example of how to use the API (commented out to prevent blocking)
"""
# To run the API, uncomment the following line:
app.run(host='0.0.0.0', port=5000, debug=False)

# Example of how to use the API with requests:
import requests

# Example request
data = {
    'age': [52],
    'sex': [1],
    'cp': [0],
    'trestbps': [125],
    'chol': [212],
    'fbs': [0],
    'restecg': [1],
    'thalach': [168],
    'exang': [0],
    'oldpeak': [1.0],
    'slope': [2],
    'ca': [2],
    'thal': [3]
}

response = requests.post('http://localhost:5000/predict', json=data)
print(response.json())
"""
```

This full ML pipeline demonstrates how to create an end-to-end system that takes raw data, processes it, trains a model, evaluates it, and prepares it for deployment through an API. It includes best practices like data preprocessing, model evaluation, early stopping, and model serialization.

## Summary

ML Pipelines are essential for streamlining the machine learning workflow, ensuring reproducibility, and facilitating deployment. In this lesson, we've covered:

1. **Data Preprocessing Pipeline**: Automating the transformation of raw data into a format suitable for machine learning models
2. **Feature Engineering Pipeline**: Creating new features and selecting the most relevant ones to improve model performance
3. **Model Training Pipeline**: Automating the training process, including hyperparameter tuning and cross-validation
4. **Model Evaluation Pipeline**: Assessing model performance using various metrics and visualization techniques
5. **Model Deployment Pipeline**: Preparing models for production, including serialization and API creation

We've also explored how these concepts are applied in real-world scenarios, common problems encountered, and how advancements in the field have addressed these challenges.

The full ML pipeline project demonstrates how to create an end-to-end system that takes raw data, processes it, trains a model, evaluates it, and prepares it for deployment. This comprehensive approach ensures that machine learning models are not only accurate but also robust, maintainable, and ready for production use.

Understanding ML pipelines is crucial for anyone looking to master machine learning and deep learning, as they provide the framework for developing, deploying, and maintaining models in real-world applications.