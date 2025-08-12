# Day 13 – Tree-Based Models

## What are Tree-Based Models and Why are They Important for Machine Learning?

Tree-based models are a class of supervised machine learning algorithms that construct a model in the form of a decision tree. These models use a tree-like structure of decisions and their possible consequences to make predictions. The tree consists of nodes that represent features, branches that represent decision rules, and leaves that represent outcomes.

Tree-based models are important in machine learning because they:
- Provide high interpretability compared to many other ML algorithms
- Can handle both numerical and categorical data
- Require minimal data preprocessing (e.g., no need for feature scaling)
- Can capture non-linear relationships between features and targets
- Are robust to outliers and missing values
- Form the building blocks of more powerful ensemble methods like Random Forest and Gradient Boosting

## Essential Subtopics

### 1. Decision Trees

**What are Decision Trees?**
Decision Trees are non-parametric supervised learning methods used for classification and regression. The model predicts the value of a target variable by learning simple decision rules inferred from the data features.

**PyTorch Implementation:**
```python
import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, n_redundant=0, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors (though scikit-learn works with numpy arrays)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create and train the decision tree
tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(X_train, y_train)

# Evaluate the model
accuracy = tree_clf.score(X_test, y_test)
print(f"Decision Tree Accuracy: {accuracy:.4f}")

# Visualize the decision tree
plt.figure(figsize=(15, 10))
plot_tree(tree_clf, filled=True, feature_names=[f'Feature {i}' for i in range(X.shape[1])], class_names=['Class 0', 'Class 1'])
plt.title("Decision Tree Visualization")
plt.show()
```

**When and where we use Decision Trees:**
- When model interpretability is important (e.g., in healthcare or finance)
- For exploratory data analysis to understand feature importance
- As base learners in ensemble methods like Random Forest and Gradient Boosting
- When dealing with both numerical and categorical features
- When the relationship between features and target is non-linear

**Real-world examples:**
- In healthcare, decision trees can help diagnose diseases based on symptoms
- In finance, they can be used to assess credit risk
- In marketing, they can help identify customer segments
- In manufacturing, they can predict equipment failure based on sensor data

### 2. Random Forest

**What is Random Forest?**
Random Forest is an ensemble learning method that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

**PyTorch Implementation:**
```python
import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create and train the Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_clf.fit(X_train, y_train)

# Evaluate the model
accuracy = rf_clf.score(X_test, y_test)
print(f"Random Forest Accuracy: {accuracy:.4f}")

# Feature importance
feature_importance = rf_clf.feature_importances_
features = [f'Feature {i}' for i in range(X.shape[1])]

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(features, feature_importance)
plt.title("Feature Importance in Random Forest")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Make predictions on new data
new_data = torch.tensor([[0.5, 0.2, 0.8, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6, 0.2]], dtype=torch.float32)
prediction = rf_clf.predict(new_data.numpy())
print(f"Prediction for new data: {prediction[0]}")
```

**When and where we use Random Forest:**
- When we need higher accuracy than a single decision tree
- When dealing with high-dimensional data
- When we want to reduce overfitting compared to a single decision tree
- When we need to estimate feature importance
- When we want to handle missing values and maintain accuracy

**Real-world examples:**
- In banking, Random Forest is used for credit scoring and fraud detection
- In healthcare, it's used for disease prediction and medical image analysis
- In e-commerce, it's used for recommendation systems and customer churn prediction
- In remote sensing, it's used for land cover classification

### 3. Gradient Boosting Machines (GBM)

**What are Gradient Boosting Machines?**
Gradient Boosting is a machine learning technique for regression and classification problems that produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion like other boosting methods, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.

**PyTorch Implementation:**
```python
import torch
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate synthetic regression data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create and train the Gradient Boosting Regressor
gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbm.fit(X_train, y_train)

# Evaluate the model
y_pred = gbm.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Gradient Boosting MSE: {mse:.4f}")

# Plot feature importance
feature_importance = gbm.feature_importances_
features = [f'Feature {i}' for i in range(X.shape[1])]

plt.figure(figsize=(10, 6))
plt.bar(features, feature_importance)
plt.title("Feature Importance in Gradient Boosting")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Make predictions on new data
new_data = torch.tensor([[0.5, 0.2, 0.8, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6, 0.2]], dtype=torch.float32)
prediction = gbm.predict(new_data.numpy())
print(f"Prediction for new data: {prediction[0]}")
```

**When and where we use Gradient Boosting Machines:**
- When we need high predictive accuracy
- When dealing with structured/tabular data
- In competitions like Kaggle where predictive performance is key
- When we have a mix of categorical and numerical features
- When we want to handle different types of loss functions

**Real-world examples:**
- In finance, GBMs are used for credit risk modeling and fraud detection
- In online advertising, they're used for click-through rate prediction
- In healthcare, they're used for predicting patient outcomes and drug discovery
- In meteorology, they're used for weather forecasting

### 4. XGBoost (Extreme Gradient Boosting)

**What is XGBoost?**
XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solves many data science problems in a fast and accurate way.

**PyTorch Implementation:**
```python
import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Convert to DMatrix for XGBoost (optimized data structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'seed': 42
}

# Train the model
num_rounds = 100
bst = xgb.train(params, dtrain, num_rounds)

# Make predictions
y_pred = bst.predict(dtest)
y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"XGBoost Accuracy: {accuracy:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(bst)
plt.title("Feature Importance in XGBoost")
plt.show()

# Make predictions on new data
new_data = torch.tensor([[0.5, 0.2, 0.8, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6, 0.2]], dtype=torch.float32)
new_data_dmatrix = xgb.DMatrix(new_data)
prediction = bst.predict(new_data_dmatrix)
prediction_binary = 1 if prediction[0] > 0.5 else 0
print(f"Prediction for new data: {prediction_binary} (probability: {prediction[0]:.4f})")
```

**When and where we use XGBoost:**
- When we need state-of-the-art performance on structured/tabular data
- In machine learning competitions where performance is critical
- When dealing with imbalanced datasets
- When we need built-in regularization to prevent overfitting
- When we want to handle missing values naturally

**Real-world examples:**
- In credit scoring, XGBoost is used to predict the likelihood of default
- In healthcare, it's used for disease prediction and patient risk stratification
- In e-commerce, it's used for recommendation systems and demand forecasting
- In the insurance industry, it's used for claim prediction and risk assessment

### 5. LightGBM

**What is LightGBM?**
LightGBM is a gradient boosting framework that uses tree-based learning algorithms. It is designed to be distributed and efficient with the following advantages: faster training speed and higher efficiency, lower memory usage, better accuracy, support for parallel and GPU learning, and capability to handle large-scale data.

**PyTorch Implementation:**
```python
import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Set parameters
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'seed': 42
}

# Train the model
num_rounds = 100
bst = lgb.train(params, train_data, num_rounds, valid_sets=[test_data])

# Make predictions
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"LightGBM Accuracy: {accuracy:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 6))
lgb.plot_importance(bst)
plt.title("Feature Importance in LightGBM")
plt.show()

# Make predictions on new data
new_data = torch.tensor([[0.5, 0.2, 0.8, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6, 0.2]], dtype=torch.float32)
prediction = bst.predict(new_data.numpy(), num_iteration=bst.best_iteration)
prediction_binary = 1 if prediction[0] > 0.5 else 0
print(f"Prediction for new data: {prediction_binary} (probability: {prediction[0]:.4f})")
```

**When and where we use LightGBM:**
- When dealing with large datasets that need faster training time
- When we have limited computational resources
- When we need high accuracy with lower memory usage
- When we want to leverage GPU acceleration for training
- When we need to handle categorical features efficiently

**Real-world examples:**
- In recommendation systems, LightGBM is used for click-through rate prediction
- In financial fraud detection, it's used to identify suspicious transactions
- In online advertising, it's used for bid optimization and ad selection
- In predictive maintenance, it's used to forecast equipment failures

## Applications in Machine Learning

### 1. Feature Importance Analysis

Tree-based models provide a natural way to assess feature importance:

```python
import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Generate synthetic data with 10 features but only 5 informative
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Train a Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importances
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), [f"Feature {i}" for i in indices])
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()

# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. Feature {indices[f]} ({importances[indices[f]]:.4f})")
```

### 2. Hyperparameter Tuning with Grid Search

Finding optimal hyperparameters is crucial for tree-based models:

```python
import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Instantiate the grid search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters
print(f"Best parameters: {grid_search.best_params_}")

# Evaluate the best model
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with best parameters: {accuracy:.4f}")

# Compare with default parameters
default_rf = RandomForestClassifier(random_state=42)
default_rf.fit(X_train, y_train)
default_y_pred = default_rf.predict(X_test)
default_accuracy = accuracy_score(y_test, default_y_pred)
print(f"Accuracy with default parameters: {default_accuracy:.4f}")
```

### 3. Handling Imbalanced Data

Tree-based models can be adapted to handle imbalanced datasets:

```python
import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Generate imbalanced synthetic data
X, y = make_classification(n_samples=1000, n_features=10, weights=[0.9, 0.1], random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Train a Random Forest without handling imbalance
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluate the model
print("Classification Report (without handling imbalance):")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (without handling imbalance)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Train a Random Forest with class_weight='balanced'
rf_balanced = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_balanced.fit(X_train, y_train)
y_pred_balanced = rf_balanced.predict(X_test)

# Evaluate the balanced model
print("\nClassification Report (with class_weight='balanced'):")
print(classification_report(y_test, y_pred_balanced))

# Plot confusion matrix for balanced model
plt.figure(figsize=(8, 6))
cm_balanced = confusion_matrix(y_test, y_pred_balanced)
sns.heatmap(cm_balanced, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (with class_weight='balanced')")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

## Common Problems and Advancements

### 1. Problem: Overfitting

**Issue:**
Decision trees are prone to overfitting, especially when they are deep and complex. This happens because the tree can keep splitting until it perfectly memorizes the training data, capturing noise and outliers that don't generalize to new data.

**How advancements solved this:**
- **Pruning**: Removing branches that have little predictive power
- **Setting constraints**: Limiting tree depth, minimum samples per leaf, etc.
- **Ensemble methods**: Combining multiple trees to reduce variance
- **Regularization**: Adding penalties for complex models in gradient boosting

```python
# Problem: Overfitting Decision Tree
import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, noise=0.3, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create an overfitting decision tree (no constraints)
overfit_tree = DecisionTreeClassifier(random_state=42)
overfit_tree.fit(X_train, y_train)

# Evaluate on training and test sets
train_accuracy = overfit_tree.score(X_train, y_train)
test_accuracy = overfit_tree.score(X_test, y_test)

print(f"Overfitting Tree - Training Accuracy: {train_accuracy:.4f}")
print(f"Overfitting Tree - Test Accuracy: {test_accuracy:.4f}")
print(f"Difference (Overfitting Gap): {train_accuracy - test_accuracy:.4f}")

# Enhancement: Regularized Decision Tree
regularized_tree = DecisionTreeClassifier(
    max_depth=5,           # Limit tree depth
    min_samples_split=10,  # Minimum samples to split
    min_samples_leaf=5,    # Minimum samples at leaf node
    random_state=42
)
regularized_tree.fit(X_train, y_train)

# Evaluate on training and test sets
reg_train_accuracy = regularized_tree.score(X_train, y_train)
reg_test_accuracy = regularized_tree.score(X_test, y_test)

print(f"\nRegularized Tree - Training Accuracy: {reg_train_accuracy:.4f}")
print(f"Regularized Tree - Test Accuracy: {reg_test_accuracy:.4f}")
print(f"Difference (Overfitting Gap): {reg_train_accuracy - reg_test_accuracy:.4f}")

# Enhancement: Random Forest (Ensemble method)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=5,           # Limit tree depth
    min_samples_split=10,  # Minimum samples to split
    random_state=42
)
rf.fit(X_train, y_train)

# Evaluate on training and test sets
rf_train_accuracy = rf.score(X_train, y_train)
rf_test_accuracy = rf.score(X_test, y_test)

print(f"\nRandom Forest - Training Accuracy: {rf_train_accuracy:.4f}")
print(f"Random Forest - Test Accuracy: {rf_test_accuracy:.4f}")
print(f"Difference (Overfitting Gap): {rf_train_accuracy - rf_test_accuracy:.4f}")
```

### 2. Problem: Bias Toward Features with Many Categories

**Issue:**
Decision trees have a bias toward features with many categories because they can provide more splits and potentially reduce impurity more than features with fewer categories, even if they're not more informative.

**How advancements solved this:**
- **Feature importance weighting**: Adjusting the importance calculation
- **Splitting criteria modifications**: Using criteria that account for the number of categories
- **Ensemble methods**: Reducing the impact of individual biased trees
- **Feature preprocessing**: Encoding categorical variables appropriately

```python
# Problem: Bias toward features with many categories
import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

# Generate synthetic data with 2 informative features and 8 noise features
X, y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=0, random_state=42)

# Convert the first 2 features to categorical with many categories
X[:, 0] = np.random.randint(0, 20, size=len(X))  # 20 categories
X[:, 1] = np.random.randint(0, 10, size=len(X))  # 10 categories

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Train a decision tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Get feature importances
importances = tree.feature_importances_
features = [f"Feature {i}" for i in range(X.shape[1])]

# Plot feature importances
plt.figure(figsize=(12, 6))
plt.bar(features, importances)
plt.title("Feature Importances (Biased toward features with many categories)")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Enhancement: Preprocess categorical features
# Identify categorical columns (first 2)
categorical_cols = [0, 1]
numeric_cols = [i for i in range(X.shape[1]) if i not in categorical_cols]

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Preprocess the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Train a decision tree on processed data
tree_processed = DecisionTreeClassifier(random_state=42)
tree_processed.fit(X_train_processed, y_train)

# Get feature importances from processed data
processed_importances = tree_processed.feature_importances_
processed_features = []
processed_features.extend([f"Feature {i}" for i in numeric_cols])

# Add one-hot encoded categorical feature names
for i in categorical_cols:
    n_categories = len(np.unique(X[:, i]))
    processed_features.extend([f"Feature {i}_cat_{j}" for j in range(n_categories)])

# Plot only the top 15 most important features
top_n = 15
indices = np.argsort(processed_importances)[-top_n:]
plt.figure(figsize=(12, 6))
plt.bar([processed_features[i] for i in indices], [processed_importances[i] for i in indices])
plt.title(f"Top {top_n} Feature Importances (After preprocessing)")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Enhancement: Use Random Forest to reduce bias
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
rf_importances = rf.feature_importances_

# Plot feature importances
plt.figure(figsize=(12, 6))
plt.bar(features, rf_importances)
plt.title("Feature Importances with Random Forest (Reduced bias)")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 3. Problem: Instability

**Issue:**
Decision trees are unstable, meaning small variations in the data can result in a completely different tree structure. This high variance makes them less reliable for interpretation and prediction.

**How advancements solved this:**
- **Ensemble methods**: Combining multiple trees to reduce variance
- **Random Forests**: Using random subsets of features and data
- **Boosting algorithms**: Building trees sequentially to correct errors
- **Bagging**: Training multiple trees on different subsets of data

```python
# Problem: Instability of Decision Trees
import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create two slightly different datasets by removing 10 random samples
np.random.seed(42)
remove_idx1 = np.random.choice(len(X_train), 10, replace=False)
remove_idx2 = np.random.choice(len(X_train), 10, replace=False)

X_train1 = np.delete(X_train, remove_idx1, axis=0)
y_train1 = np.delete(y_train, remove_idx1, axis=0)
X_train2 = np.delete(X_train, remove_idx2, axis=0)
y_train2 = np.delete(y_train, remove_idx2, axis=0)

# Train decision trees on the slightly different datasets
tree1 = DecisionTreeClassifier(max_depth=3, random_state=42)
tree1.fit(X_train1, y_train1)

tree2 = DecisionTreeClassifier(max_depth=3, random_state=42)
tree2.fit(X_train2, y_train2)

# Compare the trees
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plot_tree(tree1, filled=True, feature_names=[f'Feature {i}' for i in range(X.shape[1])], class_names=['Class 0', 'Class 1'])
plt.title("Decision Tree 1")

plt.subplot(1, 2, 2)
plot_tree(tree2, filled=True, feature_names=[f'Feature {i}' for i in range(X.shape[1])], class_names=['Class 0', 'Class 1'])
plt.title("Decision Tree 2")

plt.tight_layout()
plt.show()

# Evaluate the trees
accuracy1 = tree1.score(X_test, y_test)
accuracy2 = tree2.score(X_test, y_test)

print(f"Decision Tree 1 Accuracy: {accuracy1:.4f}")
print(f"Decision Tree 2 Accuracy: {accuracy2:.4f}")

# Enhancement: Random Forest for stability
rf1 = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf1.fit(X_train1, y_train1)

rf2 = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf2.fit(X_train2, y_train2)

# Evaluate the Random Forests
rf_accuracy1 = rf1.score(X_test, y_test)
rf_accuracy2 = rf2.score(X_test, y_test)

print(f"\nRandom Forest 1 Accuracy: {rf_accuracy1:.4f}")
print(f"Random Forest 2 Accuracy: {rf_accuracy2:.4f}")
print(f"Accuracy difference between RF models: {abs(rf_accuracy1 - rf_accuracy2):.4f}")
print(f"Accuracy difference between Decision Trees: {abs(accuracy1 - accuracy2):.4f}")

# Compare feature importances
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
features = [f'Feature {i}' for i in range(X.shape[1])]
x = np.arange(len(features))
width = 0.35

plt.bar(x - width/2, tree1.feature_importances_, width, label='Tree 1')
plt.bar(x + width/2, tree2.feature_importances_, width, label='Tree 2')
plt.title("Feature Importances: Decision Trees")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(x, features, rotation=45)
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(x - width/2, rf1.feature_importances_, width, label='RF 1')
plt.bar(x + width/2, rf2.feature_importances_, width, label='RF 2')
plt.title("Feature Importances: Random Forests")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(x, features, rotation=45)
plt.legend()

plt.tight_layout()
plt.show()
```

### 4. Problem: Difficulty with Linear Relationships

**Issue:**
Decision trees struggle to model linear relationships efficiently. They require many splits to approximate a linear relationship, which leads to deep trees and potential overfitting.

**How advancements solved this:**
- **Ensemble methods**: Combining many trees to approximate linear relationships
- **Gradient boosting**: Building trees sequentially to correct errors
- **Linear model combinations**: Combining trees with linear models
- **Feature engineering**: Creating features that capture linear relationships

```python
# Problem: Difficulty with Linear Relationships
import torch
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate synthetic data with a clear linear relationship
np.random.seed(42)
X = np.random.rand(1000, 1) * 10  # Single feature
y = 2 * X.flatten() + 3 + np.random.normal(0, 1, 1000)  # Linear relationship with noise

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Train a decision tree
tree = DecisionTreeRegressor(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

# Train a linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Train a Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# Train a Gradient Boosting model
gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)

# Make predictions
X_test_sorted = np.sort(X_test, axis=0)
tree_pred = tree.predict(X_test_sorted)
lr_pred = lr.predict(X_test_sorted)
rf_pred = rf.predict(X_test_sorted)
gb_pred = gb.predict(X_test_sorted)

# Plot the results
plt.figure(figsize=(12, 8))
plt.scatter(X_test, y_test, s=10, label='Actual Data')
plt.plot(X_test_sorted, tree_pred, 'r-', label='Decision Tree')
plt.plot(X_test_sorted, lr_pred, 'g-', label='Linear Regression')
plt.plot(X_test_sorted, rf_pred, 'b-', label='Random Forest')
plt.plot(X_test_sorted, gb_pred, 'm-', label='Gradient Boosting')
plt.title("Modeling Linear Relationships")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Calculate MSE
tree_mse = mean_squared_error(y_test, tree.predict(X_test))
lr_mse = mean_squared_error(y_test, lr.predict(X_test))
rf_mse = mean_squared_error(y_test, rf.predict(X_test))
gb_mse = mean_squared_error(y_test, gb.predict(X_test))

print(f"Decision Tree MSE: {tree_mse:.4f}")
print(f"Linear Regression MSE: {lr_mse:.4f}")
print(f"Random Forest MSE: {rf_mse:.4f}")
print(f"Gradient Boosting MSE: {gb_mse:.4f}")

# Enhancement: Feature engineering for tree-based models
# Create polynomial features to capture the linear relationship
X_train_poly = np.hstack([X_train, X_train**2, X_train**3])
X_test_poly = np.hstack([X_test, X_test**2, X_test**3])

# Train a decision tree on polynomial features
tree_poly = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_poly.fit(X_train_poly, y_train)

# Train a Random Forest on polynomial features
rf_poly = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_poly.fit(X_train_poly, y_train)

# Make predictions
X_test_sorted_poly = np.hstack([X_test_sorted, X_test_sorted**2, X_test_sorted**3])
tree_poly_pred = tree_poly.predict(X_test_sorted_poly)
rf_poly_pred = rf_poly.predict(X_test_sorted_poly)

# Plot the results
plt.figure(figsize=(12, 8))
plt.scatter(X_test, y_test, s=10, label='Actual Data')
plt.plot(X_test_sorted, tree_pred, 'r-', label='Decision Tree')
plt.plot(X_test_sorted, tree_poly_pred, 'r--', label='Decision Tree (with Polynomial Features)')
plt.plot(X_test_sorted, rf_pred, 'b-', label='Random Forest')
plt.plot(X_test_sorted, rf_poly_pred, 'b--', label='Random Forest (with Polynomial Features)')
plt.plot(X_test_sorted, lr_pred, 'g-', label='Linear Regression')
plt.title("Improving Tree-based Models with Feature Engineering")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Calculate MSE with polynomial features
tree_poly_mse = mean_squared_error(y_test, tree_poly.predict(X_test_poly))
rf_poly_mse = mean_squared_error(y_test, rf_poly.predict(X_test_poly))

print(f"\nDecision Tree MSE (with Polynomial Features): {tree_poly_mse:.4f}")
print(f"Random Forest MSE (with Polynomial Features): {rf_poly_mse:.4f}")
```

## Project: Titanic Survival Prediction

Now, let's implement a project that demonstrates the tree-based models we've covered:

```python
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_data = pd.read_csv(url)

# Explore the dataset
print("Dataset Shape:", titanic_data.shape)
print("\nFirst 5 rows:")
print(titanic_data.head())

# Check for missing values
print("\nMissing values:")
print(titanic_data.isnull().sum())

# Data preprocessing
# Select features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

# Extract features and target
X = titanic_data[features]
y = titanic_data[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Preprocessing pipeline
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Decision Tree
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Train the model
dt_pipeline.fit(X_train, y_train)

# Evaluate the model
dt_pred = dt_pipeline.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f"\nDecision Tree Accuracy: {dt_accuracy:.4f}")
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, dt_pred))

# Random Forest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
rf_pipeline.fit(X_train, y_train)

# Evaluate the model
rf_pred = rf_pipeline.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"\nRandom Forest Accuracy: {rf_accuracy:.4f}")
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# Gradient Boosting
gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Train the model
gb_pipeline.fit(X_train, y_train)

# Evaluate the model
gb_pred = gb_pipeline.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
print(f"\nGradient Boosting Accuracy: {gb_accuracy:.4f}")
print("\nGradient Boosting Classification Report:")
print(classification_report(y_test, gb_pred))

# Hyperparameter tuning for Random Forest
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 5, 10],
    'classifier__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and model
print(f"\nBest parameters: {grid_search.best_params_}")
best_rf = grid_search.best_estimator_

# Evaluate the best model
best_rf_pred = best_rf.predict(X_test)
best_rf_accuracy = accuracy_score(y_test, best_rf_pred)
print(f"\nBest Random Forest Accuracy: {best_rf_accuracy:.4f}")
print("\nBest Random Forest Classification Report:")
print(classification_report(y_test, best_rf_pred))

# Plot confusion matrix for the best model
cm = confusion_matrix(y_test, best_rf_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Best Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature importance from the best Random Forest
# Extract the classifier from the pipeline
best_rf_classifier = best_rf.named_steps['classifier']

# Get feature names after preprocessing
feature_names = []
for name, transformer, columns in preprocessor.transformers_:
    if name == 'num':
        feature_names.extend(columns)
    elif name == 'cat':
        # Get one-hot encoded feature names
        cat_features = transformer.named_steps['onehot'].get_feature_names_out(columns)
        feature_names.extend(cat_features)

# Get feature importances
importances = best_rf_classifier.feature_importances_

# Sort feature importances
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.title("Feature Importances - Best Random Forest")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.xlim([-1, len(importances)])
plt.tight_layout()
plt.show()

# Cross-validation for model stability
cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Make predictions on new data
# Create a sample passenger
new_passenger = pd.DataFrame({
    'Pclass': [3],
    'Sex': ['male'],
    'Age': [25],
    'SibSp': [0],
    'Parch': [0],
    'Fare': [7.25],
    'Embarked': ['S']
})

# Predict survival
prediction = best_rf.predict(new_passenger)
prediction_proba = best_rf.predict_proba(new_passenger)

print(f"\nPrediction for new passenger: {'Survived' if prediction[0] == 1 else 'Did not survive'}")
print(f"Survival probability: {prediction_proba[0][1]:.4f}")
```

This project demonstrates how to use tree-based models for a real-world classification problem. We've covered data preprocessing, model training, evaluation, hyperparameter tuning, and interpretation of results. The Random Forest model with optimized hyperparameters provides the best performance on the Titanic survival prediction task.

## Summary

Tree-based models are a powerful class of machine learning algorithms that have become essential tools in the data scientist's toolkit. In this lesson, we've covered:

1. **Decision Trees**: The fundamental building block that uses a tree-like structure for decision making
2. **Random Forest**: An ensemble method that combines multiple decision trees to reduce overfitting and improve accuracy
3. **Gradient Boosting Machines**: A sequential ensemble method that builds trees to correct errors of previous trees
4. **XGBoost**: An optimized implementation of gradient boosting designed for performance and efficiency
5. **LightGBM**: A gradient boosting framework that uses tree-based learning algorithms with faster training speed and lower memory usage

We've also explored how these concepts are applied in real-world scenarios, common problems encountered when using tree-based models, and how advancements in the field have addressed these challenges.

The Titanic survival prediction project provides a practical implementation of these concepts, demonstrating how to build, evaluate, and optimize tree-based models for a classification task.

Understanding tree-based models is crucial for anyone looking to master machine learning, as they provide