1# Day 11 – Regression Models

## What are Regression Models and Why are they Important for Deep Learning?

Regression models are a type of supervised learning algorithm used to predict continuous numerical values. Unlike classification models that predict discrete categories, regression models estimate the relationship between input variables and a continuous output variable.

In deep learning, regression models are important because:

- They form the foundation for many prediction tasks in real-world applications
- They help understand the relationship between input features and target values
- They can be integrated into larger deep learning architectures
- They provide a baseline for more complex models
- They are essential for tasks like forecasting, pricing, and risk assessment

## Essential Subtopics

### 1. Linear Regression

**What is Linear Regression?**
Linear regression is a statistical method that models the relationship between a dependent variable (y) and one or more independent variables (x) by fitting a linear equation to observed data. The simplest form is simple linear regression with one independent variable:

y = β₀ + β₁x + ε

Where:
- y is the dependent variable
- x is the independent variable
- β₀ is the y-intercept
- β₁ is the slope
- ε is the error term

For multiple independent variables, it becomes multiple linear regression:

y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=1, noise=20, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1)

# Define a linear regression model using PyTorch
class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

# Initialize the model
input_dim = X_train.shape[1]
model = LinearRegression(input_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
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

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).numpy()
    
# Inverse transform to get original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.4f}')
print(f'R-squared: {r2:.4f}')

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Print model parameters
print(f'Intercept: {model.linear.bias.item():.4f}')
print(f'Slope: {model.linear.weight.item():.4f}')
```

**When and where we use linear regression:**
- When there is a linear relationship between independent and dependent variables
- When interpreting the relationship between variables is important
- As a baseline model for regression tasks
- When computational efficiency is a priority
- When working with small datasets

**Real-world examples:**
- Predicting house prices based on features like size, location, and number of rooms
- Forecasting sales based on advertising spend
- Estimating the impact of temperature on energy consumption
- Predicting student performance based on study hours

**Problems and Advancements:**
- **Problem**: Linear regression assumes a linear relationship, which may not hold for complex data.
  - **Advancement**: Polynomial regression and spline regression can capture non-linear relationships.
- **Problem**: Sensitive to outliers.
  - **Advancement**: Robust regression methods like RANSAC and Huber regression.
- **Problem**: Cannot handle multicollinearity well.
  - **Advancement**: Regularization techniques like Ridge and Lasso regression.

### 2. Polynomial Regression

**What is Polynomial Regression?**
Polynomial regression is a form of regression analysis in which the relationship between the independent variable x and the dependent variable y is modeled as an nth degree polynomial. It extends linear regression by adding polynomial terms:

y = β₀ + β₁x + β₂x² + ... + βₙxⁿ + ε

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate non-linear data
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(80) * 0.1

# Split the data
X_train, X_test = X[:60], X[20:]
y_train, y_test = y[:60], y[20:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define a polynomial regression model using PyTorch
class PolynomialRegression(nn.Module):
    def __init__(self, input_dim, degree):
        super(PolynomialRegression, self).__init__()
        self.degree = degree
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        # Create polynomial features
        x_poly = torch.cat([x ** i for i in range(1, self.degree + 1)], dim=1)
        return self.linear(x_poly)

# Function to train the model
def train_model(model, X_train, y_train, num_epochs=1000, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model

# Train models with different polynomial degrees
degrees = [1, 2, 3, 5, 10]
models = []
predictions = []

for degree in degrees:
    print(f"\nTraining polynomial regression model with degree {degree}")
    model = PolynomialRegression(input_dim=1, degree=degree)
    model = train_model(model, X_train_tensor, y_train_tensor, num_epochs=2000, lr=0.01)
    models.append(model)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()
        predictions.append(y_pred)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'R-squared: {r2:.4f}')

# Visualize the results
plt.figure(figsize=(15, 10))
plt.scatter(X, y, color='blue', label='Actual data')

# Create a smooth curve for visualization
X_smooth = np.linspace(0, 5, 100).reshape(-1, 1)
X_smooth_tensor = torch.tensor(X_smooth, dtype=torch.float32)

for i, degree in enumerate(degrees):
    model = models[i]
    model.eval()
    with torch.no_grad():
        y_smooth = model(X_smooth_tensor).numpy()
    
    plt.plot(X_smooth, y_smooth, linewidth=2, label=f'Degree {degree}')

plt.title('Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Compare with sklearn implementation
print("\nComparing with sklearn implementation:")
for degree in degrees:
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X_train)
    
    # Fit linear regression
    model_sklearn = LinearRegression()
    model_sklearn.fit(X_poly, y_train)
    
    # Make predictions
    X_test_poly = poly_features.transform(X_test)
    y_pred_sklearn = model_sklearn.predict(X_test_poly)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred_sklearn)
    r2 = r2_score(y_test, y_pred_sklearn)
    
    print(f'\nDegree {degree}:')
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'R-squared: {r2:.4f}')
    print(f'Coefficients: {model_sklearn.coef_}')
    print(f'Intercept: {model_sklearn.intercept_}')
```

**When and where we use polynomial regression:**
- When the relationship between variables is curvilinear
- When domain knowledge suggests a polynomial relationship
- For interpolation within the range of observed data
- When a simple linear model is insufficient

**Real-world examples:**
- Modeling the relationship between temperature and electricity demand
- Predicting crop yield based on rainfall
- Estimating the trajectory of a projectile
- Modeling the growth rate of bacteria

**Problems and Advancements:**
- **Problem**: High-degree polynomials can overfit the data.
  - **Advancement**: Regularization techniques and cross-validation to select optimal polynomial degree.
- **Problem**: Extrapolation beyond the observed data range is unreliable.
  - **Advancement**: Confidence intervals and prediction intervals to quantify uncertainty.
- **Problem**: Computational complexity increases with polynomial degree.
  - **Advancement**: Efficient algorithms and specialized libraries for polynomial regression.

### 3. Regularization Techniques (Ridge, Lasso, Elastic Net)

**What are Regularization Techniques?**
Regularization techniques add a penalty term to the loss function to prevent overfitting by discouraging complex models. The three main types are:

1. **Ridge Regression (L2 regularization)**: Adds penalty equal to the square of the magnitude of coefficients.
   Loss = MSE + α * Σ(βᵢ²)

2. **Lasso Regression (L1 regularization)**: Adds penalty equal to the absolute value of coefficients.
   Loss = MSE + α * Σ|βᵢ|

3. **Elastic Net**: Combines both L1 and L2 penalties.
   Loss = MSE + α₁ * Σ|βᵢ| + α₂ * Σ(βᵢ²)

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic data with multicollinearity
X, y = make_regression(n_samples=200, n_features=20, n_informative=10, noise=0.1, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1)

# Define a linear regression model with regularization
class RegularizedLinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(RegularizedLinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

# Function to train the model with regularization
def train_regularized_model(model, X_train, y_train, alpha, reg_type, num_epochs=1000, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Add regularization term
        if reg_type == 'ridge':
            # L2 regularization
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)
            loss += alpha * l2_reg
        elif reg_type == 'lasso':
            # L1 regularization
            l1_reg = 0
            for param in model.parameters():
                l1_reg += torch.norm(param, 1)
            loss += alpha * l1_reg
        elif reg_type == 'elastic_net':
            # Elastic Net (combination of L1 and L2)
            l1_reg = 0
            l2_reg = 0
            for param in model.parameters():
                l1_reg += torch.norm(param, 1)
                l2_reg += torch.norm(param, 2)
            loss += alpha * (0.5 * l1_reg + 0.5 * l2_reg)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, scaler_y):
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test).numpy()
    
    # Inverse transform to get original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2, y_pred

# Train models with different regularization techniques
input_dim = X_train.shape[1]
alpha = 0.1  # Regularization strength

print("Training Linear Regression (no regularization)")
model_linear = RegularizedLinearRegression(input_dim)
model_linear = train_regularized_model(model_linear, X_train_tensor, y_train_tensor, alpha=0, reg_type='none', num_epochs=1000)
mse_linear, r2_linear, y_pred_linear = evaluate_model(model_linear, X_test_tensor, y_test, scaler_y)

print("\nTraining Ridge Regression (L2 regularization)")
model_ridge = RegularizedLinearRegression(input_dim)
model_ridge = train_regularized_model(model_ridge, X_train_tensor, y_train_tensor, alpha, 'ridge', num_epochs=1000)
mse_ridge, r2_ridge, y_pred_ridge = evaluate_model(model_ridge, X_test_tensor, y_test, scaler_y)

print("\nTraining Lasso Regression (L1 regularization)")
model_lasso = RegularizedLinearRegression(input_dim)
model_lasso = train_regularized_model(model_lasso, X_train_tensor, y_train_tensor, alpha, 'lasso', num_epochs=1000)
mse_lasso, r2_lasso, y_pred_lasso = evaluate_model(model_lasso, X_test_tensor, y_test, scaler_y)

print("\nTraining Elastic Net Regression")
model_elastic = RegularizedLinearRegression(input_dim)
model_elastic = train_regularized_model(model_elastic, X_train_tensor, y_train_tensor, alpha, 'elastic_net', num_epochs=1000)
mse_elastic, r2_elastic, y_pred_elastic = evaluate_model(model_elastic, X_test_tensor, y_test, scaler_y)

# Compare the models
print("\nModel Comparison:")
print(f"Linear Regression - MSE: {mse_linear:.4f}, R-squared: {r2_linear:.4f}")
print(f"Ridge Regression - MSE: {mse_ridge:.4f}, R-squared: {r2_ridge:.4f}")
print(f"Lasso Regression - MSE: {mse_lasso:.4f}, R-squared: {r2_lasso:.4f}")
print(f"Elastic Net - MSE: {mse_elastic:.4f}, R-squared: {r2_elastic:.4f}")

# Visualize the coefficients
plt.figure(figsize=(12, 8))

# Get coefficients from each model
coefs_linear = model_linear.linear.weight.data.numpy().flatten()
coefs_ridge = model_ridge.linear.weight.data.numpy().flatten()
coefs_lasso = model_lasso.linear.weight.data.numpy().flatten()
coefs_elastic = model_elastic.linear.weight.data.numpy().flatten()

# Plot coefficients
plt.subplot(2, 2, 1)
plt.bar(range(len(coefs_linear)), coefs_linear)
plt.title('Linear Regression Coefficients')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.bar(range(len(coefs_ridge)), coefs_ridge)
plt.title('Ridge Regression Coefficients')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.bar(range(len(coefs_lasso)), coefs_lasso)
plt.title('Lasso Regression Coefficients')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.bar(range(len(coefs_elastic)), coefs_elastic)
plt.title('Elastic Net Coefficients')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.grid(True)

plt.tight_layout()
plt.show()

# Compare coefficient magnitudes
plt.figure(figsize=(12, 6))
plt.plot(range(len(coefs_linear)), coefs_linear, 'o-', label='Linear')
plt.plot(range(len(coefs_ridge)), coefs_ridge, 'o-', label='Ridge')
plt.plot(range(len(coefs_lasso)), coefs_lasso, 'o-', label='Lasso')
plt.plot(range(len(coefs_elastic)), coefs_elastic, 'o-', label='Elastic Net')
plt.title('Comparison of Coefficients')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.legend()
plt.grid(True)
plt.show()

# Test different alpha values for Ridge regression
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_mses = []
ridge_r2s = []

print("\nTesting different alpha values for Ridge Regression:")
for alpha in alphas:
    print(f"\nAlpha = {alpha}")
    model = RegularizedLinearRegression(input_dim)
    model = train_regularized_model(model, X_train_tensor, y_train_tensor, alpha, 'ridge', num_epochs=1000)
    mse, r2, _ = evaluate_model(model, X_test_tensor, y_test, scaler_y)
    ridge_mses.append(mse)
    ridge_r2s.append(r2)
    print(f"MSE: {mse:.4f}, R-squared: {r2:.4f}")

# Plot the effect of alpha on model performance
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(alphas, ridge_mses, 'o-')
plt.xscale('log')
plt.title('Ridge Regression: MSE vs. Alpha')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Mean Squared Error')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(alphas, ridge_r2s, 'o-')
plt.xscale('log')
plt.title('Ridge Regression: R-squared vs. Alpha')
plt.xlabel('Alpha (log scale)')
plt.ylabel('R-squared')
plt.grid(True)

plt.tight_layout()
plt.show()
```

**When and where we use regularization techniques:**
- When dealing with multicollinearity in the data
- When the number of features is large compared to the number of samples
- When overfitting is a concern
- When feature selection is desired (especially with Lasso)
- When model interpretability is important

**Real-world examples:**
- Predicting house prices with many correlated features
- Financial forecasting with many economic indicators
- Medical diagnosis with numerous test results
- Customer churn prediction with many behavioral features

**Problems and Advancements:**
- **Problem**: Selecting the right regularization strength (alpha) is challenging.
  - **Advancement**: Cross-validation techniques to find optimal hyperparameters.
- **Problem**: Lasso can be unstable with highly correlated features.
  - **Advancement**: Elastic Net combines benefits of both L1 and L2 regularization.
- **Problem**: Regularization may underfit if alpha is too large.
  - **Advancement**: Adaptive regularization methods that adjust penalty per feature.

### 4. Support Vector Regression (SVR)

**What is Support Vector Regression?**
Support Vector Regression (SVR) is a regression algorithm that uses the same principles as Support Vector Machines (SVM) for classification. Instead of trying to fit a hyperplane that separates classes, SVR tries to fit a hyperplane that contains as many data points as possible within a margin (epsilon-insensitive tube).

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic data
X, y = make_regression(n_samples=200, n_features=1, noise=15, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1)

# Define a simple SVR model using PyTorch
class SVRModel(nn.Module):
    def __init__(self, input_dim):
        super(SVRModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

# Define epsilon-insensitive loss function
class EpsilonInsensitiveLoss(nn.Module):
    def __init__(self, epsilon=0.1):
        super(EpsilonInsensitiveLoss, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, outputs, targets):
        errors = torch.abs(outputs - targets)
        loss = torch.mean(torch.max(torch.zeros_like(errors), errors - self.epsilon))
        return loss

# Function to train the SVR model
def train_svr_model(model, X_train, y_train, epsilon=0.1, C=1.0, num_epochs=1000, lr=0.01):
    criterion = EpsilonInsensitiveLoss(epsilon)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Add L2 regularization (equivalent to C in SVR)
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param, 2)
        loss += (1/C) * l2_reg
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, scaler_y):
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test).numpy()
    
    # Inverse transform to get original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2, y_pred

# Train the SVR model
print("Training SVR Model (simplified implementation)")
input_dim = X_train.shape[1]
model_svr = SVRModel(input_dim)
model_svr = train_svr_model(model_svr, X_train_tensor, y_train_tensor, epsilon=0.1, C=1.0, num_epochs=1000)
mse_svr, r2_svr, y_pred_svr = evaluate_model(model_svr, X_test_tensor, y_test, scaler_y)

print(f'Mean Squared Error: {mse_svr:.4f}')
print(f'R-squared: {r2_svr:.4f}')

# Compare with sklearn SVR implementation
print("\nComparing with sklearn SVR implementation:")
# Try different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel in kernels:
    svr = SVR(kernel=kernel, C=1.0, epsilon=0.1)
    svr.fit(X_train_scaled, y_train_scaled)
    y_pred_scaled = svr.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nKernel: {kernel}")
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'R-squared: {r2:.4f}')

# Visualize the results for the best kernel (RBF)
svr_rbf = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr_rbf.fit(X_train_scaled, y_train_scaled)

# Create a smooth curve for visualization
X_smooth = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_smooth_scaled = scaler_X.transform(X_smooth)
y_smooth_scaled = svr_rbf.predict(X_smooth_scaled)
y_smooth = scaler_y.inverse_transform(y_smooth_scaled.reshape(-1, 1)).flatten()

# Plot the results
plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_smooth, y_smooth, color='red', linewidth=2, label='SVR (RBF)')
plt.title('Support Vector Regression (RBF Kernel)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Test different C and epsilon values
print("\nTesting different C and epsilon values:")
C_values = [0.1, 1, 10, 100]
epsilon_values = [0.01, 0.1, 0.5, 1.0]

results = []
for C in C_values:
    for epsilon in epsilon_values:
        svr = SVR(kernel='rbf', C=C, epsilon=epsilon)
        svr.fit(X_train_scaled, y_train_scaled)
        y_pred_scaled = svr.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'C': C,
            'epsilon': epsilon,
            'MSE': mse,
            'R2': r2
        })

# Convert results to DataFrame for easier visualization
import pandas as pd
results_df = pd.DataFrame(results)

# Plot the effect of C and epsilon on model performance
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for epsilon in epsilon_values:
    subset = results_df[results_df['epsilon'] == epsilon]
    plt.plot(subset['C'], subset['MSE'], 'o-', label=f'epsilon={epsilon}')
plt.xscale('log')
plt.title('SVR: MSE vs. C for different epsilon values')
plt.xlabel('C (log scale)')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for epsilon in epsilon_values:
    subset = results_df[results_df['epsilon'] == epsilon]
    plt.plot(subset['C'], subset['R2'], 'o-', label=f'epsilon={epsilon}')
plt.xscale('log')
plt.title('SVR: R-squared vs. C for different epsilon values')
plt.xlabel('C (log scale)')
plt.ylabel('R-squared')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Find the best parameters
best_result = results_df.loc[results_df['R2'].idxmax()]
print(f"\nBest parameters:")
print(f"C: {best_result['C']}")
print(f"Epsilon: {best_result['epsilon']}")
print(f"MSE: {best_result['MSE']:.4f}")
print(f"R-squared: {best_result['R2']:.4f}")
```

**When and where we use Support Vector Regression:**
- When dealing with non-linear relationships
- When the data has outliers
- When working in high-dimensional spaces
- When the number of dimensions is greater than the number of samples
- When model interpretability is not the primary concern

**Real-world examples:**
- Predicting stock prices with non-linear patterns
- Estimating house prices with complex feature interactions
- Forecasting energy consumption based on weather conditions
- Modeling biological processes with complex relationships

**Problems and Advancements:**
- **Problem**: SVR can be computationally expensive for large datasets.
  - **Advancement**: Approximation methods and stochastic gradient descent variants.
- **Problem**: Selecting the right kernel and hyperparameters is challenging.
  - **Advancement**: Automated hyperparameter tuning and kernel selection methods.
- **Problem**: SVR doesn't provide probability estimates.
  - **Advancement**: Bayesian variants of SVR that provide uncertainty estimates.

### 5. Decision Tree Regression

**What is Decision Tree Regression?**
Decision Tree Regression uses a decision tree to predict continuous values. It works by splitting the data into subsets based on the value of input features, creating a tree-like model of decisions. Each leaf node represents a numerical value, which is the average of the target values of the training samples that reach that leaf.

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic data
X, y = make_regression(n_samples=500, n_features=5, noise=10, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define a simple decision tree node
class DecisionTreeNode:
    def __init__(self, feature_idx=None, threshold=None, value=None, left=None, right=None):
        self.feature_idx = feature_idx  # Index of feature to split on
        self.threshold = threshold     # Threshold value for the split
        self.value = value            # Predicted value if this is a leaf node
        self.left = left              # Left child node
        self.right = right            # Right child node
        self.is_leaf = value is not None  # Is this a leaf node?
    
    def predict(self, x):
        if self.is_leaf:
            return self.value
        else:
            if x[self.feature_idx] <= self.threshold:
                return self.left.predict(x)
            else:
                return self.right.predict(x)

# Define a decision tree regressor
class DecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        # Stopping conditions
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            return DecisionTreeNode(value=np.mean(y))
        
        # Find the best split
        best_feature_idx, best_threshold = self._find_best_split(X, y)
        
        if best_feature_idx is None:
            return DecisionTreeNode(value=np.mean(y))
        
        # Split the data
        left_idx = X[:, best_feature_idx] <= best_threshold
        right_idx = ~left_idx
        
        X_left, y_left = X[left_idx], y[left_idx]
        X_right, y_right = X[right_idx], y[right_idx]
        
        # Recursively build the left and right subtrees
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        
        return DecisionTreeNode(
            feature_idx=best_feature_idx,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
    
    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_mse = float('inf')
        best_feature_idx = None
        best_threshold = None
        
        # Try each feature
        for feature_idx in range(n_features):
            # Get unique values for this feature
            unique_values = np.unique(X[:, feature_idx])
            
            # Try each unique value as a potential threshold
            for threshold in unique_values:
                # Split the data
                left_idx = X[:, feature_idx] <= threshold
                right_idx = ~left_idx
                
                # Skip if one side is empty
                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue
                
                # Calculate MSE for this split
                y_left, y_right = y[left_idx], y[right_idx]
                mse_left = np.mean((y_left - np.mean(y_left)) ** 2)
                mse_right = np.mean((y_right - np.mean(y_right)) ** 2)
                mse = (len(y_left) * mse_left + len(y_right) * mse_right) / len(y)
                
                # Update best split if this one is better
                if mse < best_mse:
                    best_mse = mse
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold
    
    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.root.predict(x))
        return np.array(predictions)

# Train the decision tree model
print("Training Decision Tree Regression Model")
tree_model = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
tree_model.fit(X_train, y_train)

# Make predictions
y_pred = tree_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.4f}')
print(f'R-squared: {r2:.4f}')

# Compare with sklearn implementation
print("\nComparing with sklearn Decision Tree:")
sklearn_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
sklearn_tree.fit(X_train, y_train)
y_pred_sklearn = sklearn_tree.predict(X_test)

mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)

print(f'Mean Squared Error: {mse_sklearn:.4f}')
print(f'R-squared: {r2_sklearn:.4f}')

# Visualize the sklearn tree
plt.figure(figsize=(20, 10))
plot_tree(sklearn_tree, filled=True, feature_names=[f'Feature {i}' for i in range(X.shape[1])])
plt.title("Decision Tree Visualization")
plt.show()

# Test different max_depth values
max_depths = [1, 2, 3, 5, 10, None]
depth_mses = []
depth_r2s = []

print("\nTesting different max_depth values:")
for depth in max_depths:
    if depth is None:
        depth_str = "None"
        depth = 100  # Use a large number instead of None
    else:
        depth_str = str(depth)
    
    print(f"\nMax Depth: {depth_str}")
    tree = DecisionTreeRegressor(max_depth=depth, min_samples_split=2)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    depth_mses.append(mse)
    depth_r2s.append(r2)
    
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'R-squared: {r2:.4f}')

# Plot the effect of max_depth on model performance
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot([str(d) for d in max_depths], depth_mses, 'o-')
plt.title('Decision Tree: MSE vs. Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Mean Squared Error')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot([str(d) for d in max_depths], depth_r2s, 'o-')
plt.title('Decision Tree: R-squared vs. Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('R-squared')
plt.grid(True)

plt.tight_layout()
plt.show()

# Feature importance
print("\nFeature Importance:")
feature_importance = np.zeros(X_train.shape[1])
self._compute_feature_importance(tree_model.root, feature_importance)
feature_importance = feature_importance / np.sum(feature_importance)  # Normalize

for i, importance in enumerate(feature_importance):
    print(f'Feature {i}: {importance:.4f}')

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance)
plt.title('Feature Importance')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.xticks(range(len(feature_importance)))
plt.grid(True, axis='y')
plt.show()

# Compare with Random Forest
print("\nComparing with Random Forest:")
rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Mean Squared Error: {mse_rf:.4f}')
print(f'R-squared: {r2_rf:.4f}')

# Plot feature importance for Random Forest
plt.figure(figsize=(10, 6))
plt.bar(range(len(rf.feature_importances_)), rf.feature_importances_)
plt.title('Random Forest Feature Importance')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.xticks(range(len(rf.feature_importances_)))
plt.grid(True, axis='y')
plt.show()

# Helper function to compute feature importance
def _compute_feature_importance(node, importance):
    if node.is_leaf:
        return
    
    # This is a split node, so it contributes to importance
    importance[node.feature_idx] += 1
    
    # Recursively compute importance for children
    if node.left is not None:
        _compute_feature_importance(node.left, importance)
    if node.right is not None:
        _compute_feature_importance(node.right, importance)
```

**When and where we use Decision Tree Regression:**
- When interpretability of the model is important
- When dealing with non-linear relationships
- When there are complex interactions between features
- When feature selection is desired
- When working with both numerical and categorical data

**Real-world examples:**
- Predicting customer lifetime value based on purchase history
- Estimating real estate prices based on property features
- Forecasting sales based on marketing spend and economic indicators
- Predicting equipment failure based on sensor readings

**Problems and Advancements:**
- **Problem**: Decision trees are prone to overfitting.
  - **Advancement**: Pruning techniques and ensemble methods like Random Forest.
- **Problem**: Small changes in data can lead to completely different trees.
  - **Advancement**: Ensemble methods that average multiple trees.
- **Problem**: Decision trees can create biased trees if some classes dominate.
  - **Advancement**: Balanced decision trees and techniques like SMOTE.

### 6. Ensemble Methods for Regression

**What are Ensemble Methods for Regression?**
Ensemble methods combine multiple regression models to produce a more accurate and robust prediction than any individual model. Common ensemble methods for regression include:

1. **Bagging (Bootstrap Aggregating)**: Trains multiple models on different random subsets of the data and averages their predictions.
2. **Random Forest**: An ensemble of decision trees trained using bagging with random feature selection.
3. **Boosting**: Trains models sequentially, with each new model focusing on the errors of the previous ones.
4. **Stacking**: Combines multiple models using another model (meta-learner) to make the final prediction.

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1)

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Function to train a model
def train_model(model, X_train, y_train, num_epochs=100, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step
        
        if (epoch+1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model

# Function to evaluate a model
def evaluate_model(model, X_test, y_test, scaler_y):
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test).numpy()
    
    # Inverse transform to get original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2, y_pred

# 1. Bagging (Bootstrap Aggregating)
print("\n=== Bagging (Bootstrap Aggregating) ===")
n_models = 10
bagging_models = []
bagging_predictions = []

input_dim = X_train.shape[1]

for i in range(n_models):
    print(f"\nTraining model {i+1}/{n_models}")
    
    # Bootstrap sample
    indices = torch.randint(0, len(X_train_tensor), (len(X_train_tensor),))
    X_bootstrap = X_train_tensor[indices]
    y_bootstrap = y_train_tensor[indices]
    
    # Train model
    model = SimpleNN(input_dim)
    model = train_model(model, X_bootstrap, y_bootstrap, num_epochs=100, lr=0.01)
    bagging_models.append(model)
    
    # Make predictions
    mse, r2, y_pred = evaluate_model(model, X_test_tensor, y_test, scaler_y)
    bagging_predictions.append(y_pred)
    
    print(f'Model {i+1} - MSE: {mse:.4f}, R-squared: {r2:.4f}')

# Average predictions from all models
bagging_predictions = np.array(bagging_predictions)
bagging_avg_prediction = np.mean(bagging_predictions, axis=0)

# Calculate metrics for bagging ensemble
bagging_mse = mean_squared_error(y_test, bagging_avg_prediction)
bagging_r2 = r2_score(y_test, bagging_avg_prediction)

print(f'\nBagging Ensemble - MSE: {bagging_mse:.4f}, R-squared: {bagging_r2:.4f}')

# 2. Random Forest
print("\n=== Random Forest ===")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

print(f'Random Forest - MSE: {rf_mse:.4f}, R-squared: {rf_r2:.4f}')

# 3. Gradient Boosting
print("\n=== Gradient Boosting ===")
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

gb_mse = mean_squared_error(y_test, y_pred_gb)
gb_r2 = r2_score(y_test, y_pred_gb)

print(f'Gradient Boosting - MSE: {gb_mse:.4f}, R-squared: {gb_r2:.4f}')

# 4. Stacking
print("\n=== Stacking ===")
# Base models
base_models = [
    ('linear', LinearRegression()),
    ('tree', DecisionTreeRegressor(max_depth=3, random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=50, random_state=42))
]

# Generate base model predictions
base_predictions_train = []
base_predictions_test = []

for name, model in base_models:
    print(f"\nTraining {name} model")
    model.fit(X_train, y_train)
    
    # Predictions on training data (for meta-learner)
    pred_train = model.predict(X_train)
    base_predictions_train.append(pred_train)
    
    # Predictions on test data
    pred_test = model.predict(X_test)
    base_predictions_test.append(pred_test)
    
    # Evaluate individual model
    mse = mean_squared_error(y_test, pred_test)
    r2 = r2_score(y_test, pred_test)
    print(f'{name} - MSE: {mse:.4f}, R-squared: {r2:.4f}')

# Create meta-features
meta_features_train = np.column_stack(base_predictions_train)
meta_features_test = np.column_stack(base_predictions_test)

# Train meta-learner
print("\nTraining meta-learner")
meta_learner = LinearRegression()
meta_learner.fit(meta_features_train, y_train)

# Make final predictions
stacking_predictions = meta_learner.predict(meta_features_test)

# Evaluate stacking ensemble
stacking_mse = mean_squared_error(y_test, stacking_predictions)
stacking_r2 = r2_score(y_test, stacking_predictions)

print(f'Stacking Ensemble - MSE: {stacking_mse:.4f}, R-squared: {stacking_r2:.4f}')

# Compare all models
print("\n=== Model Comparison ===")
print(f'Linear Regression - MSE: {mean_squared_error(y_test, LinearRegression().fit(X_train, y_train).predict(X_test)):.4f}')
print(f'Decision Tree - MSE: {mean_squared_error(y_test, DecisionTreeRegressor(max_depth=3, random_state=42).fit(X_train, y_train).predict(X_test)):.4f}')
print(f'Bagging Ensemble - MSE: {bagging_mse:.4f}')
print(f'Random Forest - MSE: {rf_mse:.4f}')
print(f'Gradient Boosting - MSE: {gb_mse:.4f}')
print(f'Stacking Ensemble - MSE: {stacking_mse:.4f}')

# Visualize the results
plt.figure(figsize=(15, 8))

# Plot actual vs predicted for each model
models = [
    ('Linear Regression', LinearRegression().fit(X_train, y_train).predict(X_test)),
    ('Decision Tree', DecisionTreeRegressor(max_depth=3, random_state=42).fit(X_train, y_train).predict(X_test)),
    ('Bagging', bagging_avg_prediction),
    ('Random Forest', y_pred_rf),
    ('Gradient Boosting', y_pred_gb),
    ('Stacking', stacking_predictions)
]

for i, (name, predictions) in enumerate(models):
    plt.subplot(2, 3, i+1)
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(name)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True)

plt.tight_layout()
plt.show()

# Plot feature importance for tree-based models
plt.figure(figsize=(15, 5))

# Random Forest feature importance
plt.subplot(1, 2, 1)
plt.bar(range(len(rf.feature_importances_)), rf.feature_importances_)
plt.title('Random Forest Feature Importance')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.xticks(range(len(rf.feature_importances_)))
plt.grid(True, axis='y')

# Gradient Boosting feature importance
plt.subplot(1, 2, 2)
plt.bar(range(len(gb.feature_importances_)), gb.feature_importances_)
plt.title('Gradient Boosting Feature Importance')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.xticks(range(len(gb.feature_importances_)))
plt.grid(True, axis='y')

plt.tight_layout()
plt.show()

# Plot residuals for each model
plt.figure(figsize=(15, 8))

for i, (name, predictions) in enumerate(models):
    plt.subplot(2, 3, i+1)
    residuals = y_test - predictions
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'{name} Residuals')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.grid(True)

plt.tight_layout()
plt.show()
```

**When and where we use Ensemble Methods for Regression:**
- When individual models have high variance (overfitting)
- When individual models have high bias (underfitting)
- When computational resources allow training multiple models
- When model accuracy is more important than interpretability
- When working with complex datasets with non-linear relationships

**Real-world examples:**
- Predicting house prices with multiple complex features
- Forecasting stock prices using multiple indicators
- Estimating customer lifetime value with diverse data sources
- Predicting energy consumption with weather and economic factors

**Problems and Advancements:**
- **Problem**: Ensemble methods can be computationally expensive.
  - **Advancement**: Parallel computing and distributed training techniques.
- **Problem**: Ensemble methods can be difficult to interpret.
  - **Advancement**: Techniques like SHAP and LIME for explaining ensemble models.
- **Problem**: Ensemble methods may not always improve performance if base models are too similar.
  - **Advancement**: Diversity-promoting techniques and careful selection of base models.

## Applications in Deep Learning

### 1. Neural Network Regression

Neural networks can be used for regression tasks by designing the output layer to produce continuous values:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1)

# Define a neural network for regression
class RegressionNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate=0.2):
        super(RegressionNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Define different neural network architectures
architectures = [
    {'name': 'Shallow', 'hidden_sizes': [32]},
    {'name': 'Medium', 'hidden_sizes': [64, 32]},
    {'name': 'Deep', 'hidden_sizes': [128, 64, 32]},
    {'name': 'Very Deep', 'hidden_sizes': [256, 128, 64, 32]}
]

# Train and evaluate different architectures
results = []

for arch in architectures:
    print(f"\nTraining {arch['name']} Neural Network")
    
    # Initialize model
    model = RegressionNN(input_size=X_train.shape[1], hidden_sizes=arch['hidden_sizes'])
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 200
    train_losses = []
    
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        train_losses.append(loss.item())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).numpy()
    
    # Inverse transform to get original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'{arch["name"]} Neural Network - MSE: {mse:.4f}, R-squared: {r2:.4f}')
    
    results.append({
        'name': arch['name'],
        'model': model,
        'train_losses': train_losses,
        'mse': mse,
        'r2': r2,
        'predictions': y_pred
    })

# Compare the models
plt.figure(figsize=(15, 8))

# Plot training losses
plt.subplot(2, 2, 1)
for result in results:
    plt.plot(result['train_losses'], label=result['name'])
plt.title('Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot MSE comparison
plt.subplot(2, 2, 2)
names = [result['name'] for result in results]
mses = [result['mse'] for result in results]
plt.bar(names, mses)
plt.title('MSE Comparison')
plt.xlabel('Model')
plt.ylabel('MSE')
plt.grid(True, axis='y')

plt.tight_layout()
plt.show()

# Plot actual vs predicted for the best model
best_result = min(results, key=lambda x: x['mse'])
plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_result['predictions'], alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'Actual vs Predicted ({best_result["name"]} Neural Network)')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True)
plt.show()

# Plot residuals
plt.figure(figsize=(10, 6))
residuals = y_test - best_result['predictions']
plt.scatter(best_result['predictions'], residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title(f'Residuals ({best_result["name"]} Neural Network)')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# Advanced: Add regularization to prevent overfitting
print("\n=== Neural Network with Regularization ===")

class RegularizedRegressionNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate=0.2, l1_lambda=0.001, l2_lambda=0.001):
        super(RegularizedRegressionNN, self).__init__()
        
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
    def l1_regularization(self):
        l1_reg = 0
        for param in self.model.parameters():
            if param.requires_grad:
                l1_reg += torch.sum(torch.abs(param))
        return self.l1_lambda * l1_reg
    
    def l2_regularization(self):
        l2_reg = 0
        for param in self.model.parameters():
            if param.requires_grad:
                l2_reg += torch.sum(param ** 2)
        return self.l2_lambda * l2_reg

# Train regularized model
print("\nTraining Regularized Neural Network")
regularized_model = RegularizedRegressionNN(
    input_size=X_train.shape[1], 
    hidden_sizes=[128, 64, 32],
    dropout_rate=0.3,
    l1_lambda=0.001,
    l2_lambda=0.001
)

criterion = nn.MSELoss()
optimizer = optim.Adam(regularized_model.parameters(), lr=0.001)

# Training loop
num_epochs = 200
train_losses = []

for epoch in range(num_epochs):
    # Forward pass
    outputs = regularized_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Add regularization
    loss += regularized_model.l1_regularization()
    loss += regularized_model.l2_regularization()
    
    train_losses.append(loss.item())
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the regularized model
regularized_model.eval()
with torch.no_grad():
    y_pred_scaled = regularized_model(X_test_tensor).numpy()

# Inverse transform to get original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Regularized Neural Network - MSE: {mse:.4f}, R-squared: {r2:.4f}')

# Compare with the best unregularized model
print(f"\nComparison:")
print(f'Best Unregularized Model ({best_result["name"]}) - MSE: {best_result["mse"]:.4f}, R-squared: {best_result["r2"]:.4f}')
print(f'Regularized Model - MSE: {mse:.4f}, R-squared: {r2:.4f}')

# Plot training losses
plt.figure(figsize=(10, 6))
plt.plot(best_result['train_losses'], label=f'{best_result["name"]} (Unregularized)')
plt.plot(train_losses, label='Regularized')
plt.title('Training Losses Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
```

### 2. Convolutional Neural Networks for Regression

CNNs are not just for image classification; they can also be used for regression tasks, especially when working with image data:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create a synthetic image dataset for regression
# Each image will have a certain number of shapes, and the target is the count
class ShapeCountDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=64, max_shapes=10):
        self.num_samples = num_samples
        self.image_size = image_size
        self.max_shapes = max_shapes
        
        # Generate images and labels
        self.images = []
        self.labels = []
        
        for _ in range(num_samples):
            # Random number of shapes
            num_shapes = np.random.randint(1, max_shapes + 1)
            self.labels.append(num_shapes)
            
            # Create blank image
            image = np.zeros((image_size, image_size), dtype=np.float32)
            
            # Add random shapes
            for _ in range(num_shapes):
                # Random position
                x, y = np.random.randint(0, image_size - 10, 2)
                size = np.random.randint(5, 15)
                
                # Random shape type
                shape_type = np.random.choice(['circle', 'square', 'triangle'])
                
                # Add shape to image
                if shape_type == 'circle':
                    for i in range(image_size):
                        for j in range(image_size):
                            if (i - x) ** 2 + (j - y) ** 2 <= size ** 2:
                                image[i, j] = 1.0
                elif shape_type == 'square':
                    image[x:x+size, y:y+size] = 1.0
                elif shape_type == 'triangle':
                    for i in range(image_size):
                        for j in range(image_size):
                            if (i >= x and i <= x + size and 
                                j >= y and j <= y + size and 
                                i - x + j - y <= size):
                                image[i, j] = 1.0
            
            self.images.append(image)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Convert to tensor and add channel dimension
        image = torch.tensor(self.images[idx]).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

# Create dataset
print("Creating synthetic image dataset...")
dataset = ShapeCountDataset(num_samples=1000, image_size=64, max_shapes=10)

# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define a CNN for regression
class CNNRegressor(nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(64, 1)
        
    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn6(self.fc3(x)))
        x = self.dropout3(x)
        
        x = self.fc4(x)
        
        return x

# Initialize the model
model = CNNRegressor()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("\nTraining CNN for regression...")
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels.unsqueeze(1))
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Evaluate the model
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions.extend(outputs.squeeze().tolist())
        actuals.extend(labels.tolist())

# Calculate metrics
mse = mean_squared_error(actuals, predictions)
r2 = r2_score(actuals, predictions)

print(f'\nCNN Regression - MSE: {mse:.4f}, R-squared: {r2:.4f}')

# Visualize some predictions
plt.figure(figsize=(15, 10))
for i in range(5):
    image, label = test_dataset[i]
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        pred = model(image.unsqueeze(0)).item()
    
    # Plot
    plt.subplot(2, 5, i+1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f'Actual: {label.item():.0f}')
    plt.axis('off')
    
    plt.subplot(2, 5, i+6)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f'Predicted: {pred:.0f}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(actuals, predictions, alpha=0.5)
plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
plt.title('Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True)
plt.show()

# Plot residuals
residuals = np.array(actuals) - np.array(predictions)
plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.grid(True)
plt.show()

# Visualize feature maps
print("\nVisualizing feature maps...")
model.eval()

# Get a sample image
sample_image, _ = test_dataset[0]
sample_image = sample_image.unsqueeze(0)  # Add batch dimension

# Hook to get feature maps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# Register hooks
model.conv1.register_forward_hook(get_activation('conv1'))
model.conv2.register_forward_hook(get_activation('conv2'))
model.conv3.register_forward_hook(get_activation('conv3'))

# Forward pass
with torch.no_grad():
    output = model(sample_image)

# Visualize feature maps
plt.figure(figsize=(15, 10))

# Original image
plt.subplot(2, 3, 1)
plt.imshow(sample_image.squeeze(), cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Feature maps from first convolutional layer
plt.subplot(2, 3, 2)
plt.imshow(activation['conv1'][0, 0].cpu(), cmap='viridis')
plt.title('Conv1 Feature Map')
plt.axis('off')

# Feature maps from second convolutional layer
plt.subplot(2, 3, 3)
plt.imshow(activation['conv2'][0, 0].cpu(), cmap='viridis')
plt.title('Conv2 Feature Map')
plt.axis('off')

# Feature maps from third convolutional layer
plt.subplot(2, 3, 4)
plt.imshow(activation['conv3'][0, 0].cpu(), cmap='viridis')
plt.title('Conv3 Feature Map')
plt.axis('off')

# Plot actual vs predicted
plt.subplot(2, 3, 5)
plt.scatter(actuals[:100], predictions[:100], alpha=0.5)
plt.plot([min(actuals[:100]), max(actuals[:100])], [min(actuals[:100]), max(actuals[:100])], 'r--')
plt.title('Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True)

# Plot residuals
plt.subplot(2, 3, 6)
plt.scatter(predictions[:100], residuals[:100], alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 3. Recurrent Neural Networks for Time Series Regression

RNNs are well-suited for time series regression tasks, where the data has temporal dependencies:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic time series data
def generate_time_series(n_samples=1000, seq_length=10, n_features=1):
    """Generate synthetic time series data"""
    X = np.zeros((n_samples, seq_length, n_features))
    y = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Base trend
        trend = np.linspace(0, 1, seq_length)
        
        # Seasonality
        seasonality = np.sin(np.linspace(0, 4 * np.pi, seq_length))
        
        # Noise
        noise = np.random.normal(0, 0.1, seq_length)
        
        # Combine components
        series = trend + seasonality + noise
        
        # Add to X
        X[i, :, 0] = series
        
        # Target is the next value in the series
        y[i] = trend[-1] + np.sin(4 * np.pi * (seq_length / (seq_length + 1))) + np.random.normal(0, 0.1)
    
    return X, y

# Generate data
seq_length = 20
n_features = 1
X, y = generate_time_series(n_samples=1000, seq_length=seq_length, n_features=n_features)

# Split data
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define RNN models
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass
        out, _ = self.rnn(x, h0)
        
        # Get the last time step output
        out = out[:, -1, :]
        
        # Fully connected layer
        out = self.fc(out)
        
        return out

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the last time step output
        out = out[:, -1, :]
        
        # Fully connected layer
        out = self.fc(out)
        
        return out

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass
        out, _ = self.gru(x, h0)
        
        # Get the last time step output
        out = out[:, -1, :]
        
        # Fully connected layer
        out = self.fc(out)
        
        return out

# Train and evaluate different models
models = [
    ('SimpleRNN', SimpleRNN(n_features, 64)),
    ('LSTM', LSTM(n_features, 64)),
    ('GRU', GRU(n_features, 64))
]

results = []

for name, model in models:
    print(f"\nTraining {name} model...")
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 50
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        train_losses.append(loss.item())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'{name} - MSE: {mse:.4f}, R-squared: {r2:.4f}')
    
    results.append({
        'name': name,
        'model': model,
        'train_losses': train_losses,
        'mse': mse,
        'r2': r2,
        'predictions': y_pred
    })

# Compare the models
plt.figure(figsize=(15, 8))

# Plot training losses
plt.subplot(2, 2, 1)
for result in results:
    plt.plot(result['train_losses'], label=result['name'])
plt.title('Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot MSE comparison
plt.subplot(2, 2, 2)
names = [result['name'] for result in results]
mses = [result['mse'] for result in results]
plt.bar(names, mses)
plt.title('MSE Comparison')
plt.xlabel('Model')
plt.ylabel('MSE')
plt.grid(True, axis='y')

plt.tight_layout()
plt.show()

# Plot actual vs predicted for the best model
best_result = min(results, key=lambda x: x['mse'])
plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_result['predictions'], alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'Actual vs Predicted ({best_result["name"]})')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True)
plt.show()

# Plot time series predictions
plt.figure(figsize=(15, 10))

# Plot some sample time series with predictions
for i in range(3):
    # Get a sample
    sample_X = X_test[i]
    sample_y = y_test[i]
    sample_pred = best_result['predictions'][i]
    
    # Plot the time series
    plt.subplot(3, 2, i*2+1)
    plt.plot(np.arange(seq_length), sample_X[:, 0], 'b-', label='Input')
    plt.scatter(seq_length, sample_y, color='green', s=100, label='Actual')
    plt.scatter(seq_length, sample_pred, color='red', s=100, label='Predicted')
    plt.title(f'Time Series Sample {i+1}')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Plot the prediction error
    plt.subplot(3, 2, i*2+2)
    error = sample_y - sample_pred
    plt.bar(['Prediction Error'], [error])
    plt.title(f'Prediction Error for Sample {i+1}')
    plt.ylabel('Error')
    plt.grid(True, axis='y')

plt.tight_layout()
plt.show()

# Advanced: Sequence-to-Sequence Prediction
print("\n=== Sequence-to-Sequence Prediction ===")

class Seq2SeqRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Seq2SeqRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.encoder = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.RNN(output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, target_len=1):
        # Encoder
        encoder_output, encoder_hidden = self.encoder(x)
        
        # Initialize decoder input with last encoder output
        decoder_input = encoder_output[:, -1:, :].unsqueeze(1)
        
        # Initialize decoder hidden state with encoder hidden state
        decoder_hidden = encoder_hidden
        
        # Store outputs
        outputs = []
        
        # Generate sequence
        for _ in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            
            # Apply fully connected layer
            output = self.fc(decoder_output)
            outputs.append(output)
            
            # Use output as next input (teacher forcing during training)
            decoder_input = output
        
        # Concatenate outputs
        outputs = torch.cat(outputs, dim=1)
        
        return outputs

# Generate sequence-to-sequence data
def generate_seq2seq_data(n_samples=1000, seq_length=10, pred_length=5, n_features=1):
    """Generate synthetic sequence-to-sequence data"""
    X = np.zeros((n_samples, seq_length, n_features))
    y = np.zeros((n_samples, pred_length, n_features))
    
    for i in range(n_samples):
        # Generate input sequence
        trend = np.linspace(0, 1, seq_length)
        seasonality = np.sin(np.linspace(0, 4 * np.pi, seq_length))
        noise = np.random.normal(0, 0.1, seq_length)
        sequence = trend + seasonality + noise
        
        X[i, :, 0] = sequence
        
        # Generate output sequence (continuation of input)
        trend = np.linspace(1, 1.5, pred_length)
        seasonality = np.sin(np.linspace(4 * np.pi, 6 * np.pi, pred_length))
        noise = np.random.normal(0, 0.1, pred_length)
        sequence = trend + seasonality + noise
        
        y[i, :, 0] = sequence
    
    return X, y

# Generate data
seq_length = 20
pred_length = 5
X_seq, y_seq = generate_seq2seq_data(n_samples=1000, seq_length=seq_length, pred_length=pred_length)

# Split data
train_size = int(0.8 * len(X_seq))
X_train_seq, X_test_seq = X_seq[:train_size], X_seq[train_size:]
y_train_seq, y_test_seq = y_seq[:train_size], y_seq[train_size:]

# Convert to PyTorch tensors
X_train_seq_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_seq_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
X_test_seq_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
y_test_seq_tensor = torch.tensor(y_test_seq, dtype=torch.float32)

# Initialize model
seq2seq_model = Seq2SeqRNN(n_features, 64, n_features)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(seq2seq_model.parameters(), lr=0.001)

# Training loop
print("\nTraining Sequence-to-Sequence model...")
num_epochs = 50

for epoch in range(num_epochs):
    seq2seq_model.train()
    
    # Forward pass
    outputs = seq2seq_model(X_train_seq_tensor, target_len=pred_length)
    loss = criterion(outputs, y_train_seq_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
seq2seq_model.eval()
with torch.no_grad():
    y_pred_seq = seq2seq_model(X_test_seq_tensor, target_len=pred_length).numpy()

# Calculate metrics
mse = mean_squared_error(y_test_seq.flatten(), y_pred_seq.flatten())
r2 = r2_score(y_test_seq.flatten(), y_pred_seq.flatten())

print(f'Sequence-to-Sequence Model - MSE: {mse:.4f}, R-squared: {r2:.4f}')

# Visualize sequence predictions
plt.figure(figsize=(15, 10))

for i in range(3):
    # Get a sample
    sample_X = X_test_seq[i, :, 0]
    sample_y = y_test_seq[i, :, 0]
    sample_pred = y_pred_seq[i, :, 0]
    
    # Create time axis
    time_input = np.arange(seq_length)
    time_output = np.arange(seq_length, seq_length + pred_length)
    
    # Plot the sequence
    plt.subplot(3, 1, i+1)
    plt.plot(time_input, sample_X, 'b-', label='Input')
    plt.plot(time_output, sample_y, 'g-', label='Actual')
    plt.plot(time_output, sample_pred, 'r--', label='Predicted')
    plt.title(f'Sequence Prediction Sample {i+1}')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
```

## Common Problems and Advancements

### 1. Problem: Overfitting in Regression Models

**Issue:**
Regression models, especially complex ones like deep neural networks, can easily overfit the training data, leading to poor generalization on unseen data.

**How advancements solved this:**
- **Regularization techniques**: L1 (Lasso), L2 (Ridge), and Elastic Net regularization add penalty terms to the loss function to prevent large coefficients.
- **Dropout**: Randomly dropping units during training to prevent co-adaptation of features.
- **Early stopping**: Monitoring validation loss and stopping training when it starts to increase.
- **Cross-validation**: Using k-fold cross-validation to get a more robust estimate of model performance.
- **Ensemble methods**: Combining multiple models to reduce variance.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1)

# Define a neural network with dropout
class RegularizedNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate=0.5):
        super(RegularizedNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Define models with different regularization approaches
models = {
    'No Regularization': RegularizedNN(X_train.shape[1], [128, 64, 32], dropout_rate=0.0),
    'Dropout Only': RegularizedNN(X_train.shape[1], [128, 64, 32], dropout_rate=0.5),
    'L2 Regularization': RegularizedNN(X_train.shape[1], [128, 64, 32], dropout_rate=0.0),
    'Dropout + L2': RegularizedNN(X_train.shape[1], [128, 64, 32], dropout_rate=0.5)
}

# Train models with different approaches
results = {}

for name, model in models.items():
    print(f"\nTraining {name} model...")
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    
    if 'L2' in name:
        # Add L2 regularization
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training with early stopping
    num_epochs = 200
    train_losses = []
    val_losses = []
    
    # Split training data for validation
    val_size = int(0.2 * len(X_train_tensor))
    X_train_sub = X_train_tensor[:-val_size]
    y_train_sub = y_train_tensor[:-val_size]
    X_val = X_train_tensor[-val_size:]
    y_val = y_train_tensor[-val_size:]
    
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        outputs = model(X_train_sub)
        loss = criterion(outputs, y_train_sub)
        train_losses.append(loss.item())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
        
        if (epoch+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).numpy()
    
    # Inverse transform to get original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'mse': mse,
        'predictions': y_pred
    }
    
    print(f'{name} - Test MSE: {mse:.4f}')

# Compare models
plt.figure(figsize=(15, 10))

# Plot training losses
plt.subplot(2, 2, 1)
for name, result in results.items():
    plt.plot(result['train_losses'], label=name)
plt.title('Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot validation losses
plt.subplot(2, 2, 2)
for name, result in results.items():
    plt.plot(result['val_losses'], label=name)
plt.title('Validation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot test MSE comparison
plt.figure(figsize=(10, 6))
names = list(results.keys())
mses = [results[name]['mse'] for name in names]
plt.bar(names, mses)
plt.title('Test MSE Comparison')
plt.xlabel('Regularization Method')
plt.ylabel('MSE')
plt.grid(True, axis='y')
plt.show()

# Plot actual vs predicted for the best model
best_model_name = min(results.keys(), key=lambda x: results[x]['mse'])
best_result = results[best_model_name]

plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_result['predictions'], alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'Actual vs Predicted ({best_model_name})')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True)
plt.show()

# Cross-validation for the best approach
print(f"\nPerforming 5-fold cross-validation for {best_model_name}...")

# Define the model architecture based on the best approach
if 'Dropout' in best_model_name:
    dropout_rate = 0.5
else:
    dropout_rate = 0.0

if 'L2' in best_model_name:
    weight_decay = 0.01
else:
    weight_decay = 0.0

# K-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_mses = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled)):
    print(f'Fold {fold+1}')
    
    # Split data
    X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_train_fold, y_val_fold = y_train_scaled[train_idx], y_train_scaled[val_idx]
    
    # Convert to PyTorch tensors
    X_train_fold_tensor = torch.tensor(X_train_fold, dtype=torch.float32)
    y_train_fold_tensor = torch.tensor(y_train_fold, dtype=torch.float32).unsqueeze(1)
    X_val_fold_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
    y_val_fold_tensor = torch.tensor(y_val_fold, dtype=torch.float32).unsqueeze(1)
    
    # Initialize model
    model = RegularizedNN(X_train.shape[1], [128, 64, 32], dropout_rate=dropout_rate)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    
    # Training
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train_fold_tensor)
        loss = criterion(outputs, y_train_fold_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_fold_tensor)
        val_loss = criterion(val_outputs, y_val_fold_tensor)
    
    # Calculate MSE on original scale
    y_val_pred = scaler_y.inverse_transform(val_outputs.numpy())
    y_val_orig = scaler_y.inverse_transform(y_val_fold.numpy())
    mse = mean_squared_error(y_val_orig, y_val_pred)
    
    cv_mses.append(mse)
    print(f'Fold {fold+1} MSE: {mse:.4f}')

print(f'Cross-validation MSE: {np.mean(cv_mses):.4f} (+/- {np.std(cv_mses):.4f})')
```

### 2. Problem: Handling Non-linear Relationships

**Issue:**
Many real-world regression problems involve non-linear relationships between input features and the target variable, which simple linear models cannot capture effectively.

**How advancements solved this:**
- **Polynomial regression**: Adding polynomial terms to capture non-linear relationships.
- **Kernel methods**: Using kernel functions to implicitly map data to higher dimensions.
- **Neural networks**: Using multiple layers with non-linear activation functions to model complex relationships.
- **Decision trees and ensemble methods**: Naturally capturing non-linear decision boundaries.
- **Feature engineering**: Creating new features that capture non-linear aspects of the data.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Generate non-linear data
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(80) * 0.1

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define models for comparison
models = {
    'Linear Regression': make_pipeline(LinearRegression()),
    'Polynomial Regression (degree=2)': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    'Polynomial Regression (degree=3)': make_pipeline(PolynomialFeatures(degree=3), LinearRegression()),
    'Polynomial Regression (degree=5)': make_pipeline(PolynomialFeatures(degree=5), LinearRegression()),
    'Ridge Regression': make_pipeline(PolynomialFeatures(degree=3), Ridge(alpha=0.1)),
    'SVR (RBF)': SVR(kernel='rbf', C=1.0, gamma=0.1),
    'Decision Tree': DecisionTreeRegressor(max_depth=5),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3)
}

# Define a neural network for non-linear regression
class NonLinearNN(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(NonLinearNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Train and evaluate models
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'{name} - MSE: {mse:.4f}, R-squared: {r2:.4f}')
    
    results[name] = {
        'predictions': y_pred,
        'mse': mse,
        'r2': r2
    }

# Train neural network
print("\nTraining Neural Network...")
input_size = X_train.shape[1]
hidden_sizes = [64, 32, 16]
model = NonLinearNN(input_size, hidden_sizes)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate neural network
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy()

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Neural Network - MSE: {mse:.4f}, R-squared: {r2:.4f}')

results['Neural Network'] = {
    'predictions': y_pred,
    'mse': mse,
    'r2': r2
}

# Compare models
plt.figure(figsize=(15, 10))

# Plot actual vs predicted for each model
for i, (name, result) in enumerate(results.items()):
    plt.subplot(3, 4, i+1)
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.scatter(X_test, result['predictions'], color='red', label='Predicted')
    plt.title(name)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Plot MSE comparison
plt.figure(figsize=(12, 6))
names = list(results.keys())
mses = [results[name]['mse'] for name in names]
plt.bar(names, mses)
plt.title('MSE Comparison')
plt.xlabel('Model')
plt.ylabel('MSE')
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y')
plt.show()

# Plot R-squared comparison
plt.figure(figsize=(12, 6))
names = list(results.keys())
r2s = [results[name]['r2'] for name in names]
plt.bar(names, r2s)
plt.title('R-squared Comparison')
plt.xlabel('Model')
plt.ylabel('R-squared')
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y')
plt.show()

# Plot the best model's predictions
best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
best_result = results[best_model_name]

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, best_result['predictions'], color='red', label='Predicted')
plt.title(f'Best Model: {best_model_name}')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Feature engineering for non-linear relationships
print("\n=== Feature Engineering for Non-linear Relationships ===")

# Create additional features
X_enhanced = np.copy(X)
X_enhanced = np.hstack([X_enhanced, X**2, X**3, np.sin(X), np.cos(X), np.sqrt(np.abs(X))])

# Split the enhanced data
X_train_enhanced, X_test_enhanced = train_test_split(X_enhanced, y_test, test_size=0.2, random_state=42)

# Train a linear model on enhanced features
print("\nTraining Linear Regression on Enhanced Features...")
linear_enhanced = LinearRegression()
linear_enhanced.fit(X_train_enhanced, y_train)
y_pred_enhanced = linear_enhanced.predict(X_test_enhanced)

# Calculate metrics
mse_enhanced = mean_squared_error(y_test, y_pred_enhanced)
r2_enhanced = r2_score(y_test, y_pred_enhanced)

print(f'Linear Regression (Enhanced Features) - MSE: {mse_enhanced:.4f}, R-squared: {r2_enhanced:.4f}')

# Compare with original linear regression
print(f'Linear Regression (Original Features) - MSE: {results["Linear Regression"]["mse"]:.4f}, R-squared: {results["Linear Regression"]["r2"]:.4f}')

# Plot comparison
plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, results["Linear Regression"]["predictions"], color='red', label='Linear (Original)')
plt.scatter(X_test, y_pred_enhanced, color='green', label='Linear (Enhanced)')
plt.title('Linear Regression with Enhanced Features')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Advanced: Neural Network with different activation functions
print("\n=== Neural Network with Different Activation Functions ===")

class ActivationFunctionNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation='relu'):
        super(ActivationFunctionNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.01))
            elif activation == 'elu':
                layers.append(nn.ELU())
            
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Train models with different activation functions
activation_functions = ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu']
activation_results = {}

for activation in activation_functions:
    print(f"\nTraining Neural Network with {activation} activation...")
    
    # Initialize model
    model = ActivationFunctionNN(input_size, [64, 32, 16], activation)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Neural Network ({activation}) - MSE: {mse:.4f}, R-squared: {r2:.4f}')
    
    activation_results[activation] = {
        'predictions': y_pred,
        'mse': mse,
        'r2': r2
    }

# Compare activation functions
plt.figure(figsize=(15, 8))

# Plot actual vs predicted for each activation function
for i, (activation, result) in enumerate(activation_results.items()):
    plt.subplot(2, 3, i+1)
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.scatter(X_test, result['predictions'], color='red', label='Predicted')
    plt.title(f'Activation: {activation}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Plot MSE comparison
plt.figure(figsize=(12, 6))
names = list(activation_results.keys())
mses = [activation_results[name]['mse'] for name in names]
plt.bar(names, mses)
plt.title('MSE Comparison by Activation Function')
plt.xlabel('Activation Function')
plt.ylabel('MSE')
plt.grid(True, axis='y')
plt.show()
```

### 3. Problem: High-Dimensional Data

**Issue:**
Regression problems often involve a large number of features, which can lead to overfitting, increased computational complexity, and difficulty in interpretation.

**How advancements solved this:**
- **Feature selection**: Methods to select the most relevant features.
- **Dimensionality reduction**: Techniques like PCA to reduce the number of features while preserving information.
- **Regularization**: L1 (Lasso) regularization that can drive some coefficients to zero, effectively performing feature selection.
- **Tree-based methods**: Random forests and gradient boosting that naturally handle high-dimensional data.
- **Neural network architectures**: Specific architectures designed for high-dimensional data.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Generate high-dimensional data
X, y = make_regression(n_samples=1000, n_features=100, n_informative=20, noise=0.1, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1)

# Define models for high-dimensional data
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.01),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Define a neural network for high-dimensional data
class HighDimNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate=0.5):
        super(HighDimNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Train and evaluate models
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'{name} - MSE: {mse:.4f}, R-squared: {r2:.4f}')
    
    # For Lasso, get selected features
    if name == 'Lasso Regression':
        selected_features = np.where(model.coef_ != 0)[0]
        print(f'Selected features: {len(selected_features)} out of {X_train.shape[1]}')
    
    results[name] = {
        'predictions': y_pred,
        'mse': mse,
        'r2': r2
    }

# Train neural network
print("\nTraining Neural Network...")
input_size = X_train.shape[1]
hidden_sizes = [128, 64, 32]
model = HighDimNN(input_size, hidden_sizes)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate neural network
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy()

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Neural Network - MSE: {mse:.4f}, R-squared: {r2:.4f}')

results['Neural Network'] = {
    'predictions': y_pred,
    'mse': mse,
    'r2': r2
}

# Dimensionality reduction with PCA
print("\n=== Dimensionality Reduction with PCA ===")

# Apply PCA
pca = PCA(n_components=20)  # Reduce to 20 components
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")

# Train models on PCA-transformed data
pca_models = {
    'Linear Regression (PCA)': LinearRegression(),
    'Ridge Regression (PCA)': Ridge(alpha=1.0),
    'Random Forest (PCA)': RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in pca_models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train_pca, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_pca)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'{name} - MSE: {mse:.4f}, R-squared: {r2:.4f}')
    
    results[name] = {
        'predictions': y_pred,
        'mse': mse,
        'r2': r2
    }

# Feature selection with SelectKBest
print("\n=== Feature Selection with SelectKBest ===")

# Select top 20 features
selector = SelectKBest(f_regression, k=20)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Get selected feature indices
selected_features = selector.get_support(indices=True)
print(f"Selected features: {np.sum(selected_features)} out of {X_train.shape[1]}")

# Train models on selected features
selected_models = {
    'Linear Regression (Selected)': LinearRegression(),
    'Ridge Regression (Selected)': Ridge(alpha=1.0),
    'Random Forest (Selected)': RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in selected_models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train_selected, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_selected)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'{name} - MSE: {mse:.4f}, R-squared: {r2:.4f}')
    
    results[name] = {
        'predictions': y_pred,
        'mse': mse,
        'r2': r2
    }

# Compare all models
plt.figure(figsize=(15, 8))

# Plot MSE comparison
plt.subplot(1, 2, 1)
names = list(results.keys())
mses = [results[name]['mse'] for name in names]
plt.bar(names, mses)
plt.title('MSE Comparison')
plt.xlabel('Model')
plt.ylabel('MSE')
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y')

# Plot R-squared comparison
plt.subplot(1, 2, 2)
names = list(results.keys())
r2s = [results[name]['r2'] for name in names]
plt.bar(names, r2s)
plt.title('R-squared Comparison')
plt.xlabel('Model')
plt.ylabel('R-squared')
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y')

plt.tight_layout()
plt.show()

# Plot the best model's predictions
best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
best_result = results[best_model_name]

plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_result['predictions'], alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'Best Model: {best_model_name}')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True)
plt.show()

# Advanced: Autoencoder for dimensionality reduction
print("\n=== Autoencoder for Dimensionality Reduction ===")

class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# Train autoencoder
print("\nTraining Autoencoder...")
encoding_dim = 20
autoencoder = Autoencoder(X_train.shape[1], encoding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs, _ = autoencoder(X_train_tensor)
    loss = criterion(outputs, X_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Get encoded representations
autoencoder.eval()
with torch.no_grad():
    _, X_train_encoded = autoencoder.encoder(X_train_tensor)
    _, X_test_encoded = autoencoder.encoder(X_test_tensor)

# Train a regression model on encoded data
print("\nTraining Regression Model on Encoded Data...")
encoded_model = LinearRegression()
encoded_model.fit(X_train_encoded.numpy(), y_train)

# Make predictions
y_pred_encoded = encoded_model.predict(X_test_encoded.numpy())

# Calculate metrics
mse_encoded = mean_squared_error(y_test, y_pred_encoded)
r2_encoded = r2_score(y_test, y_pred_encoded)

print(f'Regression on Encoded Data - MSE: {mse_encoded:.4f}, R-squared: {r2_encoded:.4f}')

# Add to results
results['Autoencoder + Linear Regression'] = {
    'predictions': y_pred_encoded,
    'mse': mse_encoded,
    'r2': r2_encoded
}

# Compare all approaches
plt.figure(figsize=(15, 8))

# Group models by approach
approaches = {
    'Original': ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Random Forest', 'Neural Network'],
    'PCA': ['Linear Regression (PCA)', 'Ridge Regression (PCA)', 'Random Forest (PCA)'],
    'Feature Selection': ['Linear Regression (Selected)', 'Ridge Regression (Selected)', 'Random Forest (Selected)'],
    'Autoencoder': ['Autoencoder + Linear Regression']
}

# Plot MSE comparison by approach
plt.subplot(1, 2, 1)
x_pos = []
y_mses = []
x_labels = []

for i, (approach, model_names) in enumerate(approaches.items()):
    for j, model_name in enumerate(model_names):
        x_pos.append(i + j * 0.2)
        y_mses.append(results[model_name]['mse'])
        x_labels.append(f"{approach}\n{model_name}")

plt.bar(x_pos, y_mses, width=0.2)
plt.title('MSE Comparison by Approach')
plt.xlabel('Approach and Model')
plt.ylabel('MSE')
plt.xticks(x_pos, x_labels, rotation=45, ha='right')
plt.grid(True, axis='y')

# Plot R-squared comparison by approach
plt.subplot(1, 2, 2)
x_pos = []
y_r2s = []

for i, (approach, model_names) in enumerate(approaches.items()):
    for j, model_name in enumerate(model_names):
        x_pos.append(i + j * 0.2)
        y_r2s.append(results[model_name]['r2'])

plt.bar(x_pos, y_r2s, width=0.2)
plt.title('R-squared Comparison by Approach')
plt.xlabel('Approach and Model')
plt.ylabel('R-squared')
plt.xticks(x_pos, x_labels, rotation=45, ha='right')
plt.grid(True, axis='y')

plt.tight_layout()
plt.show()
```

## Project: House Price Predictor

Now, let's implement a comprehensive house price predictor using the regression techniques we've covered:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load the California Housing dataset
print("Loading California Housing dataset...")
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# Explore the dataset
print("\nDataset Info:")
print(X.info())
print("\nFirst 5 rows:")
print(X.head())
print("\nTarget statistics:")
print(pd.Series(y).describe())

# Add some engineered features
print("\nEngineering features...")
X['RoomsPerPerson'] = X['AveRooms'] / X['AveOccup']
X['BedroomsPerPerson'] = X['AveBedrms'] / X['AveOccup']
X['PeoplePerHousehold'] = X['Population'] / X['Households']
X['IncomePerPerson'] = X['MedInc'] / X['AveOccup']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
print("\nPreprocessing data...")
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Fit and transform the data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_preprocessed, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_preprocessed, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define a neural network for house price prediction
class HousePriceNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate=0.2):
        super(HousePriceNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Define models to compare
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.001),
    'Elastic Net': ElasticNet(alpha=0.001, l1_ratio=0.5),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=10, gamma=0.1)
}

# Train and evaluate models
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R-squared: {r2:.4f}')
    
    results[name] = {
        'pipeline': pipeline,
        'predictions': y_pred,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

# Train neural network
print("\nTraining Neural Network...")
input_size = X_train_preprocessed.shape[1]
hidden_sizes = [128, 64, 32]
model = HousePriceNN(input_size, hidden_sizes)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with early stopping
num_epochs = 200
patience = 20
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

# Split training data for validation
val_size = int(0.2 * len(X_train_tensor))
X_train_sub = X_train_tensor[:-val_size]
y_train_sub = y_train_tensor[:-val_size]
X_val = X_train_tensor[-val_size:]
y_val = y_train_tensor[-val_size:]

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    outputs = model(X_train_sub)
    loss = criterion(outputs, y_train_sub)
    train_losses.append(loss.item())
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        val_losses.append(val_loss.item())
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Load best model
model.load_state_dict(best_model_state)

# Evaluate neural network
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy()

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Neural Network - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R-squared: {r2:.4f}')

results['Neural Network'] = {
    'predictions': y_pred,
    'rmse': rmse,
    'mae': mae,
    'r2': r2
}

# Hyperparameter tuning for the best model
print("\n=== Hyperparameter Tuning ===")

# Find the best model based on RMSE
best_model_name = min(results.keys(), key=lambda x: results[x]['rmse'])
print(f"Best model: {best_model_name}")

if best_model_name == 'Random Forest':
    # Hyperparameter tuning for Random Forest
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }
elif best_model_name == 'Gradient Boosting':
    # Hyperparameter tuning for Gradient Boosting
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7],
        'model__min_samples_split': [2, 5, 10]
    }
elif best_model_name == 'Ridge Regression':
    # Hyperparameter tuning for Ridge Regression
    param_grid = {
        'model__alpha': [0.01, 0.1, 1.0, 10.0]
    }
elif best_model_name == 'Lasso Regression':
    # Hyperparameter tuning for Lasso Regression
    param_grid = {
        'model__alpha': [0.0001, 0.001, 0.01, 0.1]
    }
else:
    print("Hyperparameter tuning not implemented for this model")
    param_grid = {}

if param_grid:
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', models[best_model_name])
    ])
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    print(f"\nPerforming grid search for {best_model_name}...")
    grid_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    
    # Evaluate the best model
    y_pred = grid_search.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Tuned {best_model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R-squared: {r2:.4f}')
    
    results[f'Tuned {best_model_name}'] = {
        'predictions': y_pred,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'model': grid_search.best_estimator_
    }

# Feature importance analysis
print("\n=== Feature Importance Analysis ===")

# Get the best model (tuned if available, otherwise original)
if f'Tuned {best_model_name}' in results:
    best_model = results[f'Tuned {best_model_name}']['model']
else:
    best_model = results[best_model_name]['pipeline']

# Extract feature names after preprocessing
feature_names = []
for name, transformer, columns in preprocessor.transformers_:
    if name == 'num':
        feature_names.extend(columns)
    elif name == 'cat':
        if hasattr(transformer.named_steps['onehot'], 'categories_'):
            categories = transformer.named_steps['onehot'].categories_
            for i, category in enumerate(categories):
                feature_names.append(f"{columns[0]}_{category}")

# Get feature importances if available
if hasattr(best_model.named_steps['model'], 'feature_importances_'):
    importances = best_model.named_steps['model'].feature_importances_
    
    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance_df['Feature'][:15], feature_importance_df['Importance'][:15])
    plt.title('Top 15 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()
    
    print("\nTop 10")