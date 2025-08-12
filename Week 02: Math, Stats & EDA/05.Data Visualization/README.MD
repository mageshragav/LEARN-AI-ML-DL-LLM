# Day 10 – Data Visualization

## What is Data Visualization and Why is it Important for Deep Learning?

Data visualization is the graphical representation of information and data using visual elements like charts, graphs, and maps. It provides an intuitive way to see and understand trends, outliers, and patterns in data, as well as to communicate insights to others.

In deep learning, data visualization is crucial because:

- It helps in understanding the distribution and characteristics of your data before modeling
- It enables detection of patterns, outliers, and anomalies that might affect model performance
- It provides a way to interpret complex model outputs and results
- It facilitates communication of findings to both technical and non-technical stakeholders
- It aids in model debugging and performance evaluation
- It helps in identifying data quality issues and biases

## Essential Subtopics

### 1. Basic Plotting with Matplotlib

**What is Matplotlib?**
Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. It provides a MATLAB-like interface and is one of the most widely used plotting libraries in the Python ecosystem.

**PyTorch Integration:**
```python
import matplotlib.pyplot as plt
import torch
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Generate some data using PyTorch
x = torch.linspace(0, 10, 100)
y = torch.sin(x) + torch.randn(x.shape) * 0.1  # sin with some noise

# Create a simple line plot
plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), y.numpy(), label='Noisy Sin Wave')
plt.title('Simple Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()

# Create a scatter plot
# Generate random data points
num_points = 100
x_scatter = torch.randn(num_points)
y_scatter = 2 * x_scatter + 1 + torch.randn(num_points) * 0.5  # y = 2x + 1 + noise

plt.figure(figsize=(10, 6))
plt.scatter(x_scatter.numpy(), y_scatter.numpy(), alpha=0.7, c='red')
plt.title('Simple Scatter Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()

# Create a histogram
data = torch.randn(1000)  # Standard normal distribution

plt.figure(figsize=(10, 6))
plt.hist(data.numpy(), bins=30, alpha=0.7, color='green', edgecolor='black')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Create a bar chart
categories = ['A', 'B', 'C', 'D', 'E']
values = torch.tensor([23, 45, 56, 78, 32])

plt.figure(figsize=(10, 6))
plt.bar(categories, values.numpy(), color='blue', alpha=0.7)
plt.title('Bar Chart')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.grid(True, axis='y')
plt.show()

# Create a box plot
# Generate data for multiple groups
group1 = torch.randn(100) * 2 + 5
group2 = torch.randn(100) * 1.5 + 7
group3 = torch.randn(100) * 3 + 3

plt.figure(figsize=(10, 6))
plt.boxplot([group1.numpy(), group2.numpy(), group3.numpy()], labels=['Group 1', 'Group 2', 'Group 3'])
plt.title('Box Plot')
plt.xlabel('Groups')
plt.ylabel('Values')
plt.grid(True)
plt.show()

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Line plot
axs[0, 0].plot(x.numpy(), y.numpy(), 'r-')
axs[0, 0].set_title('Line Plot')
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')
axs[0, 0].grid(True)

# Scatter plot
axs[0, 1].scatter(x_scatter.numpy(), y_scatter.numpy(), c='blue', alpha=0.7)
axs[0, 1].set_title('Scatter Plot')
axs[0, 1].set_xlabel('X')
axs[0, 1].set_ylabel('Y')
axs[0, 1].grid(True)

# Histogram
axs[1, 0].hist(data.numpy(), bins=30, color='green', alpha=0.7)
axs[1, 0].set_title('Histogram')
axs[1, 0].set_xlabel('Value')
axs[1, 0].set_ylabel('Frequency')
axs[1, 0].grid(True)

# Bar chart
axs[1, 1].bar(categories, values.numpy(), color='purple', alpha=0.7)
axs[1, 1].set_title('Bar Chart')
axs[1, 1].set_xlabel('Categories')
axs[1, 1].set_ylabel('Values')
axs[1, 1].grid(True, axis='y')

plt.tight_layout()
plt.show()
```

**When and where we use basic plotting:**
- When exploring the distribution of individual variables
- When examining relationships between two variables
- When comparing values across different categories
- When identifying outliers and anomalies in data
- When communicating basic findings to stakeholders

**Real-world examples:**
- Plotting the distribution of customer ages in a dataset
- Visualizing the relationship between advertising spend and sales
- Comparing performance metrics across different product categories
- Examining the distribution of model prediction errors

### 2. Statistical Visualization with Seaborn

**What is Seaborn?**
Seaborn is a Python data visualization library based on Matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics. It integrates closely with Pandas data structures and is designed to work well with the entire data science stack.

**PyTorch Integration:**
```python
import seaborn as sns
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# Create a sample dataset using PyTorch and convert to Pandas DataFrame
num_samples = 200
data = {
    'age': torch.randint(18, 70, (num_samples,)),
    'income': torch.randn(num_samples) * 20000 + 50000,
    'education_years': torch.randint(8, 20, (num_samples,)),
    'satisfaction': torch.rand(num_samples) * 4 + 1,  # Scale 1-5
    'gender': torch.randint(0, 2, (num_samples,)).apply_(lambda x: 'Male' if x == 0 else 'Female'),
    'department': torch.randint(0, 4, (num_samples,)).apply_(lambda x: ['IT', 'HR', 'Finance', 'Marketing'][x])
}

# Convert tensors to numpy arrays for DataFrame creation
for key in data:
    if isinstance(data[key], torch.Tensor):
        data[key] = data[key].numpy()

df = pd.DataFrame(data)

# Create a distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(df['income'], kde=True)
plt.title('Distribution of Income')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.show()

# Create a box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='department', y='income', data=df)
plt.title('Income by Department')
plt.xlabel('Department')
plt.ylabel('Income')
plt.show()

# Create a violin plot (combines box plot with kernel density estimation)
plt.figure(figsize=(10, 6))
sns.violinplot(x='gender', y='satisfaction', data=df)
plt.title('Satisfaction by Gender')
plt.xlabel('Gender')
plt.ylabel('Satisfaction')
plt.show()

# Create a scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(x='age', y='income', data=df, scatter_kws={'alpha': 0.6})
plt.title('Income vs. Age')
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()

# Create a pair plot for multiple variables
plt.figure(figsize=(12, 10))
sns.pairplot(df[['age', 'income', 'education_years', 'satisfaction']])
plt.suptitle('Pair Plot of Numerical Variables', y=1.02)
plt.show()

# Create a heatmap for correlation
plt.figure(figsize=(10, 8))
corr = df[['age', 'income', 'education_years', 'satisfaction']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Create a joint plot (combines scatter plots, histograms, and density estimates)
plt.figure(figsize=(10, 8))
sns.jointplot(x='age', y='income', data=df, kind='reg')
plt.suptitle('Joint Plot of Age and Income', y=1.02)
plt.show()

# Create a count plot for categorical variables
plt.figure(figsize=(10, 6))
sns.countplot(x='department', data=df, hue='gender')
plt.title('Count of Employees by Department and Gender')
plt.xlabel('Department')
plt.ylabel('Count')
plt.show()

# Create a catplot (categorical plot) for complex categorical relationships
g = sns.catplot(x='department', y='income', hue='gender', col='gender', data=df, kind='box', height=5, aspect=0.7)
g.fig.suptitle('Income by Department and Gender', y=1.02)
plt.show()

# Create a lmplot (linear model plot) with facets
g = sns.lmplot(x='education_years', y='income', hue='gender', col='department', data=df, 
               height=4, aspect=0.7, scatter_kws={'alpha': 0.6})
g.fig.suptitle('Income vs. Education Years by Department and Gender', y=1.02)
plt.show()

# Create a distplot (distribution plot) for multiple variables
plt.figure(figsize=(10, 6))
for dept in df['department'].unique():
    sns.kdeplot(df[df['department'] == dept]['income'], label=dept)
plt.title('Income Distribution by Department')
plt.xlabel('Income')
plt.ylabel('Density')
plt.legend()
plt.show()
```

**When and where we use statistical visualization:**
- When exploring the distribution of variables and their relationships
- When comparing distributions across different groups
- When identifying correlations between multiple variables
- When examining complex relationships involving multiple variables
- When validating assumptions about data distributions

**Real-world examples:**
- Analyzing the distribution of customer purchase amounts across different segments
- Examining the relationship between customer satisfaction and various factors
- Comparing performance metrics across different product categories and regions
- Visualizing the correlation between different economic indicators
- Exploring the distribution of model predictions versus actual values

### 3. Interactive Visualization with Plotly

**What is Plotly?**
Plotly is an open-source graphing library that makes interactive, publication-quality graphs online. It provides a high-level interface for drawing attractive and interactive statistical graphics.

**PyTorch Integration:**
```python
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import torch
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Create a sample dataset using PyTorch and convert to Pandas DataFrame
num_samples = 200
data = {
    'date': pd.date_range(start='2020-01-01', periods=num_samples),
    'value': torch.cumsum(torch.randn(num_samples)) + 100,  # Random walk
    'category': torch.randint(0, 3, (num_samples,)).apply_(lambda x: ['A', 'B', 'C'][x]),
    'size': torch.randint(1, 10, (num_samples,)),
    'color': torch.randint(0, 3, (num_samples,)).apply_(lambda x: ['red', 'green', 'blue'][x])
}

# Convert tensors to numpy arrays for DataFrame creation
for key in data:
    if isinstance(data[key], torch.Tensor):
        data[key] = data[key].numpy()

df = pd.DataFrame(data)

# Create an interactive line plot
fig = px.line(df, x='date', y='value', title='Interactive Line Plot')
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Value',
    hovermode='x unified'
)
fig.show()

# Create an interactive scatter plot
fig = px.scatter(df, x='value', y='size', color='category', size='size', 
                  hover_name='category', title='Interactive Scatter Plot')
fig.update_layout(
    xaxis_title='Value',
    yaxis_title='Size'
)
fig.show()

# Create an interactive bar chart
# First, aggregate data by category
agg_df = df.groupby('category').agg({'value': 'mean', 'size': 'mean'}).reset_index()

fig = px.bar(agg_df, x='category', y='value', title='Interactive Bar Chart')
fig.update_layout(
    xaxis_title='Category',
    yaxis_title='Average Value'
)
fig.show()

# Create an interactive histogram
fig = px.histogram(df, x='value', nbins=30, title='Interactive Histogram')
fig.update_layout(
    xaxis_title='Value',
    yaxis_title='Count'
)
fig.show()

# Create an interactive box plot
fig = px.box(df, x='category', y='value', title='Interactive Box Plot')
fig.update_layout(
    xaxis_title='Category',
    yaxis_title='Value'
)
fig.show()

# Create an interactive violin plot
fig = px.violin(df, x='category', y='value', title='Interactive Violin Plot')
fig.update_layout(
    xaxis_title='Category',
    yaxis_title='Value'
)
fig.show()

# Create an interactive heatmap
# First, create a correlation matrix
corr_matrix = df[['value', 'size']].corr()

fig = px.imshow(corr_matrix, 
                text_auto=True, 
                aspect="auto",
                title='Interactive Heatmap')
fig.show()

# Create a 3D scatter plot
fig = px.scatter_3d(df, x='value', y='size', z=df.index, color='category', 
                    title='Interactive 3D Scatter Plot')
fig.update_layout(
    scene=dict(
        xaxis_title='Value',
        yaxis_title='Size',
        zaxis_title='Index'
    )
)
fig.show()

# Create a time series plot with range slider
fig = px.line(df, x='date', y='value', title='Time Series with Range Slider')
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Value',
    xaxis_rangeslider_visible=True
)
fig.show()

# Create a sunburst chart
# First, create hierarchical data
hierarchical_data = {
    'ids': ['A', 'A-1', 'A-2', 'B', 'B-1', 'B-2', 'C', 'C-1', 'C-2'],
    'labels': ['A', 'A-1', 'A-2', 'B', 'B-1', 'B-2', 'C', 'C-1', 'C-2'],
    'parents': ['', 'A', 'A', '', 'B', 'B', '', 'C', 'C'],
    'values': [10, 5, 5, 15, 8, 7, 20, 12, 8]
}

fig = px.sunburst(
    hierarchical_data,
    names='labels',
    parents='parents',
    values='values',
    title='Interactive Sunburst Chart'
)
fig.show()

# Create a treemap
fig = px.treemap(
    hierarchical_data,
    names='labels',
    parents='parents',
    values='values',
    title='Interactive Treemap'
)
fig.show()

# Create a funnel chart
funnel_data = {
    'stage': ['Visit', 'Sign-up', 'Selection', 'Purchase'],
    'number': [1000, 800, 400, 200]
}

fig = px.funnel(funnel_data, x='number', y='stage', title='Interactive Funnel Chart')
fig.show()

# Create a radar chart
categories = ['Speed', 'Reliability', 'Comfort', 'Safety', 'Efficiency']
fig = go.Figure()

# Add traces for two different products
fig.add_trace(go.Scatterpolar(
    r=[5, 4, 3, 4, 5],
    theta=categories,
    fill='toself',
    name='Product A'
))

fig.add_trace(go.Scatterpolar(
    r=[3, 5, 4, 3, 4],
    theta=categories,
    fill='toself',
    name='Product B'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 5]
        )),
    showlegend=True,
    title='Interactive Radar Chart'
)

fig.show()

# Create a contour plot
# Generate some data
x = torch.linspace(-5, 5, 100)
y = torch.linspace(-5, 5, 100)
X, Y = torch.meshgrid(x, y)
Z = torch.sin(torch.sqrt(X**2 + Y**2))

fig = go.Figure(data=go.Contour(
    x=x.numpy(),
    y=y.numpy(),
    z=Z.numpy(),
    colorscale='Viridis'
))

fig.update_layout(
    title='Interactive Contour Plot',
    xaxis_title='X',
    yaxis_title='Y'
)

fig.show()
```

**When and where we use interactive visualization:**
- When creating dashboards and reports for stakeholders to explore
- When exploring complex datasets with many variables
- When presenting findings in a web-based format
- When creating visualizations that allow users to drill down into details
- When visualizing high-dimensional data

**Real-world examples:**
- Creating interactive dashboards for business intelligence
- Building web-based tools for data exploration
- Visualizing complex relationships in financial data
- Creating interactive maps for geographic data analysis
- Developing tools for exploring model predictions and explanations

### 4. Visualization for Model Interpretation

**What is model interpretation visualization?**
Model interpretation visualization involves creating visual representations to help understand how machine learning models make predictions, which features are important, and how changes in input affect outputs. This is especially important for complex deep learning models that are often considered "black boxes."

**PyTorch Integration:**
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import shap

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create a sample dataset for classification
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# Initialize the model
input_size = X_train.shape[1]
hidden_size = 64
num_classes = len(np.unique(y))
model = SimpleNN(input_size, hidden_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'Test Accuracy: {accuracy:.4f}')

# 1. Feature Importance using Permutation Importance
print("\n=== Feature Importance using Permutation Importance ===")
result = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42)

# Get feature names
feature_names = [f'Feature {i}' for i in range(X.shape[1])]

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': result.importances_mean,
    'std': result.importances_std
}).sort_values('importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=importance_df.head(10), xerr=importance_df.head(10)['std'])
plt.title('Top 10 Feature Importances (Permutation Importance)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# 2. SHAP Values for Model Interpretation
print("\n=== SHAP Values for Model Interpretation ===")

# Create a wrapper function for the model
def model_predict(data):
    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.float32)
        outputs = model(data_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        return probabilities.numpy()

# Create a SHAP explainer
explainer = shap.KernelExplainer(model_predict, X_train_scaled[:100])  # Use a subset for efficiency

# Calculate SHAP values for a subset of test data
shap_values = explainer.shap_values(X_test_scaled[:100])

# Summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_scaled[:100], feature_names=feature_names, plot_type="bar")
plt.title('SHAP Feature Importance')
plt.tight_layout()
plt.show()

# Detailed SHAP summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_scaled[:100], feature_names=feature_names)
plt.title('SHAP Summary Plot')
plt.tight_layout()
plt.show()

# Dependence plot for the most important feature
most_important_idx = importance_df.iloc[0]['feature'].split()[1]
plt.figure(figsize=(10, 6))
shap.dependence_plot(int(most_important_idx), shap_values[0], X_test_scaled[:100], feature_names=feature_names)
plt.title(f'SHAP Dependence Plot for {importance_df.iloc[0]["feature"]}')
plt.tight_layout()
plt.show()

# 3. Partial Dependence Plots
print("\n=== Partial Dependence Plots ===")

from sklearn.inspection import PartialDependenceDisplay

# Create a function to get model predictions for sklearn
def sklearn_predict_proba(data):
    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.float32)
        outputs = model(data_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        return probabilities.numpy()

# Plot partial dependence for the top 3 features
top_features = importance_df.head(3)['feature'].tolist()
top_feature_indices = [int(f.split()[1]) for f in top_features]

fig, ax = plt.subplots(figsize=(15, 5))
PartialDependenceDisplay.from_estimator(
    estimator=None,
    X=X_test_scaled,
    features=top_feature_indices,
    kind='average',
    ax=ax,
    custom_func=sklearn_predict_proba,
    feature_names=feature_names,
    target_idx=1  # Probability of class 1
)
plt.suptitle('Partial Dependence Plots for Top 3 Features')
plt.tight_layout()
plt.show()

# 4. Confusion Matrix
print("\n=== Confusion Matrix ===")
from sklearn.metrics import confusion_matrix

# Get predictions
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)

# Calculate confusion matrix
cm = confusion_matrix(y_test, predicted.numpy())

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['Class 0', 'Class 1'],
           yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 5. ROC Curve
print("\n=== ROC Curve ===")
from sklearn.metrics import roc_curve, auc

# Get prediction probabilities
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    probabilities = torch.softmax(outputs, dim=1)[:, 1].numpy()

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# 6. Learning Curves
print("\n=== Learning Curves ===")
from sklearn.model_selection import learning_curve

# Create a pipeline for sklearn compatibility
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size, hidden_size, num_classes, epochs=100, lr=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.epochs = epochs
        self.lr = lr
        self.model = None
        
    def fit(self, X, y):
        self.model = SimpleNN(self.input_size, self.hidden_size, self.num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        for epoch in range(self.epochs):
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return self
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.numpy()
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.numpy()

# Create the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', PyTorchClassifier(input_size, hidden_size, num_classes, epochs=50))
])

# Calculate learning curves
train_sizes, train_scores, test_scores = learning_curve(
    pipeline, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
)

# Calculate mean and standard deviation
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', label='Training Accuracy')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.plot(train_sizes, test_mean, color='green', marker='s', label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='green')
plt.title('Learning Curves')
plt.xlabel('Number of Training Samples')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.show()
```

**When and where we use model interpretation visualization:**
- When trying to understand which features are most important for model predictions
- When explaining model decisions to stakeholders or regulators
- When debugging model performance issues
- When trying to improve model performance by focusing on important features
- When ensuring model fairness and identifying potential biases

**Real-world examples:**
- Explaining why a loan application was rejected by a credit scoring model
- Identifying which factors contribute most to customer churn predictions
- Understanding which image features a computer vision model uses for classification
- Explaining which words or phrases are most important in a text classification model
- Visualizing how changes in input features affect model predictions

### 5. Advanced Visualization Techniques

**What are advanced visualization techniques?**
Advanced visualization techniques include methods for visualizing high-dimensional data, complex relationships, and specialized data types. These techniques are particularly useful for deep learning applications where data often has high dimensionality.

**PyTorch Integration:**
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import umap

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate high-dimensional data
n_samples = 1000
n_features = 50
n_classes = 5

X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=20, 
                            n_redundant=15, n_classes=n_classes, random_state=42)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. t-SNE for dimensionality reduction and visualization
print("\n=== t-SNE Visualization ===")
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Plot t-SNE results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Class')
plt.title('t-SNE Visualization of High-Dimensional Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True, alpha=0.3)
plt.show()

# 3D t-SNE
tsne_3d = TSNE(n_components=3, random_state=42)
X_tsne_3d = tsne_3d.fit_transform(X_scaled)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Class')
ax.set_title('3D t-SNE Visualization')
ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.set_zlabel('t-SNE Component 3')
plt.show()

# 2. PCA for dimensionality reduction and visualization
print("\n=== PCA Visualization ===")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Plot PCA results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Class')
plt.title('PCA Visualization of High-Dimensional Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, alpha=0.3)
plt.show()

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 6))
plt.bar(range(len(explained_variance)), explained_variance, alpha=0.7, align='center', label='Individual explained variance')
plt.step(range(len(cumulative_explained_variance)), cumulative_explained_variance, where='mid', label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.title('Explained Variance by Principal Components')
plt.tight_layout()
plt.show()

# 3D PCA
pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X_scaled)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Class')
ax.set_title('3D PCA Visualization')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.show()

# 3. UMAP for dimensionality reduction and visualization
print("\n=== UMAP Visualization ===")
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# Plot UMAP results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Class')
plt.title('UMAP Visualization of High-Dimensional Data')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.grid(True, alpha=0.3)
plt.show()

# 4. Parallel Coordinates
print("\n=== Parallel Coordinates ===")
# Create a DataFrame with the data and a sample of features
df = pd.DataFrame(X_scaled[:, :10], columns=[f'Feature {i}' for i in range(10)])
df['Class'] = y

# Plot parallel coordinates
plt.figure(figsize=(12, 8))
pd.plotting.parallel_coordinates(df, 'Class', colormap='viridis')
plt.title('Parallel Coordinates Plot')
plt.xlabel('Features')
plt.ylabel('Feature Values')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()

# 5. Andrews Curves
print("\n=== Andrews Curves ===")
plt.figure(figsize=(12, 8))
pd.plotting.andrews_curves(df, 'Class', colormap='viridis')
plt.title('Andrews Curves')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()

# 6. Radar Chart for Feature Comparison
print("\n=== Radar Chart for Feature Comparison ===")
# Calculate mean values for each class
class_means = df.groupby('Class').mean()

# Create a radar chart
categories = list(class_means.columns)
N = len(categories)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # Complete the loop

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

for i, class_idx in enumerate(class_means.index):
    values = class_means.loc[class_idx].values.tolist()
    values += values[:1]  # Complete the loop
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Class {class_idx}')
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_title('Radar Chart of Feature Means by Class')
ax.legend(loc='upper right')
plt.grid(True)
plt.show()

# 7. Self-Organizing Maps (SOM) for visualization
print("\n=== Self-Organizing Map (SOM) Visualization ===")
# Install minisom if not available
try:
    from minisom import MiniSom
    
    # Train a SOM
    som = MiniSom(10, 10, X_scaled.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
    som.random_weights_init(X_scaled)
    som.train_random(X_scaled, 100)  # Train with 100 iterations
    
    # Plot the SOM
    plt.figure(figsize=(10, 8))
    for i, x in enumerate(X_scaled):
        winner = som.winner(x)
        plt.plot(winner[0] + 0.5, winner[1] + 0.5, 'o', color=plt.cm.viridis(y[i] / n_classes), markersize=10)
    
    plt.title('Self-Organizing Map (SOM)')
    plt.grid(True)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.show()
    
    # Plot the distance map
    plt.figure(figsize=(10, 8))
    plt.pcolor(som.distance_map().T, cmap='viridis')
    plt.colorbar()
    plt.title('SOM Distance Map')
    plt.show()
    
except ImportError:
    print("MiniSom not installed. Skipping SOM visualization.")

# 8. Neural Network Activation Visualization
print("\n=== Neural Network Activation Visualization ===")

# Define a simple neural network for visualization
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(SimpleNN, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
    
    def get_activations(self, x):
        activations = []
        for layer in self.model:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                activations.append(x.detach().numpy())
        return activations

# Initialize the model
input_size = X_scaled.shape[1]
hidden_sizes = [64, 32, 16]
num_classes = n_classes
model = SimpleNN(input_size, hidden_sizes, num_classes)

# Get activations for a sample
sample_idx = 0
sample = torch.tensor(X_scaled[sample_idx:sample_idx+1], dtype=torch.float32)
activations = model.get_activations(sample)

# Plot activations for each layer
plt.figure(figsize=(15, 10))
for i, activation in enumerate(activations):
    plt.subplot(len(activations), 1, i+1)
    plt.bar(range(len(activation[0])), activation[0])
    plt.title(f'Layer {i+1} Activations')
    plt.xlabel('Neuron')
    plt.ylabel('Activation')
plt.tight_layout()
plt.show()

# 9. Heatmap of Neural Network Weights
print("\n=== Neural Network Weight Heatmap ===")

# Extract weights from the model
weights = []
for layer in model.model:
    if isinstance(layer, nn.Linear):
        weights.append(layer.weight.detach().numpy())

# Plot weight heatmaps
plt.figure(figsize=(15, 10))
for i, weight in enumerate(weights):
    plt.subplot(len(weights), 1, i+1)
    sns.heatmap(weight, cmap='viridis', center=0)
    plt.title(f'Layer {i+1} Weights')
    plt.xlabel('Output Neuron')
    plt.ylabel('Input Neuron')
plt.tight_layout()
plt.show()

# 10. t-SNE of Neural Network Activations
print("\n=== t-SNE of Neural Network Activations ===")

# Get activations for all samples
all_activations = []
with torch.no_grad():
    for x in torch.tensor(X_scaled, dtype=torch.float32):
        activations = model.get_activations(x.unsqueeze(0))
        all_activations.append([act[0] for act in activations])

# Extract activations from a specific layer (e.g., the second hidden layer)
layer_idx = 1  # Second hidden layer
layer_activations = np.array([act[layer_idx] for act in all_activations])

# Apply t-SNE to the activations
tsne_activations = TSNE(n_components=2, random_state=42)
activations_tsne = tsne_activations.fit_transform(layer_activations)

# Plot t-SNE of activations
plt.figure(figsize=(10, 8))
scatter = plt.scatter(activations_tsne[:, 0], activations_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Class')
plt.title(f't-SNE of Layer {layer_idx+1} Activations')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True, alpha=0.3)
plt.show()
```

**When and where we use advanced visualization techniques:**
- When visualizing high-dimensional data from deep learning models
- When exploring complex relationships in large datasets
- When trying to understand the internal workings of neural networks
- When comparing different models or model configurations
- When identifying patterns or clusters in complex data

**Real-world examples:**
- Visualizing word embeddings in natural language processing models
- Exploring the latent space of generative models like GANs or VAEs
- Understanding how different layers of a neural network transform data
- Visualizing high-dimensional features in computer vision models
- Comparing representations learned by different models

## Applications in Deep Learning

### 1. Visualizing Training Progress

Visualizing training progress is essential for monitoring model performance and diagnosing issues:

```python
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create a sample dataset
X, y = make_classification(n_samples=2000, n_features=20, n_informative=10, n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define a neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize the model
input_size = X_train.shape[1]
hidden_size = 128
num_classes = len(np.unique(y))
model = NeuralNetwork(input_size, hidden_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Lists to store training history
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# Training loop
num_epochs = 50

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # Evaluation phase
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_accuracy = correct / total
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    # Print progress
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}')

# Plot training history
plt.figure(figsize=(15, 6))

# Plot training & validation loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot training & validation accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot confusion matrix
from sklearn.metrics import confusion_matrix

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=[f'Class {i}' for i in range(num_classes)],
           yticklabels=[f'Class {i}' for i in range(num_classes)])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot ROC curve for each class
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

# Binarize the labels
y_test_bin = label_binarize(y_test, classes=range(num_classes))
n_classes = y_test_bin.shape[1]

# Get prediction probabilities
model.eval()
y_score = []

with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        y_score.append(probabilities.numpy())

y_score = np.vstack(y_score)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at these points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot ROC curves
plt.figure(figsize=(10, 8))
plt.plot(fpr["micro"], tpr["micro"],
         label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label=f'macro-average ROC curve (area = {roc_auc["macro"]:.2f})',
         color='navy', linestyle=':', linewidth=4)

colors = ['aqua', 'darkorange', 'cornflowerblue']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.show()

# Visualize model weights
plt.figure(figsize=(12, 8))

# Get weights from the first layer
weights = model.fc1.weight.data.numpy()

# Plot heatmap of weights
plt.subplot(1, 2, 1)
sns.heatmap(weights, cmap='viridis', center=0)
plt.title('Weights of First Layer')
plt.xlabel('Input Features')
plt.ylabel('Hidden Neurons')

# Plot distribution of weights
plt.subplot(1, 2, 2)
plt.hist(weights.flatten(), bins=50, alpha=0.7, edgecolor='black')
plt.title('Distribution of Weights in First Layer')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Visualize activation patterns
def get_activation(model, x, layer_idx):
    """Get activation values from a specific layer"""
    activation = None
    for i, layer in enumerate(model.model):
        x = layer(x)
        if i == layer_idx:
            activation = x
            break
    return activation

# Get activations from the first hidden layer for a sample
sample_idx = 0
sample = X_test_tensor[sample_idx:sample_idx+1]
activation = get_activation(model, sample, 1)  # First hidden layer

plt.figure(figsize=(10, 6))
plt.bar(range(len(activation[0])), activation[0].detach().numpy())
plt.title(f'Activation Pattern for Sample {sample_idx} in First Hidden Layer')
plt.xlabel('Neuron Index')
plt.ylabel('Activation Value')
plt.grid(True, alpha=0.3)
plt.show()
```

### 2. Visualizing Convolutional Neural Networks

Visualizing CNNs helps understand how they process images and what features they learn:

```python
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import seaborn as sns

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_features(self, x):
        """Extract features from the fully connected layer"""
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return x

# Initialize the model
model = SimpleCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Evaluate the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

# 1. Visualize Convolutional Filters
print("\n=== Visualizing Convolutional Filters ===")

# Get filters from the first convolutional layer
filters = model.conv1.weight.data.numpy()

# Plot filters
plt.figure(figsize=(12, 8))
for i in range(filters.shape[0]):
    plt.subplot(4, 8, i+1)
    plt.imshow(filters[i, 0], cmap='viridis')
    plt.axis('off')
plt.suptitle('Filters from First Convolutional Layer')
plt.tight_layout()
plt.show()

# 2. Visualize Feature Maps
print("\n=== Visualizing Feature Maps ===")

# Define a hook to get feature maps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# Register hooks
model.conv1.register_forward_hook(get_activation('conv1'))
model.conv2.register_forward_hook(get_activation('conv2'))

# Get a sample image
sample_image, sample_label = test_dataset[0]
sample_image = sample_image.unsqueeze(0)  # Add batch dimension

# Forward pass
output = model(sample_image)

# Visualize feature maps from the first convolutional layer
act = activation['conv1'].squeeze().numpy()
fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < act.shape[0]:
        ax.imshow(act[i], cmap='viridis')
    ax.axis('off')
plt.suptitle('Feature Maps from First Convolutional Layer')
plt.tight_layout()
plt.show()

# Visualize feature maps from the second convolutional layer
act = activation['conv2'].squeeze().numpy()
fig, axes = plt.subplots(8, 8, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    if i < act.shape[0]:
        ax.imshow(act[i], cmap='viridis')
    ax.axis('off')
plt.suptitle('Feature Maps from Second Convolutional Layer')
plt.tight_layout()
plt.show()

# 3. Visualize Activation Maximization
print("\n=== Visualizing Activation Maximization ===")

def visualize_activation_maximization(model, target_class, layer_idx, neuron_idx, epochs=100, lr=0.1):
    """Generate an image that maximizes activation of a specific neuron"""
    # Start with random noise
    input_img = torch.randn(1, 1, 28, 28, requires_grad=True)
    
    # Define optimizer
    optimizer = optim.Adam([input_img], lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        x = input_img
        for i, layer in enumerate(model.model):
            x = layer(x)
            if i == layer_idx:
                # Maximize activation of the target neuron
                loss = -x[0, neuron_idx]
                break
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Normalize to keep values in reasonable range
        with torch.no_grad():
            input_img.clamp_(0, 1)
    
    return input_img.detach().squeeze().numpy()

# Generate images that maximize activation in the first convolutional layer
plt.figure(figsize=(12, 8))
for i in range(16):
    img = visualize_activation_maximization(model, None, 0, i)
    plt.subplot(4, 4, i+1)
    plt.imshow(img, cmap='viridis')
    plt.axis('off')
plt.suptitle('Images that Maximize Activation in First Convolutional Layer')
plt.tight_layout()
plt.show()

# 4. Visualize t-SNE of Features
print("\n=== Visualizing t-SNE of Features ===")

# Extract features from the fully connected layer
model.eval()
features = []
labels = []

with torch.no_grad():
    for inputs, targets in test_loader:
        feats = model.get_features(inputs)
        features.append(feats.numpy())
        labels.append(targets.numpy())

features = np.vstack(features)
labels = np.concatenate(labels)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
features_tsne = tsne.fit_transform(features)

# Plot t-SNE
plt.figure(figsize=(10, 8))
scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, label='Digit')
plt.title('t-SNE of CNN Features')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True, alpha=0.3)
plt.show()

# 5. Visualize Occlusion Sensitivity
print("\n=== Visualizing Occlusion Sensitivity ===")

def occlusion_sensitivity(model, image, label, patch_size=4):
    """Compute occlusion sensitivity map"""
    model.eval()
    
    # Get original prediction
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        original_prob = F.softmax(output, dim=1)[0, label].item()
    
    # Create occlusion map
    occlusion_map = torch.zeros(1, 28, 28)
    
    # Slide a patch across the image
    for i in range(0, 28, patch_size):
        for j in range(0, 28, patch_size):
            # Create occluded image
            occluded_image = image.clone()
            occluded_image[i:i+patch_size, j:j+patch_size] = 0
            
            # Get prediction with occluded image
            with torch.no_grad():
                output = model(occluded_image.unsqueeze(0))
                occluded_prob = F.softmax(output, dim=1)[0, label].item()
            
            # Update occlusion map
            occlusion_map[0, i:i+patch_size, j:j+patch_size] = original_prob - occluded_prob
    
    return occlusion_map.squeeze().numpy()

# Get a sample image
sample_image, sample_label = test_dataset[0]

# Compute occlusion sensitivity map
occlusion_map = occlusion_sensitivity(model, sample_image, sample_label)

# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(sample_image.squeeze(), cmap='gray')
plt.title(f'Original Image (Digit: {sample_label})')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(occlusion_map, cmap='hot')
plt.title('Occlusion Sensitivity Map')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sample_image.squeeze(), cmap='gray')
plt.imshow(occlusion_map, cmap='hot', alpha=0.5)
plt.title('Overlay')
plt.axis('off')

plt.tight_layout()
plt.show()

# 6. Visualize Saliency Maps
print("\n=== Visualizing Saliency Maps ===")

def compute_saliency_map(model, image, target_class):
    """Compute saliency map using gradients"""
    model.eval()
    
    # Set requires_grad to True for the image
    image = image.clone().detach().requires_grad_(True)
    
    # Forward pass
    output = model(image.unsqueeze(0))
    
    # Zero gradients
    model.zero_grad()
    
    # Backward pass
    class_score = output[0, target_class]
    class_score.backward()
    
    # Get gradients
    saliency_map = image.grad.abs().squeeze().numpy()
    
    return saliency_map

# Get a sample image
sample_image, sample_label = test_dataset[0]

# Compute saliency map
saliency_map = compute_saliency_map(model, sample_image, sample_label)

# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(sample_image.squeeze(), cmap='gray')
plt.title(f'Original Image (Digit: {sample_label})')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(saliency_map, cmap='hot')
plt.title('Saliency Map')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sample_image.squeeze(), cmap='gray')
plt.imshow(saliency_map, cmap='hot', alpha=0.5)
plt.title('Overlay')
plt.axis('off')

plt.tight_layout()
plt.show()

# 7. Visualize Class Activation Maps (CAM)
print("\n=== Visualizing Class Activation Maps (CAM) ===")

class SimpleCNNWithCAM(nn.Module):
    def __init__(self):
        super(SimpleCNNWithCAM, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cam_fc = nn.Linear(64, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # For CAM
        cam_features = x
        cam_x = self.gap(cam_features)
        cam_x = cam_x.view(cam_x.size(0), -1)
        cam_output = self.cam_fc(cam_x)
        
        # For regular classification
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x, cam_output, cam_features

# Initialize the model
model_cam = SimpleCNNWithCAM()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cam.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model_cam.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        # Forward pass
        outputs, cam_outputs, _ = model_cam(inputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Evaluate the model
model_cam.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs, _, _ = model_cam(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Generate CAM for a sample image
sample_image, sample_label = test_dataset[0]

# Get CAM
model_cam.eval()
with torch.no_grad():
    _, cam_output, cam_features = model_cam(sample_image.unsqueeze(0))

# Get weights of the final layer for the target class
weights = model_cam.cam_fc.weight[sample_label].detach().numpy()

# Compute CAM
cam = np.zeros(cam_features.shape[2:])
for i, w in enumerate(weights):
    cam += w * cam_features[0, i].detach().numpy()

# Normalize CAM
cam = (cam - cam.min()) / (cam.max() - cam.min())

# Resize CAM to match image size
from skimage.transform import resize
cam_resized = resize(cam, (28, 28))

# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(sample_image.squeeze(), cmap='gray')
plt.title(f'Original Image (Digit: {sample_label})')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cam_resized, cmap='jet')
plt.title('Class Activation Map')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sample_image.squeeze(), cmap='gray')
plt.imshow(cam_resized, cmap='jet', alpha=0.5)
plt.title('Overlay')
plt.axis('off')

plt.tight_layout()
plt.show()
```

### 3. Visualizing Recurrent Neural Networks

Visualizing RNNs helps understand how they process sequential data:

```python
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
import seaborn as sns

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic sequential data
def generate_sine_wave(seq_length, num_samples):
    """Generate sine wave data"""
    x = np.linspace(0, 4*np.pi, seq_length)
    data = np.zeros((num_samples, seq_length, 1))
    for i in range(num_samples):
        # Random phase shift
        phase = np.random.uniform(0, 2*np.pi)
        # Random amplitude
        amplitude = np.random.uniform(0.5, 2.0)
        # Generate sine wave
        data[i, :, 0] = amplitude * np.sin(x + phase)
    return data

# Generate data
seq_length = 50
num_samples = 1000
data = generate_sine_wave(seq_length, num_samples)

# Split into train and test sets
train_size = int(0.8 * num_samples)
train_data = data[:train_size]
test_data = data[train_size:]

# Convert to PyTorch tensors
train_tensor = torch.tensor(train_data, dtype=torch.float32)
test_tensor = torch.tensor(test_data, dtype=torch.float32)

# Define a simple RNN
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        # Forward pass
        out, _ = self.rnn(x, h0)
        
        # Get output from the last time step
        out = self.fc(out[:, -1, :])
        return out

# Define an LSTM
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        # Forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Get output from the last time step
        out = self.fc(out[:, -1, :])
        return out
    
    def get_hidden_states(self, x):
        """Get hidden states from all time steps"""
        # Initialize hidden and cell states
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        # Forward pass
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        return out, (hn, cn)

# Define a GRU
class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        # Forward pass
        out, _ = self.gru(x, h0)
        
        # Get output from the last time step
        out = self.fc(out[:, -1, :])
        return out
    
    def get_hidden_states(self, x):
        """Get hidden states from all time steps"""
        # Initialize hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        # Forward pass
        out, hn = self.gru(x, h0)
        
        return out, hn

# Initialize models
input_size = 1
hidden_size = 32
output_size = 1  # Predicting the next value in the sequence

rnn_model = SimpleRNN(input_size, hidden_size, output_size)
lstm_model = SimpleLSTM(input_size, hidden_size, output_size)
gru_model = SimpleGRU(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.01)
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.01)
gru_optimizer = optim.Adam(gru_model.parameters(), lr=0.01)

# Prepare data for training (predict next value)
def prepare_data(data):
    """Prepare data for sequence prediction"""
    X = data[:, :-1, :]  # All but the last value
    y = data[:, 1:, :]   # All but the first value
    return X, y

X_train, y_train = prepare_data(train_tensor)
X_test, y_test = prepare_data(test_tensor)

# Training function
def train_model(model, X_train, y_train, optimizer, criterion, num_epochs=100):
    """Train the model"""
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train[:, -1, :])
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')
    
    return losses

# Train the models
print("Training RNN model...")
rnn_losses = train_model(rnn_model, X_train, y_train, rnn_optimizer, criterion)

print("\nTraining LSTM model...")
lstm_losses = train_model(lstm_model, X_train, y_train, lstm_optimizer, criterion)

print("\nTraining GRU model...")
gru_losses = train_model(gru_model, X_train, y_train, gru_optimizer, criterion)

# Plot training losses
plt.figure(figsize=(12, 6))
plt.plot(rnn_losses, label='RNN')
plt.plot(lstm_losses, label='LSTM')
plt.plot(gru_losses, label='GRU')
plt.title('Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the models
def evaluate_model(model, X_test, y_test):
    """Evaluate the model"""
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        loss = criterion(outputs, y_test[:, -1, :])
    return loss.item()

rnn_test_loss = evaluate_model(rnn_model, X_test, y_test)
lstm_test_loss = evaluate_model(lstm_model, X_test, y_test)
gru_test_loss = evaluate_model(gru_model, X_test, y_test)

print(f"\nTest Loss - RNN: {rnn_test_loss:.6f}")
print(f"Test Loss - LSTM: {lstm_test_loss:.6f}")
print(f"Test Loss - GRU: {gru_test_loss:.6f}")

# Make predictions
def make_predictions(model, X_test):
    """Make predictions using the model"""
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    return predictions.numpy()

rnn_predictions = make_predictions(rnn_model, X_test)
lstm_predictions = make_predictions(lstm_model, X_test)
gru_predictions = make_predictions(gru_model, X_test)

# Plot predictions
sample_idx = 0
plt.figure(figsize=(12, 8))

# Plot actual sequence
plt.subplot(2, 2, 1)
plt.plot(X_test[sample_idx, :, 0].numpy(), label='Input')
plt.plot(y_test[sample_idx, :, 0].numpy(), label='Target')
plt.title('Actual Sequence')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Plot RNN predictions
plt.subplot(2, 2, 2)
plt.plot(X_test[sample_idx, :, 0].numpy(), label='Input')
plt.plot(y_test[sample_idx, :, 0].numpy(), label='Target')
plt.plot(np.arange(1, seq_length), rnn_predictions[sample_idx, :], label='RNN Prediction')
plt.title('RNN Prediction')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Plot LSTM predictions
plt.subplot(2, 2, 3)
plt.plot(X_test[sample_idx, :, 0].numpy(), label='Input')
plt.plot(y_test[sample_idx, :, 0].numpy(), label='Target')
plt.plot(np.arange(1, seq_length), lstm_predictions[sample_idx, :], label='LSTM Prediction')
plt.title('LSTM Prediction')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Plot GRU predictions
plt.subplot(2, 2, 4)
plt.plot(X_test[sample_idx, :, 0].numpy(), label='Input')
plt.plot(y_test[sample_idx, :, 0].numpy(), label='Target')
plt.plot(np.arange(1, seq_length), gru_predictions[sample_idx, :], label='GRU Prediction')
plt.title('GRU Prediction')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Visualize hidden states
print("\n=== Visualizing Hidden States ===")

# Get hidden states from LSTM
lstm_model.eval()
with torch.no_grad():
    hidden_states, _ = lstm_model.get_hidden_states(X_test[:10])  # Use first 10 samples

# Reshape for t-SNE
hidden_states_flat = hidden_states.reshape(-1, hidden_size).numpy()

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
hidden_states_tsne = tsne.fit_transform(hidden_states_flat)

# Create a color map for time steps
colors = plt.cm.viridis(np.linspace(0, 1, seq_length-1))

# Plot t-SNE of hidden states
plt.figure(figsize=(12, 8))
for i in range(seq_length-1):
    indices = np.arange(i, len(hidden_states_flat), seq_length-1)
    plt.scatter(hidden_states_tsne[indices, 0], hidden_states_tsne[indices, 1], 
                c=[colors[i]], label=f'Time Step {i+1}', alpha=0.7)

plt.title('t-SNE of LSTM Hidden States')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Visualize attention weights (simplified version)
print("\n=== Visualizing Attention Weights ===")

# Define a simple attention mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, seq_len, hidden_size)
        attention_weights = self.attention(hidden_states)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)  # (batch_size, seq_len, 1)
        
        # Apply attention weights
        context = torch.sum(hidden_states * attention_weights, dim=1)  # (batch_size, hidden_size)
        
        return context, attention_weights

# Define an LSTM with attention
class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Apply attention
        context, attention_weights = self.attention(out)
        
        # Final prediction
        output = self.fc(context)
        
        return output, attention_weights

# Initialize the model
lstm_attention_model = LSTMWithAttention(input_size, hidden_size, output_size)
lstm_attention_optimizer = optim.Adam(lstm_attention_model.parameters(), lr=0.01)

# Train the model
print("\nTraining LSTM with attention model...")
lstm_attention_losses = train_model(lstm_attention_model, X_train, y_train[:, -1, :], lstm_attention_optimizer, criterion)

# Evaluate the model
lstm_attention_test_loss = evaluate_model(lstm_attention_model, X_test, y_test[:, -1, :])
print(f"Test Loss - LSTM with Attention: {lstm_attention_test_loss:.6f}")

# Get attention weights
lstm_attention_model.eval()
with torch.no_grad():
    _, attention_weights = lstm_attention_model(X_test[:1])  # Use first sample

# Plot attention weights
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(X_test[0, :, 0].numpy(), label='Input')
plt.title('Input Sequence')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.bar(range(len(attention_weights[0, :, 0])), attention_weights[0, :, 0].numpy())
plt.title('Attention Weights')
plt.xlabel('Time Step')
plt.ylabel('Attention Weight')
plt.grid(True)

plt.tight_layout()
plt.show()

# Visualize gradient flow
print("\n=== Visualizing Gradient Flow ===")

# Define a function to extract gradients
def extract_gradients(model, X, y):
    """Extract gradients from the model"""
    model.train()
    
    # Set requires_grad to True for all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Extract gradients
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.abs().mean().item()
    
    return gradients

# Extract gradients for each model
rnn_gradients = extract_gradients(rnn_model, X_train[:1], y_train[:1, -1, :])
lstm_gradients = extract_gradients(lstm_model, X_train[:1], y_train[:1, -1, :])
gru_gradients = extract_gradients(gru_model, X_train[:1], y_train[:1, -1, :])
lstm_attention_gradients = extract_gradients(lstm_attention_model, X_train[:1], y_train[:1, -1, :])

# Plot gradients
plt.figure(figsize=(12, 6))
models = ['RNN', 'LSTM', 'GRU', 'LSTM with Attention']
gradient_data = [rnn_gradients, lstm_gradients, gru_gradients, lstm_attention_gradients]

for i, (model_name, gradients) in enumerate(zip(models, gradient_data)):
    plt.subplot(2, 2, i+1)
    names = list(gradients.keys())
    values = list(gradients.values())
    plt.bar(names, values)
    plt.title(f'{model_name} Gradients')
    plt.xticks(rotation=45)
    plt.ylabel('Mean Absolute Gradient')
    plt.grid(True, axis='y')

plt.tight_layout()
plt.show()
```

## Common Problems and Advancements

### 1. Problem: Visualizing High-Dimensional Data

**Issue:**
Deep learning often deals with high-dimensional data that is difficult to visualize directly, making it challenging to understand patterns and relationships.

**How advancements solved this:**
- **Dimensionality reduction techniques**: Methods like t-SNE, UMAP, and PCA help visualize high-dimensional data in 2D or 3D
- **Interactive visualizations**: Tools that allow users to explore high-dimensional data through interactive interfaces
- **Feature importance visualization**: Methods to highlight the most important features or dimensions
- **Projection techniques**: Methods to project high-dimensional data onto lower dimensions while preserving important properties

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate high-dimensional data
n_samples = 1000
n_features = 50
n_classes = 5

X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=20, 
                            n_redundant=15, n_classes=n_classes, random_state=42)

# Scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 2. t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 3. UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# 4. MDS
mds = MDS(n_components=2, random_state=42)
X_mds = mds.fit_transform(X_scaled)

# 5. Isomap
isomap = Isomap(n_components=2)
X_isomap = isomap.fit_transform(X_scaled)

# 6. Kernel PCA
kpca = KernelPCA(n_components=2, kernel='rbf', random_state=42)
X_kpca = kpca.fit_transform(X_scaled)

# Plot all dimensionality reduction techniques
plt.figure(figsize=(15, 12))

plt.subplot(2, 3, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title('PCA')
plt.xlabel('Component 1')
plt.ylabel('Component 2')

plt.subplot(2, 3, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title('t-SNE')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

plt.subplot(2, 3, 3)
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title('UMAP')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')

plt.subplot(2, 3, 4)
plt.scatter(X_mds[:, 0], X_mds[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title('MDS')
plt.xlabel('MDS 1')
plt.ylabel('MDS 2')

plt.subplot(2, 3, 5)
plt.scatter(X_isomap[:, 0], X_isomap[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title('Isomap')
plt.xlabel('Isomap 1')
plt.ylabel('Isomap 2')

plt.subplot(2, 3, 6)
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title('Kernel PCA')
plt.xlabel('KPCA 1')
plt.ylabel('KPCA 2')

plt.tight_layout()
plt.show()

# Plot explained variance for PCA
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 6))
plt.bar(range(len(explained_variance)), explained_variance, alpha=0.7, align='center', label='Individual explained variance')
plt.step(range(len(cumulative_explained_variance)), cumulative_explained_variance, where='mid', label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.title('Explained Variance by Principal Components')
plt.tight_layout()
plt.show()

# 3D visualization
from mpl_toolkits.mplot3d import Axes3D

# 3D PCA
pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X_scaled)

# 3D t-SNE
tsne_3d = TSNE(n_components=3, random_state=42)
X_tsne_3d = tsne_3d.fit_transform(X_scaled)

# 3D UMAP
reducer_3d = umap.UMAP(n_components=3, random_state=42)
X_umap_3d = reducer_3d.fit_transform(X_scaled)

# Plot 3D visualizations
fig = plt.figure(figsize=(18, 6))

ax = fig.add_subplot(131, projection='3d')
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y, cmap='viridis', alpha=0.7)
ax.set_title('3D PCA')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')

ax = fig.add_subplot(132, projection='3d')
scatter = ax.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2], c=y, cmap='viridis', alpha=0.7)
ax.set_title('3D t-SNE')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_zlabel('t-SNE 3')

ax = fig.add_subplot(133, projection='3d')
scatter = ax.scatter(X_umap_3d[:, 0], X_umap_3d[:, 1], X_umap_3d[:, 2], c=y, cmap='viridis', alpha=0.7)
ax.set_title('3D UMAP')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')

plt.tight_layout()
plt.show()

# Interactive visualization with Plotly
import plotly.express as px

# Create an interactive t-SNE plot
df_tsne = pd.DataFrame({
    'tsne-1': X_tsne[:, 0],
    'tsne-2': X_tsne[:, 1],
    'class': y
})

fig = px.scatter(df_tsne, x='tsne-1', y='tsne-2', color='class', 
                 title='Interactive t-SNE Visualization',
                 hover_data=['class'])
fig.show()

# Create an interactive 3D UMAP plot
df_umap_3d = pd.DataFrame({
    'umap-1': X_umap_3d[:, 0],
    'umap-2': X_umap_3d[:, 1],
    'umap-3': X_umap_3d[:, 2],
    'class': y
})

fig = px.scatter_3d(df_umap_3d, x='umap-1', y='umap-2', z='umap-3', color='class',
                   title='Interactive 3D UMAP Visualization',
                   hover_data=['class'])
fig.show()
```

### 2. Problem: Visualizing Large Datasets

**Issue:**
Visualizing large datasets can be computationally expensive and result in cluttered or unreadable visualizations.

**How advancements solved this:**
- **Data aggregation and binning**: Techniques to summarize large datasets into meaningful aggregates
- **Sampling methods**: Representative sampling to create visualizations of subsets
- **Interactive visualization tools**: Tools that allow users to zoom, pan, and filter data
- **Progressive rendering**: Techniques to render visualizations progressively as data loads
- **Data reduction techniques**: Methods to reduce data size while preserving important patterns

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import datashader as ds
import datashader.transfer_functions as tf
from datashader.mpl_ext import dsshow
import plotly.express as px

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate a large dataset
n_samples = 100000
n_features = 10

X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=5, 
                            n_redundant=3, random_state=42)

# Create a DataFrame
df = pd.DataFrame(X, columns=[f'Feature {i}' for i in range(n_features)])
df['Target'] = y

# Add some categorical features
df['Category'] = np.random.choice(['A', 'B', 'C', 'D'], n_samples)

# 1. Basic visualization with the full dataset (may be slow)
print("Visualizing with basic methods...")
plt.figure(figsize=(10, 8))
plt.scatter(df['Feature 0'], df['Feature 1'], c=df['Target'], alpha=0.1, s=1)
plt.title('Basic Scatter Plot (Full Dataset)')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()

# 2. Sampling
print("Visualizing with sampling...")
sample_size = 10000
df_sample = df.sample(sample_size, random_state=42)

plt.figure(figsize=(10, 8))
plt.scatter(df_sample['Feature 0'], df_sample['Feature 1'], c=df_sample['Target'], alpha=0.5)
plt.title(f'Scatter Plot (Sample Size: {sample_size})')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()

# 3. Hexbin plot
print("Visualizing with hexbin plot...")
plt.figure(figsize=(10, 8))
plt.hexbin(df['Feature 0'], df['Feature 1'], gridsize=50, cmap='viridis')
plt.colorbar(label='Count')
plt.title('Hexbin Plot')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()

# 4. 2D histogram
print("Visualizing with 2D histogram...")
plt.figure(figsize=(10, 8))
plt.hist2d(df['Feature 0'], df['Feature 1'], bins=50, cmap='viridis')
plt.colorbar(label='Count')
plt.title('2D Histogram')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()

# 5. Datashader (for very large datasets)
print("Visualizing with Datashader...")
# Create a canvas
cvs = ds.Canvas(plot_width=800, plot_height=600)

# Aggregate the data
agg = cvs.points(df, 'Feature 0', 'Feature 1')

# Create an image
img = tf.shade(agg, cmap='viridis', how='log')

# Plot the image
plt.figure(figsize=(10, 8))
plt.imshow(img, origin='lower')
plt.title('Datashader Visualization')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.colorbar(label='Count (log scale)')
plt.show()

# 6. Datashader with categorical coloring
print("Visualizing with Datashader and categorical coloring...")
# Create a canvas
cvs = ds.Canvas(plot_width=800, plot_height=600)

# Aggregate the data by category
agg_by_cat = cvs.points(df, 'Feature 0', 'Feature 1', ds.count_cat('Category'))

# Create an image
img = tf.shade(agg_by_cat, color_key=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

# Plot the image
plt.figure(figsize=(10, 8))
plt.imshow(img, origin='lower')
plt.title('Datashader Visualization with Categories')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()

# 7. Box plots with large datasets
print("Visualizing with box plots...")
plt.figure(figsize=(12, 6))
sns.boxplot(x='Category', y='Feature 0', data=df)
plt.title('Box Plot by Category')
plt.xlabel('Category')
plt.ylabel('Feature 0')
plt.show()

# 8. Violin plots with large datasets
print("Visualizing with violin plots...")
plt.figure(figsize=(12, 6))
sns.violinplot(x='Category', y='Feature 0', data=df)
plt.title('Violin Plot by Category')
plt.xlabel('Category')
plt.ylabel('Feature 0')
plt.show()

# 9. Interactive visualization with Plotly
print("Creating interactive visualization with Plotly...")
# Sample the data for interactive visualization
df_interactive = df.sample(10000, random_state=42)

# Create a scatter plot
fig = px.scatter(df_interactive, x='Feature 0', y='Feature 1', color='Target',
                 title='Interactive Scatter Plot',
                 hover_data=['Category'])
fig.show()

# Create a 2D histogram
fig = px.density_heatmap(df_interactive, x='Feature 0', y='Feature 1',
                          title='Interactive 2D Histogram')
fig.show()

# 10. Data aggregation and visualization
print("Visualizing with data aggregation...")
# Aggregate the data
agg_df = df.groupby('Category').agg({
    'Feature 0': ['mean', 'std', 'min', 'max'],
    'Feature 1': ['mean', 'std', 'min', 'max'],
    'Target': 'mean'
}).reset_index()

# Flatten the multi-level columns
agg_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg_df.columns.values]

# Plot the aggregated data
plt.figure(figsize=(12, 6))
plt.bar(agg_df['Category'], agg_df['Feature 0_mean'], yerr=agg_df['Feature 0_std'])
plt.title('Mean Feature 0 by Category')
plt.xlabel('Category')
plt.ylabel('Mean Feature 0')
plt.show()

# 11. Time series aggregation (simulated)
print("Visualizing time series with aggregation...")
# Create a time series dataset
dates = pd.date_range(start='2020-01-01', periods=n_samples)
df_time = df.copy()
df_time['Date'] = dates
df_time['Month'] = df_time['Date'].dt.month
df_time['Day'] = df_time['Date'].dt.day

# Aggregate by month
monthly_agg = df_time.groupby('Month').agg({
    'Feature 0': 'mean',
    'Feature 1': 'mean',
    'Target': 'mean'
}).reset_index()

# Plot the aggregated time series
plt.figure(figsize=(12, 6))
plt.plot(monthly_agg['Month'], monthly_agg['Feature 0'], marker='o')
plt.title('Monthly Average of Feature 0')
plt.xlabel('Month')
plt.ylabel('Average Feature 0')
plt.grid(True)
plt.show()

# 12. Pair plot with sampling
print("Creating pair plot with sampling...")
# Sample the data
df_pair = df.sample(1000, random_state=42)

# Create a pair plot
plt.figure(figsize=(12, 10))
sns.pairplot(df_pair[['Feature 0', 'Feature 1', 'Feature 2', 'Target']], hue='Target')
plt.suptitle('Pair Plot (Sampled Data)', y=1.02)
plt.show()
```

### 3. Problem: Visualizing Model Uncertainty

**Issue:**
Deep learning models often provide point estimates without quantifying uncertainty, making it difficult to assess the reliability of predictions.

**How advancements solved this:**
- **Bayesian neural networks**: Methods to model uncertainty in neural network weights
- **Monte Carlo dropout**: Using dropout at inference time to estimate uncertainty
- **Ensemble methods**: Combining predictions from multiple models to estimate uncertainty
- **Prediction intervals**: Methods to estimate intervals that likely contain the true value
- **Visualization techniques**: Methods to effectively communicate uncertainty to users

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic data
n_samples = 1000
n_features = 10
X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=0.1, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 1. Standard Neural Network
class StandardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StandardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# 2. Bayesian Neural Network with Monte Carlo Dropout
class BayesianNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(BayesianNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# 3. Ensemble of Neural Networks
class EnsembleNN:
    def __init__(self, input_size, hidden_size, output_size, n_models=5):
        self.models = []
        for _ in range(n_models):
            model = StandardNN(input_size, hidden_size, output_size)
            self.models.append(model)
    
    def train(self, X_train, y_train, epochs=100, lr=0.01):
        for model in self.models:
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            for epoch in range(epochs):
                # Forward pass
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    def predict(self, X, n_samples=100):
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = model(X)
                predictions.append(outputs.numpy())
        
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        return mean, std

# Initialize models
input_size = X_train.shape[1]
hidden_size = 64
output_size = 1

standard_model = StandardNN(input_size, hidden_size, output_size)
bayesian_model = BayesianNN(input_size, hidden_size, output_size)
ensemble_model = EnsembleNN(input_size, hidden_size, output_size)

# Train standard model
criterion = nn.MSELoss()
optimizer = optim.Adam(standard_model.parameters(), lr=0.01)

for epoch in range(100):
    # Forward pass
    outputs = standard_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f'Standard Model - Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Train Bayesian model
criterion = nn.MSELoss()
optimizer = optim.Adam(bayesian_model.parameters(), lr=0.01)

for epoch in range(100):
    # Forward pass
    outputs = bayesian_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f'Bayesian Model - Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Train ensemble model
print("\nTraining Ensemble Model...")
ensemble_model.train(X_train_tensor, y_train_tensor, epochs=100)

# Evaluate models
standard_model.eval()
with torch.no_grad():
    standard_predictions = standard_model(X_test_tensor).numpy()

# Get predictions with uncertainty from Bayesian model
bayesian_model.eval()
n_samples = 100
bayesian_predictions = []

with torch.no_grad():
    for _ in range(n_samples):
        outputs = bayesian_model(X_test_tensor)
        bayesian_predictions.append(outputs.numpy())

bayesian_predictions = np.array(bayesian_predictions)
bayesian_mean = np.mean(bayesian_predictions, axis=0)
bayesian_std = np.std(bayesian_predictions, axis=0)

# Get predictions with uncertainty from ensemble model
ensemble_mean, ensemble_std = ensemble_model.predict(X_test_tensor)

# Calculate MSE
standard_mse = mean_squared_error(y_test, standard_predictions)
bayesian_mse = mean_squared_error(y_test, bayesian_mean)
ensemble_mse = mean_squared_error(y_test, ensemble_mean)

print(f"\nStandard Model MSE: {standard_mse:.4f}")
print(f"Bayesian Model MSE: {bayesian_mse:.4f}")
print(f"Ensemble Model MSE: {ensemble_mse:.4f}")

# 1. Visualize predictions vs actual values
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(y_test, standard_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Standard Model')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.subplot(1, 3, 2)
plt.scatter(y_test, bayesian_mean, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Bayesian Model')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.subplot(1, 3, 3)
plt.scatter(y_test, ensemble_mean, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Ensemble Model')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.tight_layout()
plt.show()

# 2. Visualize prediction intervals
# Sort by actual values for better visualization
sort_idx = np.argsort(y_test)
y_test_sorted = y_test[sort_idx]
standard_predictions_sorted = standard_predictions[sort_idx]
bayesian_mean_sorted = bayesian_mean[sort_idx]
bayesian_std_sorted = bayesian_std[sort_idx]
ensemble_mean_sorted = ensemble_mean[sort_idx]
ensemble_std_sorted = ensemble_std[sort_idx]

plt.figure(figsize=(15, 10))

# Bayesian model prediction intervals
plt.subplot(2, 1, 1)
plt.plot(y_test_sorted, 'k.', label='Actual Values')
plt.plot(bayesian_mean_sorted, 'r-', label='Predicted Mean')
plt.fill_between(range(len(y_test_sorted)), 
                 bayesian_mean_sorted.flatten() - 2 * bayesian_std_sorted.flatten(),
                 bayesian_mean_sorted.flatten() + 2 * bayesian_std_sorted.flatten(),
                 color='r', alpha=0.2, label='95% Prediction Interval')
plt.title('Bayesian Model with Prediction Intervals')
plt.xlabel('Sample Index (Sorted by Actual Value)')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Ensemble model prediction intervals
plt.subplot(2, 1, 2)
plt.plot(y_test_sorted, 'k.', label='Actual Values')
plt.plot(ensemble_mean_sorted, 'b-', label='Predicted Mean')
plt.fill_between(range(len(y_test_sorted)), 
                 ensemble_mean_sorted.flatten() - 2 * ensemble_std_sorted.flatten(),
                 ensemble_mean_sorted.flatten() + 2 * ensemble_std_sorted.flatten(),
                 color='b', alpha=0.2, label='95% Prediction Interval')
plt.title('Ensemble Model with Prediction Intervals')
plt.xlabel('Sample Index (Sorted by Actual Value)')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 3. Visualize uncertainty vs error
# Calculate absolute errors
standard_errors = np.abs(y_test - standard_predictions.flatten())
bayesian_errors = np.abs(y_test - bayesian_mean.flatten())
ensemble_errors = np.abs(y_test - ensemble_mean.flatten())

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(bayesian_std.flatten(), bayesian_errors, alpha=0.5)
plt.title('Bayesian Model: Uncertainty vs Error')
plt.xlabel('Predicted Standard Deviation')
plt.ylabel('Absolute Error')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.scatter(ensemble_std.flatten(), ensemble_errors, alpha=0.5)
plt.title('Ensemble Model: Uncertainty vs Error')
plt.xlabel('Predicted Standard Deviation')
plt.ylabel('Absolute Error')
plt.grid(True)

plt.subplot(1, 3, 3)
# Compare uncertainty calibration
# Sort by uncertainty
bayesian_sort_idx = np.argsort(bayesian_std.flatten())
ensemble_sort_idx = np.argsort(ensemble_std.flatten())

# Calculate mean error in bins of uncertainty
n_bins = 10
bayesian_bins = np.array_split(bayesian_errors[bayesian_sort_idx], n_bins)
ensemble_bins = np.array_split(ensemble_errors[ensemble_sort_idx], n_bins)

bayesian_mean_errors = [np.mean(bin) for bin in bayesian_bins]
ensemble_mean_errors = [np.mean(bin) for bin in ensemble_bins]

# Calculate mean uncertainty in bins
bayesian_std_bins = np.array_split(bayesian_std.flatten()[bayesian_sort_idx], n_bins)
ensemble_std_bins = np.array_split(ensemble_std.flatten()[ensemble_sort_idx], n_bins)

bayesian_mean_std = [np.mean(bin) for bin in bayesian_std_bins]
ensemble_mean_std = [np.mean(bin) for bin in ensemble_std_bins]

plt.plot(bayesian_mean_std, bayesian_mean_errors, 'ro-', label='Bayesian')
plt.plot(ensemble_mean_std, ensemble_mean_errors, 'bo-', label='Ensemble')
plt.plot([0, max(bayesian_mean_std + ensemble_mean_std)], [0, max(bayesian_mean_std + ensemble_mean_std)], 'k--')
plt.title('Uncertainty Calibration')
plt.xlabel('Mean Predicted Standard Deviation')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 4. Visualize predictive distribution for specific samples
sample_indices = [0, 10, 20, 30, 40]

plt.figure(figsize=(15, 10))
for i, idx in enumerate(sample_indices):
    plt.subplot(2, 3, i+1)
    
    # Plot Bayesian predictive distribution
    plt.hist(bayesian_predictions[:, idx], bins=20, alpha=0.5, density=True, color='red')
    plt.axvline(x=y_test[idx], color='k', linestyle='--', label='Actual Value')
    plt.axvline(x=bayesian_mean[idx], color='r', linestyle='-', label='Predicted Mean')
    plt.title(f'Sample {idx} (Bayesian)')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# 5. Visualize uncertainty across the input space
# Create a 2D grid for visualization
n_grid = 50
x1_min, x1_max = X_test[:, 0].min(), X_test[:, 0].max()
x2_min, x2_max = X_test[:, 1].min(), X_test[:, 1].max()

x1_grid = np.linspace(x1_min, x1_max, n_grid)
x2_grid = np.linspace(x2_min, x2_max, n_grid)
x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)

# Create grid points
grid_points = np.column_stack((x1_mesh.ravel(), x2_mesh.ravel()))

# Use only the first two features for visualization
grid_points_full = np.zeros((grid_points.shape[0], X_test.shape[1]))
grid_points_full[:, :2] = grid_points

# Scale the grid points
grid_points_scaled = scaler.transform(grid_points_full)

# Convert to PyTorch tensor
grid_tensor = torch.tensor(grid_points_scaled, dtype=torch.float32)

# Get predictions with uncertainty from Bayesian model
bayesian_model.eval()
n_samples = 50
bayesian_grid_predictions = []

with torch.no_grad():
    for _ in range(n_samples):
        outputs = bayesian_model(grid_tensor)
        bayesian_grid_predictions.append(outputs.numpy())

bayesian_grid_predictions = np.array(bayesian_grid_predictions)
bayesian_grid_mean = np.mean(bayesian_grid_predictions, axis=0)
bayesian_grid_std = np.std(bayesian_grid_predictions, axis=0)

# Reshape for plotting
bayesian_grid_mean = bayesian_grid_mean.reshape(x1_mesh.shape)
bayesian_grid_std = bayesian_grid_std.reshape(x1_mesh.shape)

# Plot the mean prediction
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.contourf(x1_mesh, x2_mesh, bayesian_grid_mean, 20, cmap='viridis')
plt.colorbar(label='Predicted Value')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', s=10, alpha=0.5)
plt.title('Mean Prediction')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')

# Plot the uncertainty
plt.subplot(1, 2, 2)
plt.contourf(x1_mesh, x2_mesh, bayesian_grid_std, 20, cmap='plasma')
plt.colorbar(label='Standard Deviation')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', s=10, alpha=0.5)
plt.title('Prediction Uncertainty')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')

plt.tight_layout()
plt.show()

# 6. Visualize ensemble diversity
# Get predictions from each model in the ensemble
ensemble_individual_predictions = []
for model in ensemble_model.models:
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        ensemble_individual_predictions.append(outputs.numpy())

ensemble_individual_predictions = np.array(ensemble_individual_predictions)

# Plot predictions from each model
plt.figure(figsize=(15, 10))
for i in range(min(5, len(ensemble_individual_predictions))):  # Plot first 5 models
    plt.subplot(2, 3, i+1)
    plt.scatter(y_test, ensemble_individual_predictions[i], alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'Ensemble Model {i+1}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)

plt.tight_layout()
plt.show()

# 7. Visualize uncertainty in a decision-making context
# Define a decision threshold
threshold = 0.0

# Calculate decision confidence
bayesian_confidence = np.abs(bayesian_mean - threshold) / bayesian_std
ensemble_confidence = np.abs(ensemble_mean - threshold) / ensemble_std

# Plot confidence vs error
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.scatter(bayesian_confidence.flatten(), bayesian_errors, alpha=0.5)
plt.title('Bayesian Model: Confidence vs Error')
plt.xlabel('Decision Confidence')
plt.ylabel('Absolute Error')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(ensemble_confidence.flatten(), ensemble_errors, alpha=0.5)
plt.title('Ensemble Model: Confidence vs Error')
plt.xlabel('Decision Confidence')
plt.ylabel('Absolute Error')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 4. Problem: Visualizing Model Performance Across Different Subgroups

**Issue:**
Models may perform differently across different subgroups of data, and visualizing these differences is important for fairness and model evaluation.

**How advancements solved this:**
- **Subgroup analysis techniques**: Methods to analyze and visualize performance across different subgroups
- **Fairness metrics visualization**: Visual representations of fairness metrics across groups
- **Interactive dashboards**: Tools that allow users to explore performance across different dimensions
- **Comparative visualization**: Methods to directly compare performance across groups
- **Error analysis visualization**: Techniques to understand where and why models fail for different groups

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic data with subgroups
n_samples = 5000
n_features = 10

# Create data with different characteristics for different subgroups
X1, y1 = make_classification(n_samples=n_samples//2, n_features=n_features, n_informative=5, 
                            n_redundant=3, weights=[0.8, 0.2], random_state=42)
X2, y2 = make_classification(n_samples=n_samples//2, n_features=n_features, n_informative=5, 
                            n_redundant=3, weights=[0.6, 0.4], random_state=43)

# Combine the data
X = np.vstack((X1, X2))
y = np.hstack((y1, y2))

# Create subgroup labels
subgroups = np.array(['Group A'] * (n_samples//2) + ['Group B'] * (n_samples//2))

# Add additional subgroup dimensions
ages = np.concatenate([
    np.random.normal(40, 10, n_samples//2),
    np.random.normal(35, 15, n_samples//2)
])

# Create age groups
age_groups = np.array(['Young'] * n_samples)
age_groups[ages > 40] = 'Old'

# Create a DataFrame
df = pd.DataFrame(X, columns=[f'Feature {i}' for i in range(n_features)])
df['Target'] = y
df['Subgroup'] = subgroups
df['Age'] = ages
df['Age_Group'] = age_groups

# Split the data
X_train, X_test, y_train, y_test, subgroups_train, subgroups_test, age_groups_train, age_groups_test = train_test_split(
    X, y, subgroups, age_groups, test_size=0.2, random_state=42, stratify=y
)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define a neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize the model
input_size = X_train.shape[1]
hidden_size = 64
num_classes = len(np.unique(y))
model = NeuralNetwork(input_size, hidden_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    probabilities = torch.softmax(outputs, dim=1)[:, 1]

# Create a DataFrame for analysis
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': predicted.numpy(),
    'Probability': probabilities.numpy(),
    'Subgroup': subgroups_test,
    'Age_Group': age_groups_test
})

# 1. Overall performance metrics
print("\n=== Overall Performance ===")
print(classification_report(y_test, predicted))

# 2. Performance by subgroup
print("\n=== Performance by Subgroup ===")
for subgroup in results_df['Subgroup'].unique():
    subgroup_df = results_df[results_df['Subgroup'] == subgroup]
    print(f"\nSubgroup: {subgroup}")
    print(classification_report(subgroup_df['Actual'], subgroup_df['Predicted']))

# 3. Performance by age group
print("\n=== Performance by Age Group ===")
for age_group in results_df['Age_Group'].unique():
    age_group_df = results_df[results_df['Age_Group'] == age_group]
    print(f"\nAge Group: {age_group}")
    print(classification_report(age_group_df['Actual'], age_group_df['Predicted']))

# 4. Visualize confusion matrices
plt.figure(figsize=(15, 5))

# Overall confusion matrix
plt.subplot(1, 3, 1)
cm = confusion_matrix(y_test, predicted)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['Class 0', 'Class 1'],
           yticklabels=['Class 0', 'Class 1'])
plt.title('Overall Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Confusion matrix for Group A
plt.subplot(1, 3, 2)
group_a_df = results_df[results_df['Subgroup'] == 'Group A']
cm_a = confusion_matrix(group_a_df['Actual'], group_a_df['Predicted'])
sns.heatmap(cm_a, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['Class 0', 'Class 1'],
           yticklabels=['Class 0', 'Class 1'])
plt.title('Group A Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Confusion matrix for Group B
plt.subplot(1, 3, 3)
group_b_df = results_df[results_df['Subgroup'] == 'Group B']
cm_b = confusion_matrix(group_b_df['Actual'], group_b_df['Predicted'])
sns.heatmap(cm_b, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['Class 0', 'Class 1'],
           yticklabels=['Class 0', 'Class 1'])
plt.title('Group B Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# 5. Visualize ROC curves by subgroup
plt.figure(figsize=(10, 8))

# Overall ROC curve
fpr, tpr, _ = roc_curve(y_test, results_df['Probability'])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'Overall (AUC = {roc_auc:.2f})')

# ROC curves by subgroup
for subgroup in results_df['Subgroup'].unique():
    subgroup_df = results_df[results_df['Subgroup'] == subgroup]
    fpr, tpr, _ = roc_curve(subgroup_df['Actual'], subgroup_df['Probability'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{subgroup} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves by Subgroup')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 6. Visualize precision-recall curves by subgroup
plt.figure(figsize=(10, 8))

# Overall precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, results_df['Probability'])
ap = average_precision_score(y_test, results_df['Probability'])
plt.plot(recall, precision, label=f'Overall (AP = {ap:.2f})')

# Precision-recall curves by subgroup
for subgroup in results_df['Subgroup'].unique():
    subgroup_df = results_df[results_df['Subgroup'] == subgroup]
    precision, recall, _ = precision_recall_curve(subgroup_df['Actual'], subgroup_df['Probability'])
    ap = average_precision_score(subgroup_df['Actual'], subgroup_df['Probability'])
    plt.plot(recall, precision, label=f'{subgroup} (AP = {ap:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves by Subgroup')
plt.legend(loc="lower left")
plt.grid(True)
plt.show()

# 7. Visualize calibration curves by subgroup
plt.figure(figsize=(10, 8))

# Overall calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, results_df['Probability'], n_bins=10)
plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Overall')

# Calibration curves by subgroup
for subgroup in results_df['Subgroup'].unique():
    subgroup_df = results_df[results_df['Subgroup'] == subgroup]
    fraction_of_positives, mean_predicted_value = calibration_curve(
        subgroup_df['Actual'], subgroup_df['Probability'], n_bins=10
    )
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=subgroup)

plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curves by Subgroup')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 8. Visualize performance metrics by subgroup
# Calculate metrics for each subgroup
subgroup_metrics = []
for subgroup in results_df['Subgroup'].unique():
    subgroup_df = results_df[results_df['Subgroup'] == subgroup]
    
    # Calculate metrics
    cm = confusion_matrix(subgroup_df['Actual'], subgroup_df['Predicted'])
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    subgroup_metrics.append({
        'Subgroup': subgroup,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    })

metrics_df = pd.DataFrame(subgroup_metrics)

# Plot metrics by subgroup
plt.figure(figsize=(12, 6))
metrics_df.set_index('Subgroup').plot(kind='bar', figsize=(12, 6))
plt.title('Performance Metrics by Subgroup')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.grid(True, axis='y')
plt.legend(loc='lower right')
plt.show()

# 9. Visualize error rates by subgroup and age group
# Create a pivot table of error rates
error_rates = results_df.copy()
error_rates['Correct'] = (error_rates['Actual'] == error_rates['Predicted']).astype(int)
error_pivot = error_rates.groupby(['Subgroup', 'Age_Group'])['Correct'].mean().reset_index()
error_pivot['Error Rate'] = 1 - error_pivot['Correct']

# Plot error rates
plt.figure(figsize=(10, 6))
sns.barplot(x='Subgroup', y='Error Rate', hue='Age_Group', data=error_pivot)
plt.title('Error Rates by Subgroup and Age Group')
plt.ylabel('Error Rate')
plt.ylim(0, 1)
plt.grid(True, axis='y')
plt.legend(title='Age Group')
plt.show()

# 10. Visualize probability distributions by subgroup and actual class
plt.figure(figsize=(15, 6))

# Group A
plt.subplot(1, 2, 1)
group_a_df = results_df[results_df['Subgroup'] == 'Group A']
sns.histplot(data=group_a_df, x='Probability', hue='Actual', kde=True, bins=20, alpha=0.5)
plt.title('Probability Distribution by Actual Class (Group A)')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')

# Group B
plt.subplot(1, 2, 2)
group_b_df = results_df[results_df['Subgroup'] == 'Group B']
sns.histplot(data=group_b_df, x='Probability', hue='Actual', kde=True, bins=20, alpha=0.5)
plt.title('Probability Distribution by Actual Class (Group B)')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# 11. Interactive visualization with Plotly
# Create an interactive scatter plot of predicted probabilities vs. actual values
fig = px.scatter(
    results_df, 
    x='Probability', 
    y='Actual',
    color='Subgroup',
    symbol='Age_Group',
    title='Predicted Probabilities vs. Actual Values',
    hover_data=['Predicted'],
    opacity=0.7
)
fig.add_shape(
    type="line", line=dict(dash="dot"),
    x0=0, x1=1, y0=0, y1=1
)
fig.update_layout(
    xaxis_title="Predicted Probability",
    yaxis_title="Actual Value",
    legend_title="Subgroup"
)
fig.show()

# Create an interactive bar chart of performance metrics
fig = px.bar(
    metrics_df.melt(id_vars='Subgroup', var_name='Metric', value_name='Score'),
    x='Subgroup',
    y='Score',
    color='Metric',
    barmode='group',
    title='Performance Metrics by Subgroup'
)
fig.update_layout(
    yaxis_title='Score',
    legend_title='Metric'
)
fig.show()

# Create an interactive heatmap of error rates
fig = px.density_heatmap(
    error_pivot,
    x='Subgroup',
    y='Age_Group',
    z='Error Rate',
    title='Error Rates by Subgroup and Age Group',
    color_continuous_scale='Viridis'
)
fig.show()

# 12. Disparate impact analysis
# Calculate selection rates by subgroup
threshold = 0.5  # Classification threshold

selection_rates = []
for subgroup in results_df['Subgroup'].unique():
    subgroup_df = results_df[results_df['Subgroup'] == subgroup]
    
    # Calculate selection rate (positive predictions)
    selection_rate = (subgroup_df['Predicted'] == 1).mean()
    
    selection_rates.append({
        'Subgroup': subgroup,
        'Selection Rate': selection_rate
    })

selection_rates_df = pd.DataFrame(selection_rates)

# Calculate disparate impact ratio
base_rate = selection_rates_df[selection_rates_df['Subgroup'] == 'Group A']['Selection Rate'].values[0]
for idx, row in selection_rates_df.iterrows():
    if row['Subgroup'] != 'Group A':
        ratio = row['Selection Rate'] / base_rate
        selection_rates_df.loc[idx, 'Disparate Impact Ratio'] = ratio

selection_rates_df.loc[selection_rates_df['Subgroup'] == 'Group A', 'Disparate Impact Ratio'] = 1.0

# Plot selection rates and disparate impact ratios
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.barplot(x='Subgroup', y='Selection Rate', data=selection_rates_df)
plt.title('Selection Rates by Subgroup')
plt.ylabel('Selection Rate')
plt.ylim(0, 1)
plt.grid(True, axis='y')

plt.subplot(1, 2, 2)
sns.barplot(x='Subgroup', y='Disparate Impact Ratio', data=selection_rates_df)
plt.title('Disparate Impact Ratios')
plt.ylabel('Ratio')
plt.axhline(y=0.8, color='r', linestyle='--', label='Threshold (0.8)')
plt.legend()
plt.grid(True, axis='y')

plt.tight_layout()
plt.show()
```

## Project: Startup Funding Dashboard

Now, let's implement a comprehensive startup funding dashboard using the visualization techniques we've covered:

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
from datetime import datetime, timedelta

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic startup funding data
def generate_startup_data(n_startups=1000, n_years=5):
    """Generate synthetic startup funding data"""
    startups = []
    
    # Startup categories
    categories = ['AI/ML', 'FinTech', 'HealthTech', 'EdTech', 'E-commerce', 'SaaS', 'Biotech', 'CleanTech']
    
    # Funding stages
    stages = ['Seed', 'Series A', 'Series B', 'Series C', 'Series D+', 'Acquired', 'IPO']
    
    # Regions
    regions = ['North America', 'Europe', 'Asia', 'South America', 'Africa', 'Oceania']
    
    # Generate startups
    for i in range(n_startups):
        # Basic info
        name = f"Startup_{i}"
        category = np.random.choice(categories)
        region = np.random.choice(regions)
        
        # Founding date
        founding_year = np.random.randint(2015, 2022)
        founding_month = np.random.randint(1, 13)
        founding_day = np.random.randint(1, 29)  # Simplified, not handling month length
        founding_date = datetime(founding_year, founding_month, founding_day)
        
        # Funding rounds
        n_rounds = np.random.randint(1, 6)
        current_stage = stages[min(n_rounds - 1, 4)]  # Limit to Series D+ at most
        
        # Total funding (in millions)
        if category == 'AI/ML':
            base_funding = np.random.lognormal(2, 0.5)
        elif category == 'Biotech':
            base_funding = np.random.lognormal(2.2, 0.6)
        elif category == 'FinTech':
            base_funding = np.random.lognormal(1.8, 0.7)
        else:
            base_funding = np.random.lognormal(1.5, 0.8)
        
        # Adjust by region
        if region == 'North America':
            base_funding *= 1.2
        elif region == 'Asia':
            base_funding *= 1.1
        elif region == 'Europe':
            base_funding *= 1.0
        else:
            base_funding *= 0.8
        
        # Add some randomness
        total_funding = base_funding * np.random.uniform(0.8, 1.2)
        
        # Employee count
        if total_funding > 100:
            employees = int(np.random.normal(300, 100))
        elif total_funding > 10:
            employees = int(np.random.normal(100, 50))
        else:
            employees = int(np.random.normal(20, 10))
        
        employees = max(1, employees)  # At least 1 employee
        
        # Valuation (if available)
        if current_stage in ['Series C', 'Series D+', 'Acquired', 'IPO']:
            # Valuation is typically 5-10x funding for later stages
            valuation = total_funding * np.random.uniform(5, 10)
        else:
            valuation = np.nan
        
        # Status
        if current_stage in ['Acquired', 'IPO']:
            status = current_stage
        else:
            status = 'Active'
        
        # Add to list
        startups.append({
            'Name': name,
            'Category': category,
            'Region': region,
            'Founding_Date': founding_date,
            'Founding_Year': founding_year,
            'Founding_Month': founding_month,
            'Current_Stage': current_stage,
            'Total_Funding_M': total_funding,
            'Employee_Count': employees,
            'Valuation_M': valuation,
            'Status': status
        })
    
    return pd.DataFrame(startups)

# Generate the data
startup_df = generate_startup_data(n_startups=1000, n_years=5)

# Add some derived columns
startup_df['Years_Since_Founding'] = 2022 - startup_df['Founding_Year']
startup_df['Funding_Per_Employee'] = startup_df['Total_Funding_M'] / startup_df['Employee_Count']

# Generate funding events data
def generate_funding_events(startup_df):
    """Generate funding events for startups"""
    events = []
    
    for _, startup in startup_df.iterrows():
        name = startup['Name']
        category = startup['Category']
        region = startup['Region']
        founding_date = startup['Founding_Date']
        current_stage = startup['Current_Stage']
        total_funding = startup['Total_Funding_M']
        
        # Determine number of funding rounds
        if current_stage == 'Seed':
            n_rounds = 1
        elif current_stage == 'Series A':
            n_rounds = 2
        elif current_stage == 'Series B':
            n_rounds = 3
        elif current_stage == 'Series C':
            n_rounds = 4
        elif current_stage == 'Series D+':
            n_rounds = 5
        else:
            n_rounds = 6  # Acquired or IPO
        
        # Generate funding amounts for each round
        round_names = ['Seed', 'Series A', 'Series B', 'Series C', 'Series D', 'Series E']
        
        # Distribute total funding across rounds
        # Early rounds are smaller, later rounds are larger
        if n_rounds == 1:
            round_amounts = [total_funding]
        else:
            # Generate increasing amounts
            base_amount = total_funding / (2 ** n_rounds - 1)
            round_amounts = [base_amount * (2 ** i) for i in range(n_rounds)]
            
            # Add some randomness
            round_amounts = [amt * np.random.uniform(0.8, 1.2) for amt in round_amounts]
            
            # Normalize to sum to total funding
            round_amounts = [amt * total_funding / sum(round_amounts) for amt in round_amounts]
        
        # Generate dates for each round
        dates = []
        for i in range(n_rounds):
            # Each round is roughly 1-2 years apart
            years_since_founding = (i + 1) * np.random.uniform(1, 2)
            event_date = founding_date + timedelta(days=years_since_founding * 365)
            dates.append(event_date)
        
        # Create events
        for i in range(n_rounds):
            events.append({
                'Startup_Name': name,
                'Category': category,
                'Region': region,
                'Round': round_names[i],
                'Amount_M': round_amounts[i],
                'Date': dates[i],
                'Year': dates[i].year,
                'Month': dates[i].month
            })
    
    return pd.DataFrame(events)

# Generate funding events
funding_events_df = generate_funding_events(startup_df)

# 1. Static Visualizations with Matplotlib and Seaborn
print("Creating static visualizations...")

# Overall funding distribution
plt.figure(figsize=(12, 6))
sns.histplot(startup_df['Total_Funding_M'], bins=50, kde=True)
plt.title('Distribution of Total Funding')
plt.xlabel('Total Funding ($M)')
plt.ylabel('Count')
plt.xscale('log')
plt.grid(True, alpha=0.3)
plt.show()

# Funding by category
plt.figure(figsize=(12, 6))
sns.boxplot(x='Category', y='Total_Funding_M', data=startup_df)
plt.title('Funding by Category')
plt.xlabel('Category')
plt.ylabel('Total Funding ($M)')
plt.yscale('log')
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.show()

# Funding by region
plt.figure(figsize=(12, 6))
sns.boxplot(x='Region', y='Total_Funding_M', data=startup_df)
plt.title('Funding by Region')
plt.xlabel('Region')
plt.ylabel('Total Funding ($M)')
plt.yscale('log')
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.show()

# Funding by stage
plt.figure(figsize=(12, 6))
sns.boxplot(x='Current_Stage', y='Total_Funding_M', data=startup_df)
plt.title('Funding by Current Stage')
plt.xlabel('Current Stage')
plt.ylabel('Total Funding ($M)')
plt.yscale('log')
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.show()

# Number of startups by category and region
plt.figure(figsize=(12, 8))
category_region_counts = startup_df.groupby(['Category', 'Region']).size().unstack()
sns.heatmap(category_region_counts, annot=True, fmt='d', cmap='viridis')
plt.title('Number of Startups by Category and Region')
plt.xlabel('Region')
plt.ylabel('Category')
plt.tight_layout()
plt.show()

# Funding trends over time
funding_by_year = funding_events_df.groupby('Year')['Amount_M'].sum().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(funding_by_year['Year'], funding_by_year['Amount_M'], marker='o')
plt.title('Total Funding by Year')
plt.xlabel('Year')
plt.ylabel('Total Funding ($M)')
plt.grid(True)
plt.show()

# Funding trends by category over time
category_year_funding = funding_events_df.groupby(['Year', 'Category'])['Amount_M'].sum().reset_index()

plt.figure(figsize=(12, 8))
for category in category_year_funding['Category'].unique():
    category_data = category_year_funding[category_year_funding['Category'] == category]
    plt.plot(category_data['Year'], category_data['Amount_M'], marker='o', label=category)

plt.title('Funding Trends by Category')
plt.xlabel('Year')
plt.ylabel('Total Funding ($M)')
plt.legend()
plt.grid(True)
plt.show()

# Employee count vs. funding
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Employee_Count', y='Total_Funding_M', hue='Category', data=startup_df, alpha=0.7)
plt.title('Employee Count vs. Total Funding')
plt.xlabel('Employee Count')
plt.ylabel('Total Funding ($M)')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Funding rounds distribution
plt.figure(figsize=(12, 6))
round_counts = funding_events_df['Round'].value_counts()
sns.barplot(x=round_counts.index, y=round_counts.values)
plt.title('Distribution of Funding Rounds')
plt.xlabel('Funding Round')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.show()

# Average funding by round and category
round_category_funding = funding_events_df.groupby(['Round', 'Category'])['Amount_M'].mean().reset_index()

plt.figure(figsize=(12, 8))
for category in round_category_funding['Category'].unique():
    category_data = round_category_funding[round_category_funding['Category'] == category]
    plt.plot(category_data['Round'], category_data['Amount_M'], marker='o', label=category)

plt.title('Average Funding by Round and Category')
plt.xlabel('Funding Round')
plt.ylabel('Average Funding ($M)')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

# 2. Interactive Visualizations with Plotly
print("Creating interactive visualizations...")

# Interactive funding distribution
fig = px.histogram(
    startup_df, 
    x='Total_Funding_M',
    nbins=50,
    title='Distribution of Total Funding',
    labels={'Total_Funding_M': 'Total Funding ($M)'}
)
fig.update_layout(
    xaxis_type="log",
    bargap=0.1
)
fig.show()

# Interactive funding by category
fig = px.box(
    startup_df,
    x='Category',
    y='Total_Funding_M',
    title='Funding by Category',
    labels={'Total_Funding_M': 'Total Funding ($M)'}
)
fig.update_layout(
    yaxis_type="log"
)
fig.show()

# Interactive funding by region
fig = px.box(
    startup_df,
    x='Region',
    y='Total_Funding_M',
    title='Funding by Region',
    labels={'Total_Funding_M': 'Total Funding ($M)'}
)
fig.update_layout(
    yaxis_type="log"
)
fig.show()

# Interactive funding trends over time
fig = px.line(
    funding_by_year,
    x='Year',
    y='Amount_M',
    title='Total Funding by Year',
    labels={'Amount_M': 'Total Funding ($M)'},
    markers=True
)
fig.update_layout(
    xaxis=dict(tickmode='linear'),
    yaxis_type="log"
)
fig.show()

# Interactive funding trends by category
fig = px.line(
    category_year_funding,
    x='Year',
    y='Amount_M',
    color='Category',
    title='Funding Trends by Category',
    labels={'Amount_M': 'Total Funding ($M)'},
    markers=True,
    line_shape='linear'
)
fig.update_layout(
    xaxis=dict(tickmode='linear'),
    yaxis_type="log"
)
fig.show()

# Interactive employee count vs. funding
fig = px.scatter(
    startup_df,
    x='Employee_Count',
    y='Total_Funding_M',
    color='Category',
    size='Years_Since_Founding',
    hover_data=['Name', 'Current_Stage'],
    title='Employee Count vs. Total Funding',
    labels={
        'Employee_Count': 'Employee Count',
        'Total_Funding_M': 'Total Funding ($M)',
        'Years_Since_Founding': 'Years Since Founding'
    },
    opacity=0.7
)
fig.update_layout(
    xaxis_type="log",
    yaxis_type="log"
)
fig.show()

# Interactive heatmap of startups by category and region
category_region_counts = startup_df.groupby(['Category', 'Region']).size().reset_index(name='Count')

fig = px.density_heatmap(
    category_region_counts,
    x='Category',
    y='Region',
    z='Count',
    title='Number of Startups by Category and Region',
    color_continuous_scale='Viridis'
)
fig.show()

# Interactive average funding by round and category
fig = px.line(
    round_category_funding,
    x='Round',
    y='Amount_M',
    color='Category',
    title='Average Funding by Round and Category',
    labels={'Amount_M': 'Average Funding ($M)'},
    markers=True
)
fig.update_layout(
    yaxis_type="log"
)
fig.show()

# 3. Advanced Dashboard with Multiple Visualizations
print("Creating advanced dashboard...")

# Create a comprehensive dashboard
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('Funding by Category', 'Funding by Region', 
                  'Funding Over Time', 'Funding by Stage',
                  'Employee Count vs. Funding', 'Funding Rounds'),
    specs=[[{"type": "bar"}, {"type": "bar"}],
           [{"type": "scatter"}, {"type": "bar"}],
           [{"type": "scatter"}, {"type": "bar"}]]
)

# Funding by category
category_funding = startup_df.groupby('Category')['Total_Funding_M'].sum().reset_index()
fig.add_trace(
    go.Bar(x=category_funding['Category'], y=category_funding['Total_Funding_M']),
    row=1, col=1
)

# Funding by region
region_funding = startup_df.groupby('Region')['Total_Funding_M'].sum().reset_index()
fig.add_trace(
    go.Bar(x=region_funding['Region'], y=region_funding['Total_Funding_M']),
    row=1, col=2
)

# Funding over time
fig.add_trace(
    go.Scatter(x=funding_by_year['Year'], y=funding_by_year['Amount_M'], mode='lines+markers'),
    row=2, col=1
)

# Funding by stage
stage_funding = startup_df.groupby('Current_Stage')['Total_Funding_M'].sum().reset_index()
fig.add_trace(
    go.Bar(x=stage_funding['Current_Stage'], y=stage_funding['Total_Funding_M']),
    row=2, col=2
)

# Employee count vs. funding
fig.add_trace(
    go.Scatter(
        x=startup_df['Employee_Count'],
        y=startup_df['Total_Funding_M'],
        mode='markers',
        marker=dict(size=5, opacity=0.7),
        text=startup_df['Name'],
        hovertemplate='<b>%{text}</b><br>Employees: %{x}<br>Funding: $%{y:.2f}M<extra></extra>'
    ),
    row=3, col=1
)

# Funding rounds
round_counts = funding_events_df['Round'].value_counts().reset_index()
fig.add_trace(
    go.Bar(x=round_counts['index'], y=round_counts['Round']),
    row=3, col=2
)

# Update layout
fig.update_layout(
    height=1200,
    title_text="Startup Funding Dashboard",
    showlegend=False
)

# Update y-axes to log scale where appropriate
fig.update_yaxes(type="log", row=1, col=1)
fig.update_yaxes(type="log", row=1, col=2)
fig.update_yaxes(type="log", row=2, col=1)
fig.update_yaxes(type="log", row=2, col=2)
fig.update_yaxes(type="log", row=3, col=1)

fig.show()

# 4. Time Series Analysis
print("Creating time series analysis...")

# Monthly funding trends
funding_events_df['YearMonth'] = funding_events_df['Date'].dt.to_period('M')
monthly_funding = funding_events_df.groupby('YearMonth')['Amount_M'].sum().reset_index()
monthly_funding['YearMonth'] = monthly_funding['YearMonth'].dt.to_timestamp()

plt.figure(figsize=(12, 6))
plt.plot(monthly_funding['YearMonth'], monthly_funding['Amount_M'], marker='o')
plt.title('Monthly Funding Trends')
plt.xlabel('Date')
plt.ylabel('Total Funding ($M)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Interactive monthly funding trends
fig = px.line(
    monthly_funding,
    x='YearMonth',
    y='Amount_M',
    title='Monthly Funding Trends',
    labels={'Amount_M': 'Total Funding ($M)'},
    markers=True
)
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Total Funding ($M)"
)
fig.show()

# 5. Geospatial Analysis (simulated)
print("Creating geospatial analysis...")

# Create a simple mapping of regions to coordinates (simplified for demonstration)
region_coordinates = {
    'North America': {'lat': 45, 'lon': -100},
    'Europe': {'lat': 50, 'lon': 10},
    'Asia': {'lat': 30, 'lon': 100},
    'South America': {'lat': -15, 'lon': -60},
    'Africa': {'lat': 0, 'lon': 20},
    'Oceania': {'lat': -25, 'lon': 135}
}

# Aggregate funding by region
region_funding = startup_df.groupby('Region')['Total_Funding_M'].sum().reset_index()

# Add coordinates
region_funding['lat'] = region_funding['Region'].map(lambda x: region_coordinates[x]['lat'])
region_funding['lon'] = region_funding['Region'].map(lambda x: region_coordinates[x]['lon'])

# Create a simple map using scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(region_funding['lon'], region_funding['lat'], s=region_funding['Total_Funding_M']/10, alpha=0.7)
for i, row in region_funding.iterrows():
    plt.annotate(row['Region'], (row['lon'], row['lat']), fontsize=12)
plt.title('Funding by Region (Bubble Size = Total Funding)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.tight_layout()
plt.show()

# Interactive map using Plotly
fig = px.scatter_geo(
    region_funding,
    lat='lat',
    lon='lon',
    size='Total_Funding_M',
    hover_name='Region',
    size_max=50,
    title='Funding by Region',
    projection='natural earth'
)
fig.show()

# 6. Correlation Analysis
print("Creating correlation analysis...")

# Select numerical columns for correlation analysis
numerical_cols = ['Total_Funding_M', 'Employee_Count', 'Years_Since_Founding', 'Funding_Per_Employee']
correlation_matrix = startup_df[numerical_cols].corr()

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Interactive correlation matrix
fig = px.imshow(
    correlation_matrix,
    text_auto=True,
    aspect="auto",
    color_continuous_scale='RdBu_r',
    title='Correlation Matrix'
)
fig.show()

# 7. Summary Statistics
print("\n=== Summary Statistics ===")
print(startup_df.describe())

print("\n=== Funding by Category ===")
category_stats = startup_df.groupby('Category')['Total_Funding_M'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
print(category_stats)

print("\n=== Funding by Region ===")
region_stats = startup_df.groupby('Region')['Total_Funding_M'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
print(region_stats)

print("\n=== Funding by Stage ===")
stage_stats = startup_df.groupby('Current_Stage')['Total_Funding_M'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
print(stage_stats)

print("\n=== Top 10 Most Funded Startups ===)
top_startups = startup_df.nlargest(10, 'Total_Funding_M')[['Name', 'Category', 'Region', 'Total_Funding_M', 'Current_Stage']]
print(top_startups)

print("\n=== Funding Events by Round ===")
round_stats = funding_events_df.groupby('Round')['Amount_M'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
print(round_stats)

print("\n=== Dashboard Complete ===")
```

This startup funding dashboard demonstrates a comprehensive approach to data visualization, incorporating:

1. **Static visualizations** using Matplotlib and Seaborn for basic exploration and analysis
2. **Interactive visualizations** using Plotly for deeper exploration and presentation
3. **Advanced dashboard** combining multiple visualizations in a single view
4. **Time series analysis** to understand trends over time
5. **Geospatial analysis** to visualize regional differences
6. **Correlation analysis** to understand relationships between variables
7. **Summary statistics** to provide key insights

The dashboard provides insights into:
- Distribution of funding across startups
- Differences in funding by category, region, and stage
- Trends in funding over time
- Relationship between employee count and funding
- Geographic distribution of funding
- Correlations between different variables

This type of comprehensive visualization approach is essential for deep learning projects, as it helps understand the data, identify patterns, and communicate findings effectively.

## Summary

Data visualization is a critical component of the deep learning pipeline. In this lesson, we've covered:

1. **Basic Plotting with Matplotlib**: Creating fundamental visualizations like line plots, scatter plots, histograms, and bar charts
2. **Statistical Visualization with Seaborn**: Creating more advanced statistical visualizations with less code
3. **Interactive Visualization with Plotly**: Building interactive, web-based visualizations for exploration and presentation
4. **Visualization for Model Interpretation**: Creating visualizations to understand and interpret machine learning models
5. **Advanced Visualization Techniques**: Methods for visualizing high-dimensional data and complex relationships

We've also explored how these concepts are applied in deep learning, common problems encountered, and how advancements in the field have addressed these challenges.

The startup funding dashboard project provides a practical implementation of these concepts, demonstrating how to create a comprehensive visualization system that combines multiple