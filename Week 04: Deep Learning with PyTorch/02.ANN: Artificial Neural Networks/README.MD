# Day 17 – ANN: Artificial Neural Networks

## What are Artificial Neural Networks and Why are They Important for Deep Learning?

Artificial Neural Networks (ANNs) are computing systems inspired by the biological neural networks that constitute animal brains. They are the foundation of deep learning and consist of interconnected nodes (neurons) organized in layers. Each connection between neurons has a weight that adjusts as learning proceeds, allowing the network to adapt to inputs and generate appropriate outputs.

ANNs are essential because:
- They can model complex non-linear relationships in data
- They can learn from data without being explicitly programmed with rules
- They form the building blocks for more advanced architectures like CNNs, RNNs, and Transformers
- They have achieved state-of-the-art performance in many domains
- They can automatically extract features from raw data

## Essential Subtopics

### 1. Neurons and Layers

**What are neurons and layers in ANNs?**
In ANNs, a neuron (or node) is a mathematical function that models the behavior of a biological neuron. It receives inputs, applies weights to them, adds a bias, and passes the result through an activation function. Layers are collections of neurons that process information in parallel. The most common types of layers are:
- Input layer: Receives the raw data
- Hidden layers: Intermediate processing layers
- Output layer: Produces the final prediction

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Define a single neuron
class SingleNeuron(nn.Module):
    def __init__(self, input_size):
        super(SingleNeuron, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_size))
        self.bias = nn.Parameter(torch.randn(1))
    
    def forward(self, x):
        # Linear transformation: z = x·w + b
        z = torch.matmul(x, self.weights) + self.bias
        # Apply activation function (sigmoid)
        a = torch.sigmoid(z)
        return a

# Create a simple neural network with multiple layers
class SimpleANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleANN, self).__init__()
        # Input layer to first hidden layer
        self.layer1 = nn.Linear(input_size, hidden_size)
        # First hidden layer to second hidden layer
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        # Second hidden layer to output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Input layer to first hidden layer with ReLU activation
        x = F.relu(self.layer1(x))
        # First hidden layer to second hidden layer with ReLU activation
        x = F.relu(self.layer2(x))
        # Output layer (no activation for now, will be applied later)
        x = self.output_layer(x)
        return x

# Example usage
input_size = 10
hidden_size = 20
output_size = 2

# Create a single neuron
neuron = SingleNeuron(input_size)
print(f"Single neuron weights: {neuron.weights}")
print(f"Single neuron bias: {neuron.bias}")

# Create a simple ANN
model = SimpleANN(input_size, hidden_size, output_size)
print("\nSimple ANN architecture:")
print(model)

# Visualize the network architecture
def visualize_network(model):
    # This is a simplified visualization for demonstration
    print("\nNetwork Architecture Visualization:")
    print("Input Layer")
    print("    ↓")
    print("Hidden Layer 1 (with ReLU activation)")
    print("    ↓")
    print("Hidden Layer 2 (with ReLU activation)")
    print("    ↓")
    print("Output Layer")

visualize_network(model)

# Test the model with random input
random_input = torch.randn(1, input_size)
output = model(random_input)
print(f"\nOutput for random input: {output}")
```

**When and where we use neurons and layers:**
- In virtually all deep learning models as the fundamental building blocks
- When we need to model complex non-linear relationships
- In both regression and classification tasks
- In feature extraction and transformation
- When we need to learn hierarchical representations of data

**Real-world examples:**
- In image recognition, where early layers detect edges and textures, middle layers detect patterns, and final layers identify objects
- In natural language processing, where layers process words, phrases, and sentences to understand meaning
- In financial forecasting, where layers process raw market data to predict trends

### 2. Activation Functions

**What are activation functions?**
Activation functions introduce non-linearity into neural networks, allowing them to learn and model complex patterns. Without activation functions, neural networks would essentially be linear regression models, regardless of their depth. Common activation functions include:

- Sigmoid: Maps inputs to values between 0 and 1
- Tanh: Maps inputs to values between -1 and 1
- ReLU (Rectified Linear Unit): Outputs the input if positive, otherwise 0
- Leaky ReLU: Similar to ReLU but allows a small gradient when the unit is not active
- Softmax: Converts a vector of values to a probability distribution

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Define a neural network to demonstrate different activation functions
class ActivationDemo(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActivationDemo, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, activation='relu'):
        x = self.fc1(x)
        
        # Apply different activation functions
        if activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif activation == 'tanh':
            x = torch.tanh(x)
        elif activation == 'relu':
            x = F.relu(x)
        elif activation == 'leaky_relu':
            x = F.leaky_relu(x, negative_slope=0.01)
        elif activation == 'softmax':
            # Note: Softmax is typically applied only to the output layer
            x = F.softmax(x, dim=1)
        
        x = self.fc2(x)
        
        if activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif activation == 'tanh':
            x = torch.tanh(x)
        elif activation == 'relu':
            x = F.relu(x)
        elif activation == 'leaky_relu':
            x = F.leaky_relu(x, negative_slope=0.01)
        
        x = self.fc3(x)
        
        # For classification, apply softmax to the output layer
        if activation != 'softmax':
            x = F.softmax(x, dim=1)
        
        return x

# Visualize different activation functions
def plot_activation_functions():
    x = torch.linspace(-5, 5, 100)
    
    # Sigmoid
    sigmoid = torch.sigmoid(x)
    
    # Tanh
    tanh = torch.tanh(x)
    
    # ReLU
    relu = F.relu(x)
    
    # Leaky ReLU
    leaky_relu = F.leaky_relu(x, negative_slope=0.01)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(x.numpy(), sigmoid.numpy())
    plt.title('Sigmoid')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(x.numpy(), tanh.numpy())
    plt.title('Tanh')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(x.numpy(), relu.numpy())
    plt.title('ReLU')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(x.numpy(), leaky_relu.numpy())
    plt.title('Leaky ReLU')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Plot activation functions
plot_activation_functions()

# Demonstrate the effect of different activation functions
input_size = 10
hidden_size = 20
output_size = 2
model = ActivationDemo(input_size, hidden_size, output_size)

# Test with random input
random_input = torch.randn(1, input_size)

print("Output with different activation functions:")
print(f"Sigmoid: {model(random_input, activation='sigmoid')}")
print(f"Tanh: {model(random_input, activation='tanh')}")
print(f"ReLU: {model(random_input, activation='relu')}")
print(f"Leaky ReLU: {model(random_input, activation='leaky_relu')}")
print(f"Softmax: {model(random_input, activation='softmax')}")

# Demonstrate the vanishing gradient problem with sigmoid
def demonstrate_vanishing_gradient():
    # Create a deep network with sigmoid activations
    class DeepSigmoidNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(DeepSigmoidNet, self).__init__()
            self.layers = nn.ModuleList()
            
            # Input layer
            self.layers.append(nn.Linear(input_size, hidden_size))
            
            # Hidden layers
            for _ in range(num_layers):
                self.layers.append(nn.Linear(hidden_size, hidden_size))
            
            # Output layer
            self.layers.append(nn.Linear(hidden_size, 1))
        
        def forward(self, x):
            for layer in self.layers[:-1]:
                x = torch.sigmoid(layer(x))
            x = self.layers[-1](x)
            return x
    
    # Create a deep network
    model = DeepSigmoidNet(input_size=1, hidden_size=10, num_layers=10)
    
    # Create a simple input and target
    x = torch.ones(1, 1)
    y = torch.ones(1, 1)
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Print gradients of the first layer
    first_layer_grad = model.layers[0].weight.grad.norm().item()
    print(f"\nGradient norm of first layer with sigmoid: {first_layer_grad:.6f}")
    
    # Create a deep network with ReLU activations
    class DeepReLUNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(DeepReLUNet, self).__init__()
            self.layers = nn.ModuleList()
            
            # Input layer
            self.layers.append(nn.Linear(input_size, hidden_size))
            
            # Hidden layers
            for _ in range(num_layers):
                self.layers.append(nn.Linear(hidden_size, hidden_size))
            
            # Output layer
            self.layers.append(nn.Linear(hidden_size, 1))
        
        def forward(self, x):
            for layer in self.layers[:-1]:
                x = F.relu(layer(x))
            x = self.layers[-1](x)
            return x
    
    # Create a deep network
    model = DeepReLUNet(input_size=1, hidden_size=10, num_layers=10)
    
    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Print gradients of the first layer
    first_layer_grad = model.layers[0].weight.grad.norm().item()
    print(f"Gradient norm of first layer with ReLU: {first_layer_grad:.6f}")

demonstrate_vanishing_gradient()
```

**When and where we use activation functions:**
- After each hidden layer in a neural network to introduce non-linearity
- In the output layer, with sigmoid for binary classification, softmax for multi-class classification, and linear activation for regression
- When we need to model complex non-linear relationships
- In both feedforward and recurrent neural networks
- In convolutional neural networks after convolution operations

**Real-world examples:**
- ReLU is commonly used in hidden layers of deep neural networks due to its computational efficiency and mitigation of vanishing gradients
- Sigmoid is used in binary classification problems and in the output layer of binary classifiers
- Softmax is used in the output layer for multi-class classification problems
- Tanh is often used in recurrent neural networks and when we need outputs in the range [-1, 1]

### 3. Loss Functions

**What are loss functions?**
Loss functions (also called cost functions or objective functions) measure how well a neural network is performing by calculating the difference between the predicted outputs and the actual target values. The goal of training is to minimize this loss. Common loss functions include:

- Mean Squared Error (MSE): Used for regression tasks
- Binary Cross-Entropy: Used for binary classification
- Categorical Cross-Entropy: Used for multi-class classification
- Hinge Loss: Used for maximum-margin classification, like SVMs

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Demonstrate different loss functions
def demonstrate_loss_functions():
    # Create sample data
    input_size = 10
    hidden_size = 20
    output_size = 2
    batch_size = 5
    
    # Create model
    model = SimpleNet(input_size, hidden_size, output_size)
    
    # Create random inputs and targets
    inputs = torch.randn(batch_size, input_size)
    
    # For regression (continuous targets)
    regression_targets = torch.randn(batch_size, 1)
    
    # For binary classification (binary targets)
    binary_targets = torch.randint(0, 2, (batch_size, 1)).float()
    
    # For multi-class classification (class indices)
    class_indices = torch.randint(0, output_size, (batch_size,))
    
    # For multi-class classification (one-hot encoded)
    one_hot_targets = F.one_hot(class_indices, num_classes=output_size).float()
    
    # Get model outputs
    outputs = model(inputs)
    
    # Mean Squared Error (for regression)
    mse_loss = nn.MSELoss()
    regression_loss = mse_loss(outputs[:, :1], regression_targets)
    print(f"MSE Loss (regression): {regression_loss.item():.4f}")
    
    # Binary Cross-Entropy (for binary classification)
    # Note: For binary cross-entropy, we need to apply sigmoid to the output
    binary_outputs = torch.sigmoid(outputs[:, :1])
    bce_loss = nn.BCELoss()
    binary_classification_loss = bce_loss(binary_outputs, binary_targets)
    print(f"Binary Cross-Entropy Loss: {binary_classification_loss.item():.4f}")
    
    # Categorical Cross-Entropy (for multi-class classification)
    # Note: CrossEntropyLoss in PyTorch combines LogSoftmax and NLLLoss
    ce_loss = nn.CrossEntropyLoss()
    multi_class_loss = ce_loss(outputs, class_indices)
    print(f"Categorical Cross-Entropy Loss: {multi_class_loss.item():.4f}")
    
    # Hinge Loss (for SVM-like classification)
    # We'll create a simple implementation for demonstration
    def hinge_loss(outputs, targets, margin=1.0):
        # Convert targets to -1 and 1
        targets = 2 * targets - 1
        # Calculate hinge loss
        loss = torch.mean(torch.clamp(1 - targets * outputs, min=0))
        return loss
    
    # For hinge loss, we'll use a binary scenario
    hinge = hinge_loss(outputs[:, :1], binary_targets)
    print(f"Hinge Loss: {hinge.item():.4f}")
    
    # Visualize different loss functions
    x = torch.linspace(-2, 2, 100)
    
    # MSE Loss: (y_true - y_pred)^2
    mse = x**2
    
    # Binary Cross-Entropy: -[y*log(y_pred) + (1-y)*log(1-y_pred)]
    # For y_true = 1: -log(y_pred)
    bce_y1 = -torch.log(torch.sigmoid(x) + 1e-10)  # Add small value to avoid log(0)
    # For y_true = 0: -log(1-y_pred)
    bce_y0 = -torch.log(1 - torch.sigmoid(x) + 1e-10)
    
    # Hinge Loss: max(0, 1 - y_true * y_pred)
    # For y_true = 1: max(0, 1 - y_pred)
    hinge_y1 = torch.clamp(1 - x, min=0)
    # For y_true = -1: max(0, 1 + y_pred)
    hinge_y0 = torch.clamp(1 + x, min=0)
    
    # Plot
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(x.numpy(), mse.numpy())
    plt.title('Mean Squared Error')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(x.numpy(), bce_y1.numpy(), label='y_true = 1')
    plt.plot(x.numpy(), bce_y0.numpy(), label='y_true = 0')
    plt.title('Binary Cross-Entropy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(x.numpy(), hinge_y1.numpy(), label='y_true = 1')
    plt.plot(x.numpy(), hinge_y0.numpy(), label='y_true = -1')
    plt.title('Hinge Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

demonstrate_loss_functions()

# Demonstrate the effect of different loss functions on training
def compare_loss_effects():
    # Create synthetic data for binary classification
    torch.manual_seed(42)
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,)).float()
    
    # Split data
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    # Define two identical models
    model_bce = SimpleNet(10, 20, 1)
    model_mse = SimpleNet(10, 20, 1)
    
    # Define optimizers
    optimizer_bce = optim.Adam(model_bce.parameters(), lr=0.01)
    optimizer_mse = optim.Adam(model_mse.parameters(), lr=0.01)
    
    # Define loss functions
    bce_loss = nn.BCEWithLogitsLoss()  # Combines sigmoid and BCE for numerical stability
    mse_loss = nn.MSELoss()
    
    # Training loop
    num_epochs = 100
    bce_losses = []
    mse_losses = []
    bce_accuracies = []
    mse_accuracies = []
    
    for epoch in range(num_epochs):
        # Train with BCE
        model_bce.train()
        optimizer_bce.zero_grad()
        outputs_bce = model_bce(X_train).squeeze()
        loss_bce = bce_loss(outputs_bce, y_train)
        loss_bce.backward()
        optimizer_bce.step()
        
        # Train with MSE
        model_mse.train()
        optimizer_mse.zero_grad()
        outputs_mse = model_mse(X_train).squeeze()
        loss_mse = mse_loss(torch.sigmoid(outputs_mse), y_train)  # Apply sigmoid for MSE
        loss_mse.backward()
        optimizer_mse.step()
        
        # Record losses
        bce_losses.append(loss_bce.item())
        mse_losses.append(loss_mse.item())
        
        # Evaluate
        model_bce.eval()
        model_mse.eval()
        
        with torch.no_grad():
            # BCE model
            test_outputs_bce = torch.sigmoid(model_bce(X_test).squeeze())
            test_preds_bce = (test_outputs_bce > 0.5).float()
            bce_acc = (test_preds_bce == y_test).float().mean().item()
            bce_accuracies.append(bce_acc)
            
            # MSE model
            test_outputs_mse = torch.sigmoid(model_mse(X_test).squeeze())
            test_preds_mse = (test_outputs_mse > 0.5).float()
            mse_acc = (test_preds_mse == y_test).float().mean().item()
            mse_accuracies.append(mse_acc)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(bce_losses, label='BCE Loss')
    plt.plot(mse_losses, label='MSE Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(bce_accuracies, label='BCE Accuracy')
    plt.plot(mse_accuracies, label='MSE Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Final BCE Accuracy: {bce_accuracies[-1]:.4f}")
    print(f"Final MSE Accuracy: {mse_accuracies[-1]:.4f}")

compare_loss_effects()
```

**When and where we use loss functions:**
- During the training phase to guide the optimization process
- To evaluate model performance on validation and test sets
- When comparing different models or hyperparameters
- In custom applications where standard loss functions need to be adapted
- In multi-task learning where different tasks may require different loss functions

**Real-world examples:**
- MSE is used in regression problems like house price prediction or temperature forecasting
- Binary cross-entropy is used in medical diagnosis (disease vs. no disease)
- Categorical cross-entropy is used in image classification (identifying objects in images)
- Custom loss functions are used in applications like object detection where both classification and localization need to be optimized

### 4. Optimization Algorithms

**What are optimization algorithms?**
Optimization algorithms are used to minimize the loss function by adjusting the weights and biases of the neural network. They determine how the model learns from the training data. Common optimization algorithms include:

- Stochastic Gradient Descent (SGD): Updates weights using the gradient of the loss with respect to the weights
- SGD with Momentum: Accelerates SGD by adding a fraction of the previous update vector
- Adam (Adaptive Moment Estimation): Combines ideas from RMSProp and Momentum, adapting learning rates for each parameter
- RMSProp: Adapts learning rates based on a moving average of recent gradients
- Adagrad: Adapts learning rates based on the historical sum of squares of gradients

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create a simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
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

# Define training function
def train_model(model, optimizer, criterion, X_train, y_train, X_test, y_test, num_epochs=100):
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())
            
            _, predicted = torch.max(test_outputs.data, 1)
            correct = (predicted == y_test).sum().item()
            accuracy = correct / y_test.size(0)
            test_accuracies.append(accuracy)
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Accuracy: {accuracy:.4f}')
    
    return train_losses, test_losses, test_accuracies

# Compare different optimizers
input_size = 20
hidden_size = 50
output_size = 2
num_epochs = 100

# SGD
model_sgd = SimpleNet(input_size, hidden_size, output_size)
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
print("Training with SGD:")
train_losses_sgd, test_losses_sgd, test_accuracies_sgd = train_model(
    model_sgd, optimizer_sgd, criterion, X_train, y_train, X_test, y_test, num_epochs
)

# SGD with Momentum
model_sgd_momentum = SimpleNet(input_size, hidden_size, output_size)
optimizer_sgd_momentum = optim.SGD(model_sgd_momentum.parameters(), lr=0.01, momentum=0.9)
print("\nTraining with SGD with Momentum:")
train_losses_sgd_momentum, test_losses_sgd_momentum, test_accuracies_sgd_momentum = train_model(
    model_sgd_momentum, optimizer_sgd_momentum, criterion, X_train, y_train, X_test, y_test, num_epochs
)

# Adam
model_adam = SimpleNet(input_size, hidden_size, output_size)
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
print("\nTraining with Adam:")
train_losses_adam, test_losses_adam, test_accuracies_adam = train_model(
    model_adam, optimizer_adam, criterion, X_train, y_train, X_test, y_test, num_epochs
)

# RMSprop
model_rmsprop = SimpleNet(input_size, hidden_size, output_size)
optimizer_rmsprop = optim.RMSprop(model_rmsprop.parameters(), lr=0.001)
print("\nTraining with RMSprop:")
train_losses_rmsprop, test_losses_rmsprop, test_accuracies_rmsprop = train_model(
    model_rmsprop, optimizer_rmsprop, criterion, X_train, y_train, X_test, y_test, num_epochs
)

# Plot results
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(train_losses_sgd, label='SGD')
plt.plot(train_losses_sgd_momentum, label='SGD with Momentum')
plt.plot(train_losses_adam, label='Adam')
plt.plot(train_losses_rmsprop, label='RMSprop')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(test_losses_sgd, label='SGD')
plt.plot(test_losses_sgd_momentum, label='SGD with Momentum')
plt.plot(test_losses_adam, label='Adam')
plt.plot(test_losses_rmsprop, label='RMSprop')
plt.title('Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(test_accuracies_sgd, label='SGD')
plt.plot(test_accuracies_sgd_momentum, label='SGD with Momentum')
plt.plot(test_accuracies_adam, label='Adam')
plt.plot(test_accuracies_rmsprop, label='RMSprop')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Visualize optimization paths
def visualize_optimization_paths():
    # Create a simple 2D loss landscape
    def loss_function(w1, w2):
        return 0.5 * (w1**2 + w2**2) + 0.25 * np.sin(5 * w1) * np.sin(5 * w2)
    
    # Create a grid of points
    w1 = np.linspace(-2, 2, 100)
    w2 = np.linspace(-2, 2, 100)
    W1, W2 = np.meshgrid(w1, w2)
    Z = loss_function(W1, W2)
    
    # Plot the loss landscape
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.contourf(W1, W2, Z, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title('Loss Landscape')
    plt.xlabel('w1')
    plt.ylabel('w2')
    
    # Simulate optimization paths
    def simulate_optimization(optimizer, start_point, num_steps=50):
        path = [start_point]
        current_point = start_point
        
        for _ in range(num_steps):
            # Compute gradient (negative gradient direction for minimization)
            w1, w2 = current_point
            grad_w1 = w1 + 1.25 * np.cos(5 * w1) * np.sin(5 * w2)
            grad_w2 = w2 + 1.25 * np.sin(5 * w1) * np.cos(5 * w2)
            
            # Update based on optimizer
            if optimizer == 'SGD':
                lr = 0.1
                next_point = [current_point[0] - lr * grad_w1, current_point[1] - lr * grad_w2]
            elif optimizer == 'SGD with Momentum':
                lr = 0.1
                momentum = 0.9
                if not hasattr(simulate_optimization, 'velocity'):
                    simulate_optimization.velocity = [0, 0]
                simulate_optimization.velocity = [
                    momentum * simulate_optimization.velocity[0] - lr * grad_w1,
                    momentum * simulate_optimization.velocity[1] - lr * grad_w2
                ]
                next_point = [
                    current_point[0] + simulate_optimization.velocity[0],
                    current_point[1] + simulate_optimization.velocity[1]
                ]
            elif optimizer == 'Adam':
                lr = 0.1
                beta1, beta2 = 0.9, 0.999
                epsilon = 1e-8
                
                if not hasattr(simulate_optimization, 'm'):
                    simulate_optimization.m = [0, 0]
                    simulate_optimization.v = [0, 0]
                    simulate_optimization.t = 0
                
                simulate_optimization.t += 1
                
                # Update biased first moment estimate
                simulate_optimization.m = [
                    beta1 * simulate_optimization.m[0] + (1 - beta1) * grad_w1,
                    beta1 * simulate_optimization.m[1] + (1 - beta1) * grad_w2
                ]
                
                # Update biased second raw moment estimate
                simulate_optimization.v = [
                    beta2 * simulate_optimization.v[0] + (1 - beta2) * grad_w1**2,
                    beta2 * simulate_optimization.v[1] + (1 - beta2) * grad_w2**2
                ]
                
                # Compute bias-corrected first moment estimate
                m_hat = [
                    simulate_optimization.m[0] / (1 - beta1**simulate_optimization.t),
                    simulate_optimization.m[1] / (1 - beta1**simulate_optimization.t)
                ]
                
                # Compute bias-corrected second raw moment estimate
                v_hat = [
                    simulate_optimization.v[0] / (1 - beta2**simulate_optimization.t),
                    simulate_optimization.v[1] / (1 - beta2**simulate_optimization.t)
                ]
                
                # Update parameters
                next_point = [
                    current_point[0] - lr * m_hat[0] / (np.sqrt(v_hat[0]) + epsilon),
                    current_point[1] - lr * m_hat[1] / (np.sqrt(v_hat[1]) + epsilon)
                ]
            
            # Check if we're still in bounds
            if abs(next_point[0]) < 2 and abs(next_point[1]) < 2:
                path.append(next_point)
                current_point = next_point
            else:
                break
        
        return np.array(path)
    
    # Simulate paths for different optimizers
    start_point = [1.5, 1.5]
    
    # SGD
    path_sgd = simulate_optimization('SGD', start_point)
    
    # Reset momentum for next simulation
    if hasattr(simulate_optimization, 'velocity'):
        del simulate_optimization.velocity
    
    # SGD with Momentum
    path_sgd_momentum = simulate_optimization('SGD with Momentum', start_point)
    
    # Reset Adam variables for next simulation
    if hasattr(simulate_optimization, 'm'):
        del simulate_optimization.m
        del simulate_optimization.v
        del simulate_optimization.t
    
    # Adam
    path_adam = simulate_optimization('Adam', start_point)
    
    # Plot paths
    plt.subplot(1, 3, 2)
    plt.contourf(W1, W2, Z, levels=50, cmap='viridis')
    plt.colorbar()
    plt.plot(path_sgd[:, 0], path_sgd[:, 1], 'r-', label='SGD', linewidth=2)
    plt.plot(path_sgd_momentum[:, 0], path_sgd_momentum[:, 1], 'g-', label='SGD with Momentum', linewidth=2)
    plt.plot(path_adam[:, 0], path_adam[:, 1], 'b-', label='Adam', linewidth=2)
    plt.plot(start_point[0], start_point[1], 'ko', markersize=8)
    plt.title('Optimization Paths')
    plt.xlabel('w1')
    plt.ylabel('w2')
    plt.legend()
    
    # Plot loss over optimization steps
    plt.subplot(1, 3, 3)
    loss_sgd = [loss_function(p[0], p[1]) for p in path_sgd]
    loss_sgd_momentum = [loss_function(p[0], p[1]) for p in path_sgd_momentum]
    loss_adam = [loss_function(p[0], p[1]) for p in path_adam]
    
    plt.plot(loss_sgd, 'r-', label='SGD', linewidth=2)
    plt.plot(loss_sgd_momentum, 'g-', label='SGD with Momentum', linewidth=2)
    plt.plot(loss_adam, 'b-', label='Adam', linewidth=2)
    plt.title('Loss over Optimization Steps')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

visualize_optimization_paths()
```

**When and where we use optimization algorithms:**
- During the training phase of neural networks to update weights and biases
- When we need to find the minimum of a loss function
- In transfer learning when fine-tuning pre-trained models
- In reinforcement learning for policy optimization
- In generative models like GANs and VAEs

**Real-world examples:**
- Adam is widely used as the default optimizer for many deep learning tasks due to its adaptive learning rates and good performance
- SGD with momentum is often used in computer vision tasks, especially when training CNNs
- Adagrad is used in natural language processing for sparse features like word embeddings
- RMSprop is used in recurrent neural networks for sequence modeling

### 5. Regularization Techniques

**What are regularization techniques?**
Regularization techniques are used to prevent overfitting in neural networks by adding constraints or penalties to the model. They help improve generalization by discouraging complex models that fit the training data too closely. Common regularization techniques include:

- L1 and L2 Regularization: Adding penalty terms to the loss function based on the weights
- Dropout: Randomly setting a fraction of input units to 0 during training
- Early Stopping: Monitoring validation loss and stopping training when it starts to increase
- Batch Normalization: Normalizing the inputs of each layer to reduce internal covariate shift
- Data Augmentation: Artificially increasing the size and diversity of the training data

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create a neural network with regularization options
class RegularizedNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5, use_batch_norm=False):
        super(RegularizedNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.use_batch_norm = use_batch_norm
        
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        if self.use_batch_norm:
            self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.fc1(x)
        
        if self.use_batch_norm:
            x = self.bn1(x)
        
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        if self.use_batch_norm:
            x = self.bn2(x)
        
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x

# Create synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
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

# Define training function with L2 regularization
def train_model_l2(model, optimizer, criterion, X_train, y_train, X_test, y_test, weight_decay=0.0, num_epochs=100):
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Add L2 regularization (weight decay)
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param, 2)
        loss += weight_decay * l2_reg
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())
            
            _, predicted = torch.max(test_outputs.data, 1)
            correct = (predicted == y_test).sum().item()
            accuracy = correct / y_test.size(0)
            test_accuracies.append(accuracy)
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Accuracy: {accuracy:.4f}')
    
    return train_losses, test_losses, test_accuracies

# Compare different regularization techniques
input_size = 20
hidden_size = 100
output_size = 2
num_epochs = 100

# Baseline model without regularization
model_baseline = RegularizedNet(input_size, hidden_size, output_size, dropout_rate=0.0, use_batch_norm=False)
optimizer_baseline = optim.Adam(model_baseline.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
print("Training baseline model (no regularization):")
train_losses_baseline, test_losses_baseline, test_accuracies_baseline = train_model_l2(
    model_baseline, optimizer_baseline, criterion, X_train, y_train, X_test, y_test, weight_decay=0.0, num_epochs=num_epochs
)

# Model with L2 regularization
model_l2 = RegularizedNet(input_size, hidden_size, output_size, dropout_rate=0.0, use_batch_norm=False)
optimizer_l2 = optim.Adam(model_l2.parameters(), lr=0.001, weight_decay=0.01)  # weight_decay is L2 regularization
print("\nTraining model with L2 regularization:")
train_losses_l2, test_losses_l2, test_accuracies_l2 = train_model_l2(
    model_l2, optimizer_l2, criterion, X_train, y_train, X_test, y_test, weight_decay=0.0, num_epochs=num_epochs
)

# Model with Dropout
model_dropout = RegularizedNet(input_size, hidden_size, output_size, dropout_rate=0.5, use_batch_norm=False)
optimizer_dropout = optim.Adam(model_dropout.parameters(), lr=0.001)
print("\nTraining model with Dropout:")
train_losses_dropout, test_losses_dropout, test_accuracies_dropout = train_model_l2(
    model_dropout, optimizer_dropout, criterion, X_train, y_train, X_test, y_test, weight_decay=0.0, num_epochs=num_epochs
)

# Model with Batch Normalization
model_bn = RegularizedNet(input_size, hidden_size, output_size, dropout_rate=0.0, use_batch_norm=True)
optimizer_bn = optim.Adam(model_bn.parameters(), lr=0.001)
print("\nTraining model with Batch Normalization:")
train_losses_bn, test_losses_bn, test_accuracies_bn = train_model_l2(
    model_bn, optimizer_bn, criterion, X_train, y_train, X_test, y_test, weight_decay=0.0, num_epochs=num_epochs
)

# Model with combined regularization
model_combined = RegularizedNet(input_size, hidden_size, output_size, dropout_rate=0.3, use_batch_norm=True)
optimizer_combined = optim.Adam(model_combined.parameters(), lr=0.001, weight_decay=0.001)
print("\nTraining model with combined regularization:")
train_losses_combined, test_losses_combined, test_accuracies_combined = train_model_l2(
    model_combined, optimizer_combined, criterion, X_train, y_train, X_test, y_test, weight_decay=0.0, num_epochs=num_epochs
)

# Plot results
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(train_losses_baseline, label='Baseline')
plt.plot(train_losses_l2, label='L2')
plt.plot(train_losses_dropout, label='Dropout')
plt.plot(train_losses_bn, label='Batch Norm')
plt.plot(train_losses_combined, label='Combined')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(test_losses_baseline, label='Baseline')
plt.plot(test_losses_l2, label='L2')
plt.plot(test_losses_dropout, label='Dropout')
plt.plot(test_losses_bn, label='Batch Norm')
plt.plot(test_losses_combined, label='Combined')
plt.title('Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(test_accuracies_baseline, label='Baseline')
plt.plot(test_accuracies_l2, label='L2')
plt.plot(test_accuracies_dropout, label='Dropout')
plt.plot(test_accuracies_bn, label='Batch Norm')
plt.plot(test_accuracies_combined, label='Combined')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Demonstrate early stopping
def demonstrate_early_stopping():
    # Create a more complex model that is prone to overfitting
    class ComplexNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(ComplexNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size)
            self.fc4 = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return x
    
    # Create a smaller dataset to encourage overfitting
    X_small, y_small = make_classification(n_samples=200, n_features=20, n_classes=2, random_state=42)
    X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(X_small, y_small, test_size=0.3, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_small = scaler.fit_transform(X_train_small)
    X_test_small = scaler.transform(X_test_small)
    
    # Convert to PyTorch tensors
    X_train_small = torch.FloatTensor(X_train_small)
    y_train_small = torch.LongTensor(y_train_small)
    X_test_small = torch.FloatTensor(X_test_small)
    y_test_small = torch.LongTensor(y_test_small)
    
    # Train without early stopping
    model_no_es = ComplexNet(20, 100, 2)
    optimizer_no_es = optim.Adam(model_no_es.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    train_losses_no_es = []
    test_losses_no_es = []
    test_accuracies_no_es = []
    
    for epoch in range(200):
        # Training
        model_no_es.train()
        optimizer_no_es.zero_grad()
        outputs = model_no_es(X_train_small)
        loss = criterion(outputs, y_train_small)
        loss.backward()
        optimizer_no_es.step()
        
        train_losses_no_es.append(loss.item())
        
        # Evaluation
        model_no_es.eval()
        with torch.no_grad():
            test_outputs = model_no_es(X_test_small)
            test_loss = criterion(test_outputs, y_test_small)
            test_losses_no_es.append(test_loss.item())
            
            _, predicted = torch.max(test_outputs.data, 1)
            correct = (predicted == y_test_small).sum().item()
            accuracy = correct / y_test_small.size(0)
            test_accuracies_no_es.append(accuracy)
    
    # Train with early stopping
    model_es = ComplexNet(20, 100, 2)
    optimizer_es = optim.Adam(model_es.parameters(), lr=0.01)
    
    train_losses_es = []
    test_losses_es = []
    test_accuracies_es = []
    
    best_test_loss = float('inf')
    patience = 10
    counter = 0
    best_model_state = None
    
    for epoch in range(200):
        # Training
        model_es.train()
        optimizer_es.zero_grad()
        outputs = model_es(X_train_small)
        loss = criterion(outputs, y_train_small)
        loss.backward()
        optimizer_es.step()
        
        train_losses_es.append(loss.item())
        
        # Evaluation
        model_es.eval()
        with torch.no_grad():
            test_outputs = model_es(X_test_small)
            test_loss = criterion(test_outputs, y_test_small)
            test_losses_es.append(test_loss.item())
            
            _, predicted = torch.max(test_outputs.data, 1)
            correct = (predicted == y_test_small).sum().item()
            accuracy = correct / y_test_small.size(0)
            test_accuracies_es.append(accuracy)
        
        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            counter = 0
            best_model_state = model_es.state_dict()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model
    model_es.load_state_dict(best_model_state)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses_no_es, label='No Early Stopping')
    plt.plot(train_losses_es, label='Early Stopping')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(test_losses_no_es, label='No Early Stopping')
    plt.plot(test_losses_es, label='Early Stopping')
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(test_accuracies_no_es, label='No Early Stopping')
    plt.plot(test_accuracies_es, label='Early Stopping')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Final test accuracy without early stopping: {test_accuracies_no_es[-1]:.4f}")
    print(f"Final test accuracy with early stopping: {test_accuracies_es[-1]:.4f}")

demonstrate_early_stopping()
```

**When and where we use regularization techniques:**
- When we have limited training data and want to prevent overfitting
- When using complex models with many parameters
- In deep neural networks where overfitting is a common problem
- When we need to improve the generalization of our models
- In transfer learning when fine-tuning pre-trained models on small datasets

**Real-world examples:**
- Dropout is widely used in computer vision models like CNNs to prevent overfitting
- Batch normalization is used in deep networks to stabilize and accelerate training
- L2 regularization is used in natural language processing models to prevent large weights
- Early stopping is used in virtually all deep learning applications to determine when to stop training
- Data augmentation is used in image recognition to artificially increase the size of the training dataset

## Applications in Deep Learning

### 1. Customer Churn Prediction

Let's implement a customer churn prediction system using ANNs:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Create a synthetic customer churn dataset
np.random.seed(42)
n_samples = 10000

# Generate customer data
age = np.random.randint(18, 80, n_samples)
gender = np.random.choice(['Male', 'Female'], n_samples)
tenure = np.random.randint(1, 72, n_samples)  # Months as customer
monthly_charges = np.random.uniform(20, 200, n_samples)
total_charges = tenure * monthly_charges * (0.9 + 0.2 * np.random.random(n_samples))
contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2])
payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples)
internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.3, 0.5, 0.2])
online_security = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2])
tech_support = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2])

# Generate churn labels based on features
churn_prob = (
    0.1 * (tenure < 12) +  # Higher churn for new customers
    0.15 * (contract_type == 'Month-to-month') +  # Higher churn for month-to-month contracts
    0.1 * (payment_method == 'Electronic check') +  # Higher churn for electronic check
    0.1 * (internet_service == 'Fiber optic') +  # Higher churn for fiber optic
    0.05 * (online_security == 'No') +  # Higher churn without online security
    0.05 * (tech_support == 'No')  # Higher churn without tech support
)

churn_prob = np.clip(churn_prob, 0.05, 0.8)  # Clip to reasonable range
churn = np.random.binomial(1, churn_prob)

# Create DataFrame
data = pd.DataFrame({
    'age': age,
    'gender': gender,
    'tenure': tenure,
    'monthly_charges': monthly_charges,
    'total_charges': total_charges,
    'contract_type': contract_type,
    'payment_method': payment_method,
    'internet_service': internet_service,
    'online_security': online_security,
    'tech_support': tech_support,
    'churn': churn
})

# Preprocess data
X = data.drop('churn', axis=1)
y = data['churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Create preprocessing pipeline
numerical_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Fit and transform the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_processed)
y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_processed)
y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)

# Define the neural network model
class ChurnPredictionNet(nn.Module):
    def __init__(self, input_size):
        super(ChurnPredictionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = torch.sigmoid(self.fc4(x))
        return x

# Initialize model, loss function, and optimizer
model = ChurnPredictionNet(input_size=X_train_processed.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 regularization

# Training loop
num_epochs = 100
train_losses = []
test_losses = []
test_accuracies = []

best_test_loss = float('inf')
patience = 10
counter = 0
best_model_state = None

for epoch in range(num_epochs):
    # Training
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        test_losses.append(test_loss.item())
        
        test_preds = (test_outputs > 0.5).float()
        accuracy = (test_preds == y_test_tensor).float().mean().item()
        test_accuracies.append(accuracy)
    
    # Early stopping
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        counter = 0
        best_model_state = model.state_dict()
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Accuracy: {accuracy:.4f}')

# Load the best model
model.load_state_dict(best_model_state)

# Plot training and validation loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_preds = (test_outputs > 0.5).float()
    
    # Calculate metrics
    accuracy = (test_preds == y_test_tensor).float().mean().item()
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test_tensor, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test_tensor, test_preds))

# Feature importance analysis using permutation importance
def permutation_importance(model, X, y, preprocessor, n_repeats=10):
    model.eval()
    baseline_accuracy = accuracy_score(y, (model(X) > 0.5).float().numpy())
    importances = []
    
    # Get feature names from preprocessor
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(columns)
        elif name == 'cat':
            # Get one-hot encoded feature names
            encoder = transformer.named_steps['onehot']
            for col in columns:
                categories = encoder.categories_[list(columns).index(col)]
                feature_names.extend([f"{col}_{cat}" for cat in categories])
    
    for i in range(X.shape[1]):
        X_permuted = X.clone()
        accuracies = []
        
        for _ in range(n_repeats):
            # Permute feature i
            X_permuted[:, i] = X_permuted[torch.randperm(X_permuted.size(0)), i]
            
            # Calculate accuracy with permuted feature
            with torch.no_grad():
                preds = (model(X_permuted) > 0.5).float()
                acc = accuracy_score(y, preds)
                accuracies.append(acc)
        
        # Importance is decrease in accuracy
        importance = baseline_accuracy - np.mean(accuracies)
        importances.append(importance)
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    sorted_importances = [importances[i] for i in indices]
    sorted_feature_names = [feature_names[i] for i in indices]
    
    return sorted_importances, sorted_feature_names

# Calculate feature importance
importances, feature_names = permutation_importance(model, X_test_tensor, y_test_tensor, preprocessor)

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.barh(range(len(importances[:15])), importances[:15])  # Show top 15 features
plt.yticks(range(len(importances[:15])), feature_names[:15])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance (Top 15)')
plt.tight_layout()
plt.show()
```

### 2. Image Classification with ANN

Let's implement an image classification system using ANNs:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Define the neural network for image classification
class ImageClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(ImageClassifier, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        # Flatten the image
        x = x.view(x.size(0), -1)
        return self.model(x)

# Load and preprocess the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
input_size = 28 * 28  # MNIST images are 28x28 pixels
hidden_sizes = [512, 256, 128]
num_classes = 10  # Digits 0-9

model = ImageClassifier(input_size, hidden_sizes, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # Evaluation
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')

# Plot training and validation metrics
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Visualize some predictions
def visualize_predictions(model, data_loader, num_samples=10):
    model.eval()
    
    # Get a batch of test data
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Plot images with predictions
    plt.figure(figsize=(15, 7))
    for i in range(min(num_samples, len(images))):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"True: {labels[i].item()}\nPred: {predicted[i].item()}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize predictions
visualize_predictions(model, test_loader)

# Visualize the learned weights of the first layer
def visualize_first_layer_weights(model):
    # Get the weights of the first linear layer
    weights = model.model[0].weight.data.numpy()
    
    # Reshape weights to visualize them as images
    # We'll visualize the first 20 neurons
    num_neurons = min(20, weights.shape[0])
    
    plt.figure(figsize=(15, 8))
    for i in range(num_neurons):
        plt.subplot(4, 5, i+1)
        # Reshape weights to 28x28 image
        plt.imshow(weights[i].reshape(28, 28), cmap='viridis')
        plt.title(f"Neuron {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle('Weights of First Layer Neurons', y=1.05)
    plt.show()

# Visualize first layer weights
visualize_first_layer_weights(model)

# Analyze misclassified examples
def analyze_misclassifications(model, data_loader):
    model.eval()
    
    misclassified = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Find misclassified examples
            misclassified_mask = predicted != labels
            
            if misclassified_mask.any():
                misclassified_images = images[misclassified_mask]
                misclassified_labels = labels[misclassified_mask]
                misclassified_preds = predicted[misclassified_mask]
                
                for i in range(len(misclassified_images)):
                    misclassified.append({
                        'image': misclassified_images[i],
                        'true_label': misclassified_labels[i].item(),
                        'predicted_label': misclassified_preds[i].item()
                    })
    
    return misclassified

# Analyze misclassifications
misclassified = analyze_misclassifications(model, test_loader)
print(f"Number of misclassified examples: {len(misclassified)}")

# Visualize some misclassified examples
plt.figure(figsize=(15, 8))
for i in range(min(10, len(misclassified))):
    plt.subplot(2, 5, i+1)
    plt.imshow(misclassified[i]['image'].squeeze(), cmap='gray')
    plt.title(f"True: {misclassified[i]['true_label']}\nPred: {misclassified[i]['predicted_label']}")
    plt.axis('off')

plt.tight_layout()
plt.suptitle('Misclassified Examples', y=1.05)
plt.show()
```

## Common Problems and Advancements

### 1. Problem: Vanishing and Exploding Gradients

**Issue:**
In deep neural networks, gradients can become extremely small (vanish) or extremely large (explode) during backpropagation. Vanishing gradients make it difficult for the network to learn early layers, while exploding gradients can cause the training to diverge.

**How advancements solved this:**
- **Activation functions**: ReLU and its variants mitigate the vanishing gradient problem
- **Weight initialization**: Techniques like Xavier/Glorot and He initialization help maintain gradient scale
- **Batch normalization**: Normalizes layer inputs to reduce internal covariate shift
- **Gradient clipping**: Prevents gradients from becoming too large
- **Residual connections**: Allow gradients to flow directly through skip connections

```python
# Problem code: Vanishing gradients in deep networks
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Create a deep network with sigmoid activations (prone to vanishing gradients)
class DeepSigmoidNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(DeepSigmoidNet, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        
        # Hidden layers
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_size, 1))
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.sigmoid(layer(x))
        x = self.layers[-1](x)
        return x

# Create a simple dataset
np.random.seed(42)
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100).reshape(-1, 1)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Create a deep network with sigmoid activations
model = DeepSigmoidNet(input_size=10, hidden_size=10, num_layers=10)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Function to track gradients
def track_gradients(model):
    gradients = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.grad is not None:
            gradients.append(param.grad.norm().item())
    return gradients

# Training loop
num_epochs = 100
all_gradients = []

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Track gradients
    gradients = track_gradients(model)
    all_gradients.append(gradients)
    
    # Update weights
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot gradients
plt.figure(figsize=(10, 6))
for i in range(0, len(all_gradients[0]), 2):  # Plot every other layer to avoid clutter
    layer_gradients = [grads[i] for grads in all_gradients]
    plt.plot(layer_gradients, label=f'Layer {i//2+1}')
plt.xlabel('Epoch')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norms Over Training (Sigmoid Network)')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()

# Enhancement code: Network with ReLU and batch normalization
class DeepReLUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(DeepReLUNet, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.batch_norms.append(nn.BatchNorm1d(hidden_size))
        
        # Hidden layers
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_size, 1))
    
    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            x = self.batch_norms[i](x)
            x = torch.relu(x)
        x = self.layers[-1](x)
        return x

# Create a deep network with ReLU and batch normalization
model = DeepReLUNet(input_size=10, hidden_size=10, num_layers=10)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
all_gradients = []

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Track gradients
    gradients = track_gradients(model)
    all_gradients.append(gradients)
    
    # Update weights
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot gradients
plt.figure(figsize=(10, 6))
for i in range(0, len(all_gradients[0]), 2):  # Plot every other layer to avoid clutter
    layer_gradients = [grads[i] for grads in all_gradients]
    plt.plot(layer_gradients, label=f'Layer {i//2+1}')
plt.xlabel('Epoch')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norms Over Training (ReLU + BatchNorm Network)')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()

# Enhancement code: Network with residual connections
class DeepResNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(DeepResNet, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Hidden layers
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Input layer
        x = torch.relu(self.input_layer(x))
        
        # Hidden layers with residual connections
        for i, layer in enumerate(self.hidden_layers):
            residual = x
            x = layer(x)
            x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = x + residual  # Residual connection
        
        # Output layer
        x = self.output_layer(x)
        return x

# Create a deep network with residual connections
model = DeepResNet(input_size=10, hidden_size=10, num_layers=10)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
all_gradients = []

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Track gradients
    gradients = track_gradients(model)
    all_gradients.append(gradients)
    
    # Update weights
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot gradients
plt.figure(figsize=(10, 6))
for i in range(0, len(all_gradients[0]), 2):  # Plot every other layer to avoid clutter
    layer_gradients = [grads[i] for grads in all_gradients]
    plt.plot(layer_gradients, label=f'Layer {i//2+1}')
plt.xlabel('Epoch')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norms Over Training (Residual Network)')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()
```

### 2. Problem: Overfitting

**Issue:**
Neural networks with many parameters can easily overfit to the training data, memorizing it instead of learning general patterns. This leads to poor performance on new, unseen data.

**How advancements solved this:**
- **Regularization techniques**: L1/L2 regularization, dropout, and early stopping
- **Data augmentation**: Artificially increasing the size and diversity of the training data
- **Batch normalization**: Reducing internal covariate shift and acting as a regularizer
- **Ensemble methods**: Combining multiple models to reduce variance
- **Transfer learning**: Leveraging pre-trained models on large datasets

```python
# Problem code: Overfitting in neural networks
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create a small dataset to encourage overfitting
np.random.seed(42)
X = np.random.randn(200, 10)
y = np.random.randint(0, 2, 200)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).view(-1, 1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).view(-1, 1)

# Define a large model that can easily overfit
class OverfittingNet(nn.Module):
    def __init__(self, input_size):
        super(OverfittingNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x

# Initialize model, loss function, and optimizer
model = OverfittingNet(input_size=10)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 200
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    # Training
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    train_preds = (outputs > 0.5).float()
    train_accuracy = (train_preds == y_train).float().mean().item()
    train_accuracies.append(train_accuracy)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses.append(test_loss.item())
        
        test_preds = (test_outputs > 0.5).float()
        test_accuracy = (test_preds == y_test).float().mean().item()
        test_accuracies.append(test_accuracy)
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}')

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Enhancement code: Model with regularization techniques
class RegularizedNet(nn.Module):
    def __init__(self, input_size):
        super(RegularizedNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize model, loss function, and optimizer
model = RegularizedNet(input_size=10)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 regularization

# Training loop with early stopping
num_epochs = 200
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

best_test_loss = float('inf')
patience = 20
counter = 0
best_model_state = None

for epoch in range(num_epochs):
    # Training
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    train_preds = (outputs > 0.5).float()
    train_accuracy = (train_preds == y_train).float().mean().item()
    train_accuracies.append(train_accuracy)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses.append(test_loss.item())
        
        test_preds = (test_outputs > 0.5).float()
        test_accuracy = (test_preds == y_test).float().mean().item()
        test_accuracies.append(test_accuracy)
    
    # Early stopping
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        counter = 0
        best_model_state = model.state_dict()
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}')

# Load the best model
model.load_state_dict(best_model_state)

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss (Regularized)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy (Regularized)')
plt.legend()

plt.tight_layout()
plt.show()

# Enhancement code: Data augmentation for tabular data
def augment_tabular_data(X, y, augmentation_factor=2):
    augmented_X = [X]
    augmented_y = [y]
    
    for _ in range(augmentation_factor - 1):
        # Add small random noise to create new samples
        noise = np.random.normal(0, 0.1, X.shape)
        X_augmented = X + noise
        augmented_X.append(X_augmented)
        augmented_y.append(y)
    
    return np.vstack(augmented_X), np.hstack(augmented_y)

# Apply data augmentation
X_train_augmented, y_train_augmented = augment_tabular_data(X_train.numpy(), y_train.numpy().flatten(), augmentation_factor=3)

# Convert to PyTorch tensors
X_train_augmented = torch.FloatTensor(X_train_augmented)
y_train_augmented = torch.FloatTensor(y_train_augmented).view(-1, 1)

# Initialize a new model
model = RegularizedNet(input_size=10)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Training loop with early stopping
num_epochs = 200
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

best_test_loss = float('inf')
patience = 20
counter = 0
best_model_state = None

for epoch in range(num_epochs):
    # Training
    model.train()
    outputs = model(X_train_augmented)
    loss = criterion(outputs, y_train_augmented)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    train_preds = (outputs > 0.5).float()
    train_accuracy = (train_preds == y_train_augmented).float().mean().item()
    train_accuracies.append(train_accuracy)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses.append(test_loss.item())
        
        test_preds = (test_outputs > 0.5).float()
        test_accuracy = (test_preds == y_test).float().mean().item()
        test_accuracies.append(test_accuracy)
    
    # Early stopping
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        counter = 0
        best_model_state = model.state_dict()
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}')

# Load the best model
model.load_state_dict(best_model_state)

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss (With Data Augmentation)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy (With Data Augmentation)')
plt.legend()

plt.tight_layout()
plt.show()
```

### 3. Problem: Computational Complexity

**Issue:**
Training deep neural networks can be computationally expensive, requiring significant time and resources. This limits the size of models we can train and the speed at which we can experiment.

**How advancements solved this:**
- **GPU acceleration**: Leveraging parallel processing capabilities of GPUs
- **Distributed training**: Spreading training across multiple devices or machines
- **Model optimization techniques**: Quantization, pruning, and knowledge distillation
- **Efficient architectures**: Designing networks that achieve good performance with fewer parameters
- **Mixed precision training**: Using lower precision for most operations while maintaining accuracy

```python
# Problem code: Training without optimization
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Create a larger dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# Define a larger model
class LargeModel(nn.Module):
    def __init__(self):
        super(LargeModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Initialize model, loss function, and optimizer
model = LargeModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create data loaders with small batch size
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop without optimization
num_epochs = 5
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%')

training_time = time.time() - start_time
print(f'Training time without optimization: {training_time:.2f} seconds')

# Enhancement code: Training with optimization techniques
# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a more efficient model
class EfficientModel(nn.Module):
    def __init__(self):
        super(EfficientModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize model, loss function, and optimizer
model = EfficientModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create data loaders with larger batch size and multiple workers
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

# Use mixed precision training if available
use_amp = device.type == 'cuda'
scaler = torch.cuda.amp.GradScaler() if use_amp else None

# Training loop with optimization
num_epochs = 5
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Use mixed precision if available
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        
        if (i+1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%')

training_time = time.time() - start_time
print(f'Training time with optimization: {training_time:.2f} seconds')

# Enhancement code: Model quantization for inference
print("\nApplying quantization for faster inference...")
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Evaluate quantized model
quantized_model.eval()
correct = 0
total = 0

start_time = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = quantized_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

inference_time = time.time() - start_time
accuracy = 100 * correct / total
print(f'Quantized Model Test Accuracy: {accuracy:.2f}%')
print(f'Inference time: {inference_time:.4f} seconds')

# Compare model sizes
original_size = sum(p.numel() * p.element_size() for p in model.parameters())
quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())

print(f"\nOriginal model size: {original_size / 1024 / 1024:.2f} MB")
print(f"Quantized model size: {quantized_size / 1024 / 1024:.2f} MB")
print(f"Size reduction: {(1 - quantized_size / original_size) * 100:.2f}%")
```

## Project: Customer Churn Predictor

Now, let's implement a customer churn prediction system using ANNs:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load the Telco Customer Churn dataset
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
data = pd.read_csv(url)

# Data preprocessing
# Handle missing values
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.dropna()

# Convert target variable to binary
data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Drop customer ID column
data = data.drop('customerID', axis=1)

# Split features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Create preprocessing pipeline
numerical_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Fit and transform the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_processed)
y_train_tensor = torch.FloatTensor(y_train.values)
X_test_tensor = torch.FloatTensor(X_test_processed)
y_test_tensor = torch.FloatTensor(y_test.values)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the neural network model
class ChurnPredictor(nn.Module):
    def __init__(self, input_size):
        super(ChurnPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = torch.sigmoid(self.fc4(x))
        return x

# Initialize model, loss function, and optimizer
model = ChurnPredictor(input_size=X_train_processed.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 regularization

# Training loop
num_epochs = 50
train_losses = []
test_losses = []
test_accuracies = []
test_aucs = []

best_test_loss = float('inf')
patience = 10
counter = 0
best_model_state = None

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # Evaluation
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            test_loss += loss.item()
            
            probs = outputs.squeeze().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())
    
    test_loss = test_loss / len(test_loader)
    test_losses.append(test_loss)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    test_accuracies.append(accuracy)
    test_aucs.append(auc)
    
    # Early stopping
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        counter = 0
        best_model_state = model.state_dict()
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}')

# Load the best model
model.load_state_dict(best_model_state)

# Plot training and validation metrics
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(test_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(test_aucs, label='Validation AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('Validation AUC')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model
model.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        probs = outputs.squeeze().numpy()
        preds = (probs > 0.5).astype(int)
        
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(batch_y.numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test AUC: {auc:.4f}")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

# Feature importance analysis using permutation importance
def permutation_importance(model, X, y, preprocessor, n_repeats=10):
    model.eval()
    baseline_auc = roc_auc_score(y, model(X).squeeze().numpy())
    importances = []
    
    # Get feature names from preprocessor
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(columns)
        elif name == 'cat':
            # Get one-hot encoded feature names
            encoder = transformer.named_steps['onehot']
            for col in columns:
                categories = encoder.categories_[list(columns).index(col)]
                feature_names.extend([f"{col}_{cat}" for cat in categories])
    
    for i in range(X.shape[1]):
        X_permuted = X.clone()
        aucs = []
        
        for _ in range(n_repeats):
            # Permute feature i
            X_permuted[:, i] = X_permuted[torch.randperm(X_permuted.size(0)), i]
            
            # Calculate AUC with permuted feature
            with torch.no_grad():
                probs = model(X_permuted).squeeze().numpy()
                auc = roc_auc_score(y, probs)
                aucs.append(auc)
        
        # Importance is decrease in AUC
        importance = baseline_auc - np.mean(aucs)
        importances.append(importance)
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    sorted_importances = [importances[i] for i in indices]
    sorted_feature_names = [feature_names[i] for i in indices]
    
    return sorted_importances, sorted_feature_names

# Calculate feature importance
importances, feature_names = permutation_importance(model, X_test_tensor, y_test_tensor, preprocessor)

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.barh(range(len(importances[:15])), importances[:15])  # Show top 15 features
plt.yticks(range(len(importances[:15])), feature_names[:15])
plt.xlabel('Importance (AUC decrease)')
plt.ylabel('Feature')
plt.title('Feature Importance (Top 15)')
plt.tight_layout()
plt.show()

# Create a function to predict churn for new customers
def predict_churn(customer_data, model, preprocessor):
    """
    Predict churn probability for new customer data.
    
    Args:
        customer_data (DataFrame): Customer data with the same features as training data
        model: Trained churn prediction model
        preprocessor: Fitted preprocessor
    
    Returns:
        float: Churn probability
    """
    # Preprocess the data
    processed_data = preprocessor.transform(customer_data)
    
    # Convert to PyTorch tensor
    tensor_data = torch.FloatTensor(processed_data)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        churn_prob = model(tensor_data).item()
    
    return churn_prob

# Example usage
example_customer = X_test.iloc[[0]]  # Take the first customer from the test set
churn_prob = predict_churn(example_customer, model, preprocessor)
print(f"\nExample customer churn probability: {churn_prob:.4f}")
print(f"Actual churn status: {y_test.iloc[0]}")
```

This customer churn predictor demonstrates how ANNs can be applied to real-world business problems. The model uses various techniques like batch normalization, dropout, and early stopping to prevent overfitting and improve generalization. It also includes feature importance analysis to understand which factors are most influential in predicting customer churn.

## Summary

Artificial Neural Networks (ANNs) are the foundation of deep learning, providing a powerful framework for modeling complex non-linear relationships in data. In this lesson, we've covered:

1. **Neurons and Layers**: The basic building blocks of neural networks, including input, hidden, and output layers
2. **Activation Functions**: Functions that introduce non-linearity into neural networks, such as ReLU, sigmoid, and tanh
3. **Loss Functions**: Measures of how well a neural network is performing, including MSE, cross-entropy, and hinge loss
4. **Optimization Algorithms**: Methods for minimizing the loss function, such as SGD, Adam, and RMSprop
5. **Regularization Techniques**: Methods to prevent overfitting, including dropout, batch normalization, and early stopping

We've also explored how these concepts are applied in real-world scenarios, common problems encountered, and how advancements in the field have addressed these challenges.

The customer churn predictor project provides a practical application of ANNs, demonstrating how to preprocess data, train a model, evaluate its performance, and analyze feature importance.

Understanding ANNs is crucial for anyone looking to master deep learning, as they form the foundation for more advanced architectures like CNNs, RNNs, and Transformers.