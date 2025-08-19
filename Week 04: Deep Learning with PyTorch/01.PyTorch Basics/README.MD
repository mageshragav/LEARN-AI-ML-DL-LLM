# Day 16 – PyTorch Basics

## What is PyTorch and Why is it Important for Deep Learning?

PyTorch is an open-source machine learning library developed by Facebook's AI Research lab. It provides a flexible and efficient framework for building and training deep learning models. PyTorch is known for its dynamic computation graph, which allows for more intuitive model development and debugging compared to static graph frameworks.

PyTorch is important for deep learning because it:
- Provides a simple and intuitive Python API that feels natural to Python developers
- Offers dynamic computation graphs that allow for on-the-fly model modifications
- Includes extensive support for GPU acceleration, enabling faster training of large models
- Has a rich ecosystem of tools and libraries for computer vision, NLP, and other domains
- Facilitates easy debugging with its Python-first approach
- Provides seamless deployment options from research to production

## Essential Subtopics

### 1. Tensors in PyTorch

**What are Tensors?**
Tensors are the fundamental data structure in PyTorch, similar to NumPy arrays but with additional capabilities like GPU acceleration. They are multi-dimensional arrays that can represent scalars (0D), vectors (1D), matrices (2D), and higher-dimensional data. Tensors are the building blocks for creating and manipulating data in deep learning models.

**PyTorch Implementation:**
```python
import torch
import numpy as np

# Creating tensors from data
# From Python lists
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"Tensor from list:\n{x_data}\n")

# From NumPy arrays
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"Tensor from NumPy array:\n{x_np}\n")

# Creating tensors with specific properties
# Uninitialized tensor
x_empty = torch.empty(2, 3)
print(f"Uninitialized tensor:\n{x_empty}\n")

# Random tensor
x_rand = torch.rand(2, 3)
print(f"Random tensor:\n{x_rand}\n")

# Zeros tensor
x_zeros = torch.zeros(2, 3)
print(f"Zeros tensor:\n{x_zeros}\n")

# Ones tensor
x_ones = torch.ones(2, 3)
print(f"Ones tensor:\n{x_ones}\n")

# Tensor with specific data type
x_float = torch.ones(2, 3, dtype=torch.float32)
print(f"Float tensor:\n{x_float}\n")

# Tensor operations
# Addition
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
print(f"Addition: {x + y}\n")

# Multiplication
print(f"Multiplication: {x * y}\n")

# Matrix multiplication
matrix1 = torch.tensor([[1, 2], [3, 4]])
matrix2 = torch.tensor([[5, 6], [7, 8]])
print(f"Matrix multiplication:\n{torch.matmul(matrix1, matrix2)}\n")

# Reshaping tensors
x = torch.randn(4, 4)
print(f"Original tensor shape: {x.shape}")
x_reshaped = x.view(2, 8)  # or x.reshape(2, 8)
print(f"Reshaped tensor shape: {x_reshaped.shape}\n")

# Slicing tensors
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"First row: {x[0, :]}")
print(f"First column: {x[:, 0]}")
print(f"Element at (1, 1): {x[1, 1]}\n")

# Moving tensors between CPU and GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create tensor on CPU
x_cpu = torch.randn(2, 3)
print(f"Tensor device (before): {x_cpu.device}")

# Move tensor to GPU
x_gpu = x_cpu.to(device)
print(f"Tensor device (after): {x_gpu.device}")

# Move tensor back to CPU
x_cpu_again = x_gpu.cpu()
print(f"Tensor device (back to CPU): {x_cpu_again.device}\n")

# Tensor to NumPy conversion
x = torch.ones(5)
numpy_array = x.numpy()
print(f"Tensor: {x}")
print(f"NumPy array: {numpy_array}")

# NumPy to tensor conversion
numpy_array = np.array([1, 2, 3, 4, 5])
x = torch.from_numpy(numpy_array)
print(f"NumPy array: {numpy_array}")
print(f"Tensor: {x}")
```

**When and where we use Tensors:**
- When representing input data for neural networks (images, text, audio)
- When storing model parameters (weights and biases)
- When computing gradients during backpropagation
- When implementing custom layers or operations
- When preprocessing or transforming data

**Real-world examples:**
- In computer vision, images are represented as 3D tensors (height × width × channels)
- In NLP, sentences are represented as 2D tensors (sequence_length × embedding_dimension)
- In reinforcement learning, state representations are stored as tensors
- In recommendation systems, user-item interactions are stored in tensors

### 2. Autograd and Computation Graphs

**What is Autograd?**
Autograd is PyTorch's automatic differentiation engine that powers neural network training. It automatically computes gradients of tensor operations with respect to parameters that have `requires_grad=True`, which is essential for optimizing models using gradient descent.

**PyTorch Implementation:**
```python
import torch

# Basic autograd example
# Create tensors with requires_grad=True to track computation
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Build a computation graph
y = w * x + b  # y = 3 * 2 + 1 = 7

# Compute gradients
y.backward()

# Print gradients
print(f"x.grad: {x.grad}")  # dy/dx = w = 3
print(f"w.grad: {w.grad}")  # dy/dw = x = 2
print(f"b.grad: {b.grad}")  # dy/db = 1

# More complex example with a neural network layer
# Create input and parameters
x = torch.randn(1, 3)  # Input tensor
w = torch.randn(3, 3, requires_grad=True)  # Weight matrix
b = torch.randn(1, 3, requires_grad=True)  # Bias vector

# Forward pass
y = x @ w + b  # Matrix multiplication followed by addition
z = y.sigmoid()  # Apply sigmoid activation
loss = (1 - z).sum()  # Dummy loss function

# Backward pass
loss.backward()

# Check gradients
print(f"\nw.grad shape: {w.grad.shape}")
print(f"b.grad shape: {b.grad.shape}")

# Example with multiple operations
x = torch.ones(2, 2, requires_grad=True)
print(f"\nOriginal x:\n{x}")

# Operation 1
y = x + 2
print(f"After adding 2:\n{y}")

# Operation 2
z = y * y * 3
print(f"After squaring and multiplying by 3:\n{z}")

# Operation 3
out = z.mean()
print(f"Mean of z: {out}")

# Backward pass
out.backward()

# Gradient of x
print(f"\nGradient of x:\n{x.grad}")

# Example with preventing gradient tracking
# Method 1: torch.no_grad()
x = torch.ones(2, 2, requires_grad=True)
print(f"\nx requires_grad: {x.requires_grad}")

with torch.no_grad():
    y = x * 2
    print(f"y requires_grad: {y.requires_grad}")

# Method 2: detach()
x = torch.ones(2, 2, requires_grad=True)
y = x.detach()
print(f"\nx requires_grad: {x.requires_grad}")
print(f"y requires_grad: {y.requires_grad}")

# Example with gradient accumulation
model = torch.nn.Linear(10, 1)  # Simple linear model
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Simulate multiple batches
for batch in range(3):
    # Create dummy data
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    # Forward pass
    outputs = model(x)
    loss = torch.nn.functional.mse_loss(outputs, y)
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    
    # Print gradient for the first parameter
    print(f"\nBatch {batch+1} - Gradient for first parameter: {model.weight.grad[0, 0].item():.4f}")
    
    # Zero gradients for next iteration
    optimizer.zero_grad()

# Example with custom autograd function
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)  # Save input for backward pass
        return x.clamp(min=0)  # ReLU: max(0, x)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[x < 0] = 0  # Gradient is 0 for x < 0
        return grad_x

# Use custom function
x = torch.tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
y = MyReLU.apply(x)
y.sum().backward()

print(f"\nCustom ReLU forward: {y}")
print(f"Custom ReLU gradient: {x.grad}")
```

**When and where we use Autograd:**
- When training neural networks with gradient-based optimization
- When implementing custom layers or loss functions
- When computing gradients for sensitivity analysis
- When debugging models by examining gradient flow
- When implementing advanced optimization techniques

**Real-world examples:**
- In image classification, autograd computes gradients for backpropagation through CNNs
- In machine translation, it enables training of sequence-to-sequence models
- In generative adversarial networks (GANs), it's used to train both generator and discriminator
- In reinforcement learning, it computes gradients for policy optimization

### 3. Neural Network Module (nn.Module)

**What is nn.Module?**
`nn.Module` is the base class for all neural network modules in PyTorch. It provides a convenient way to encapsulate parameters, define the forward pass, and handle other aspects of neural networks. By subclassing `nn.Module`, you can create custom layers or entire models with learnable parameters.

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic neural network using nn.Module
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(784, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()  # Activation function
        self.dropout = nn.Dropout(0.2)  # Dropout layer
        
    def forward(self, x):
        # Define forward pass
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Create an instance of the network
model = SimpleNet()
print(model)

# Accessing model parameters
print("\nModel parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Moving model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"\nModel moved to: {device}")

# Example with a CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)
        # Fully connected layers with ReLU and dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create an instance of the CNN
cnn_model = SimpleCNN()
print(f"\nCNN model:\n{cnn_model}")

# Example with a recurrent neural network
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        # Forward pass through RNN
        out, _ = self.rnn(x, h0)
        # Get the output from the last time step
        out = out[:, -1, :]
        # Pass through fully connected layer
        out = self.fc(out)
        return out

# Create an instance of the RNN
input_size = 10
hidden_size = 20
output_size = 5
rnn_model = SimpleRNN(input_size, hidden_size, output_size)
print(f"\nRNN model:\n{rnn_model}")

# Example with a custom layer
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.bias)
        
    def forward(self, x):
        # Custom linear transformation
        return F.linear(x, self.weight, self.bias)

# Example using the custom layer
class NetWithCustomLayer(nn.Module):
    def __init__(self):
        super(NetWithCustomLayer, self).__init__()
        self.custom_fc1 = CustomLinear(784, 128)
        self.custom_fc2 = CustomLinear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.custom_fc1(x))
        x = self.relu(self.custom_fc2(x))
        x = self.fc3(x)
        return x

# Create an instance of the network with custom layer
custom_model = NetWithCustomLayer()
print(f"\nModel with custom layer:\n{custom_model}")

# Example with sequential container
sequential_model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
print(f"\nSequential model:\n{sequential_model}")

# Example with nested modules
class NestedNet(nn.Module):
    def __init__(self):
        super(NestedNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Create an instance of the nested network
nested_model = NestedNet()
print(f"\nNested model:\n{nested_model}")

# Example with saving and loading model parameters
# Save model
torch.save(model.state_dict(), 'simple_model.pth')
print("\nModel saved to simple_model.pth")

# Load model
new_model = SimpleNet()
new_model.load_state_dict(torch.load('simple_model.pth'))
print("Model loaded from simple_model.pth")

# Example with freezing layers
for param in model.fc1.parameters():
    param.requires_grad = False

print("\nFrozen fc1 layer parameters:")
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f"{name}: requires_grad={param.requires_grad}")
```

**When and where we use nn.Module:**
- When defining custom neural network architectures
- When creating reusable layers or components
- When organizing complex models with multiple sub-components
- When implementing models with shared parameters
- When building models that need to save/load state

**Real-world examples:**
- In computer vision, CNNs are defined as nn.Module subclasses for image classification
- In NLP, transformers are implemented using nn.Module for language modeling
- In reinforcement learning, policy networks are defined as nn.Module subclasses
- In generative modeling, VAEs and GANs use nn.Module for their architectures

### 4. Optimizers

**What are Optimizers?**
Optimizers are algorithms that adjust the parameters of a neural network to minimize the loss function. They implement various variants of gradient descent and are essential for training neural networks. PyTorch provides several optimizers in the `torch.optim` module.

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Create a simple model
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# Create dummy data
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# Define loss function
criterion = nn.MSELoss()

# Example with different optimizers
optimizers = {
    'SGD': optim.SGD(model.parameters(), lr=0.01),
    'SGD with Momentum': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    'Adam': optim.Adam(model.parameters(), lr=0.001),
    'RMSprop': optim.RMSprop(model.parameters(), lr=0.01),
    'Adagrad': optim.Adagrad(model.parameters(), lr=0.01),
    'Adadelta': optim.Adadelta(model.parameters(), lr=1.0)
}

# Training function
def train_model(optimizer, X, y, epochs=100):
    # Reset model parameters
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    
    losses = []
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return losses

# Train with different optimizers
plt.figure(figsize=(12, 8))
for name, optimizer in optimizers.items():
    print(f"\nTraining with {name}:")
    losses = train_model(optimizer, X, y)
    plt.plot(losses, label=name)

plt.title('Training Loss with Different Optimizers')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Example with learning rate scheduling
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

optimizer = optim.SGD(model.parameters(), lr=0.1)

# Define learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Training with learning rate scheduling
losses = []
lrs = []
for epoch in range(100):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Update learning rate
    scheduler.step()
    
    losses.append(loss.item())
    lrs.append(optimizer.param_groups[0]['lr'])
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{100}], Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

# Plot loss and learning rate
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss with Learning Rate Scheduling')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(lrs)
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.grid(True)

plt.tight_layout()
plt.show()

# Example with custom optimizer
class CustomSGD(optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0):
        defaults = dict(lr=lr, momentum=momentum)
        super(CustomSGD, self).__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            momentum = group['momentum']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('CustomSGD does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    if momentum > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                
                state['step'] += 1
                
                if momentum > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)
                    p.data.add_(-group['lr'], buf)
                else:
                    p.data.add_(-group['lr'], grad)
        
        return loss

# Use custom optimizer
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

custom_optimizer = CustomSGD(model.parameters(), lr=0.01, momentum=0.9)

print("\nTraining with custom SGD optimizer:")
losses = train_model(custom_optimizer, X, y)

plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Training Loss with Custom SGD Optimizer')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Example with different parameter groups
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# Different learning rates for different layers
optimizer = optim.SGD([
    {'params': model[0].parameters(), 'lr': 0.001},
    {'params': model[2].parameters(), 'lr': 0.01}
], momentum=0.9)

print("\nTraining with different learning rates for different layers:")
losses = train_model(optimizer, X, y)

plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Training Loss with Different Learning Rates')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Example with gradient clipping
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training with gradient clipping
losses = []
for epoch in range(100):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Optimize
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{100}], Loss: {loss.item():.4f}')

plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Training Loss with Gradient Clipping')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
```

**When and where we use Optimizers:**
- When training neural networks with gradient-based optimization
- When fine-tuning pre-trained models with different learning rates
- When implementing custom optimization algorithms
- When dealing with vanishing or exploding gradients
- When training models with sparse gradients

**Real-world examples:**
- In computer vision, Adam optimizer is commonly used for training CNNs
- In NLP, transformers are often trained with AdamW optimizer
- In reinforcement learning, policy gradients are optimized with RMSprop
- In GAN training, different optimizers might be used for generator and discriminator

### 5. Datasets and DataLoaders

**What are Datasets and DataLoaders?**
PyTorch provides two data primitives: `Dataset` and `DataLoader`. The `Dataset` stores the samples and their corresponding labels, and `DataLoader` wraps an iterable around the `Dataset` to enable easy access to the samples. These utilities help in efficiently loading and preprocessing data for training neural networks.

**PyTorch Implementation:**
```python
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Example with TensorDataset for simple tensor data
# Create dummy data
X = torch.randn(1000, 10)  # 1000 samples, 10 features
y = torch.randint(0, 2, (1000,))  # 1000 binary labels

# Create TensorDataset
tensor_dataset = TensorDataset(X, y)

# Create DataLoader
batch_size = 32
tensor_dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

# Iterate through the DataLoader
print("TensorDataset and DataLoader example:")
for i, (inputs, labels) in enumerate(tensor_dataloader):
    print(f"Batch {i+1}: inputs shape: {inputs.shape}, labels shape: {labels.shape}")
    if i >= 2:  # Just show first 3 batches
        break

# Example with custom Dataset
class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label

# Create custom dataset with a simple transform
transform = lambda x: x * 2  # Simple transformation function
custom_dataset = CustomDataset(X, y, transform=transform)

# Create DataLoader
custom_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Iterate through the DataLoader
print("\nCustomDataset and DataLoader example:")
for i, (inputs, labels) in enumerate(custom_dataloader):
    print(f"Batch {i+1}: inputs shape: {inputs.shape}, labels shape: {labels.shape}")
    print(f"Original sample: {X[i*batch_size]}")
    print(f"Transformed sample: {inputs[0]}")
    if i >= 0:  # Just show first batch
        break

# Example with image dataset
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # For this example, we'll just use a dummy label
        label = torch.tensor(0)
        
        return image, label

# Define transforms for image preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Note: This example assumes you have a directory with images
# Uncomment and modify the following lines if you have such a directory
# image_dataset = CustomImageDataset('path/to/your/images', transform=transform)
# image_dataloader = DataLoader(image_dataset, batch_size=4, shuffle=True)

# Example with built-in datasets (MNIST)
# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load the MNIST training dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True,
    download=True,
    transform=transform
)

# Download and load the MNIST test dataset
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Explore the dataset
print("\nMNIST dataset example:")
print(f"Training dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Get a sample from the dataset
sample_image, sample_label = train_dataset[0]
print(f"Sample image shape: {sample_image.shape}")
print(f"Sample label: {sample_label}")

# Visualize some samples
plt.figure(figsize=(10, 5))
for i in range(10):
    image, label = train_dataset[i]
    plt.subplot(2, 5, i+1)
    plt.imshow(image.squeeze().numpy(), cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Example with data splitting
# Split the training dataset into training and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# Create DataLoaders for the subsets
train_subset_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_subset_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

print(f"\nTraining subset size: {len(train_subset)}")
print(f"Validation subset size: {len(val_subset)}")

# Example with collate_fn for custom batching
class CustomCollate:
    def __call__(self, batch):
        # Custom collate function for handling variable-sized data
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        
        # Pad sequences to the same length (for NLP tasks)
        if isinstance(data[0], torch.Tensor) and len(data[0].shape) == 1:  # 1D sequences
            # Get lengths of each sequence
            lengths = [len(x) for x in data]
            max_length = max(lengths)
            
            # Pad sequences
            padded_data = torch.zeros(len(data), max_length)
            for i, seq in enumerate(data):
                padded_data[i, :lengths[i]] = seq
                
            return padded_data, torch.tensor(target)
        else:
            # Default behavior for regular tensors
            return torch.stack(data), torch.tensor(target)

# Create a custom dataset with variable-length sequences
class SequenceDataset(Dataset):
    def __init__(self, num_samples, max_length=100):
        self.num_samples = num_samples
        self.max_length = max_length
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate a random length sequence
        length = np.random.randint(10, self.max_length)
        sequence = torch.randn(length)
        label = torch.tensor(np.random.randint(0, 2))
        return sequence, label

# Create the dataset and dataloader with custom collate function
seq_dataset = SequenceDataset(100)
seq_dataloader = DataLoader(seq_dataset, batch_size=4, collate_fn=CustomCollate())

# Iterate through the DataLoader
print("\nCustom collate function example:")
for i, (sequences, labels) in enumerate(seq_dataloader):
    print(f"Batch {i+1}: sequences shape: {sequences.shape}, labels shape: {labels.shape}")
    if i >= 1:  # Just show first 2 batches
        break

# Example with weighted sampling for imbalanced datasets
# Create an imbalanced dataset
class ImbalancedDataset(Dataset):
    def __init__(self, num_samples=1000, imbalance_ratio=0.1):
        self.num_samples = num_samples
        self.imbalance_ratio = imbalance_ratio
        
        # Generate imbalanced labels
        minority_size = int(num_samples * imbalance_ratio)
        majority_size = num_samples - minority_size
        
        minority_labels = torch.ones(minority_size)
        majority_labels = torch.zeros(majority_size)
        
        self.labels = torch.cat([minority_labels, majority_labels])
        # Shuffle the labels
        indices = torch.randperm(num_samples)
        self.labels = self.labels[indices]
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random features
        features = torch.randn(10)
        label = self.labels[idx]
        return features, label

# Create the imbalanced dataset
imbalanced_dataset = ImbalancedDataset(num_samples=1000, imbalance_ratio=0.1)

# Count the number of samples in each class
class_counts = torch.bincount(imbalanced_dataset.labels.long())
print(f"\nImbalanced dataset class counts: {class_counts}")

# Calculate class weights
class_weights = 1. / class_counts
sample_weights = class_weights[imbalanced_dataset.labels.long()]

# Create a WeightedRandomSampler
sampler = torch.utils.data.WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# Create DataLoader with the sampler
imbalanced_dataloader = DataLoader(
    imbalanced_dataset,
    batch_size=32,
    sampler=sampler
)

# Count the number of samples in each class in the sampled batches
sampled_class_counts = torch.zeros(2)
for i, (features, labels) in enumerate(imbalanced_dataloader):
    if i >= 10:  # Just count first 10 batches
        break
    sampled_class_counts += torch.bincount(labels.long(), minlength=2)

print(f"Sampled class counts (first 10 batches): {sampled_class_counts}")
```

**When and where we use Datasets and DataLoaders:**
- When loading and preprocessing data for training neural networks
- When working with large datasets that don't fit in memory
- When applying data augmentation or transformations
- When dealing with imbalanced datasets
- When working with variable-length sequences (e.g., in NLP)

**Real-world examples:**
- In computer vision, ImageFolder and custom datasets are used for image classification
- In NLP, custom datasets handle text data with variable-length sequences
- In reinforcement learning, datasets store experience tuples for training
- In healthcare, custom datasets manage medical images and patient records

## Applications in Deep Learning

### 1. Building a Simple Neural Network

Let's build a simple neural network for classification using PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the neural network
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Hyperparameters
input_size = X.shape[1]
hidden_size = 64
num_classes = 2
learning_rate = 0.001
num_epochs = 50

# Initialize the model
model = SimpleClassifier(input_size, hidden_size, num_classes)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Calculate training loss and accuracy
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # Validation
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
    
    # Calculate validation loss and accuracy
    val_loss = running_loss / len(test_loader)
    val_accuracy = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    # Print statistics
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

# Plot training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model on the test set
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f'Test Accuracy: {accuracy:.4f}')

# Save the model
torch.save(model.state_dict(), 'simple_classifier.pth')
print("Model saved to simple_classifier.pth")

# Load the model
new_model = SimpleClassifier(input_size, hidden_size, num_classes)
new_model.load_state_dict(torch.load('simple_classifier.pth'))
print("Model loaded from simple_classifier.pth")
```

### 2. Implementing a Convolutional Neural Network (CNN)

Let's implement a CNN for image classification using PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Define transforms for data preprocessing and augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Download and load the CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform_train
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform_test
)

# Create DataLoaders
batch_size = 128
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Dropout
        self.dropout = nn.Dropout(0.25)
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten the output
        x = x.view(-1, 128 * 4 * 4)
        # Apply dropout
        x = self.dropout(x)
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize the model
model = CNN(num_classes=10)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using device: {device}")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# Training function
def train(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}], Step [{i+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

# Testing function
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Calculate statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

# Training loop
num_epochs = 20
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch, device)
    test_loss, test_acc = test(model, test_loader, criterion, device)
    
    # Update learning rate
    scheduler.step(test_loss)
    
    # Save statistics
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    
    # Print statistics
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

# Plot training and test loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Training and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training and test accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model on the test set
model.eval()
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# Print per-class accuracy
for i in range(10):
    print(f'Accuracy of class {i}: {100 * class_correct[i] / class_total[i]:.2f}%')

# Save the model
torch.save(model.state_dict(), 'cifar10_cnn.pth')
print("Model saved to cifar10_cnn.pth")

# Load the model
new_model = CNN(num_classes=10)
new_model.load_state_dict(torch.load('cifar10_cnn.pth'))
new_model = new_model.to(device)
print("Model loaded from cifar10_cnn.pth")

# Visualize some predictions
def visualize_predictions(model, test_loader, device, classes, num_images=5):
    model.eval()
    
    # Get a batch of test data
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Move to device
    images, labels = images.to(device), labels.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Move back to CPU for visualization
    images = images.cpu()
    labels = labels.cpu()
    predicted = predicted.cpu()
    
    # Denormalize images
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    images = images * std + mean
    
    # Plot images
    plt.figure(figsize=(15, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(np.transpose(images[i], (1, 2, 0)))
        plt.title(f'Pred: {classes[predicted[i]]}, True: {classes[labels[i]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Visualize predictions
visualize_predictions(new_model, test_loader, device, classes)
```

### 3. Implementing a Recurrent Neural Network (RNN)

Let's implement an RNN for sequence classification using PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Create a synthetic sequence dataset
class SequenceDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=20, input_size=10, num_classes=2):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Generate random sequences
        self.sequences = torch.randn(num_samples, seq_length, input_size)
        
        # Generate labels based on a simple rule
        # If the sum of the first half of the sequence is > 0, class 1, else class 0
        first_half_sum = self.sequences[:, :seq_length//2, :].sum(dim=(1, 2))
        self.labels = (first_half_sum > 0).long()
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return sequence, label

# Create dataset and split into train and test
dataset = SequenceDataset(num_samples=2000, seq_length=20, input_size=10, num_classes=2)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through RNN
        out, _ = self.rnn(x, h0)
        
        # Get the output from the last time step
        out = out[:, -1, :]
        
        # Pass through fully connected layer
        out = self.fc(out)
        
        return out

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the output from the last time step
        out = out[:, -1, :]
        
        # Pass through fully connected layer
        out = self.fc(out)
        
        return out

# Define the GRU model
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through GRU
        out, _ = self.gru(x, h0)
        
        # Get the output from the last time step
        out = out[:, -1, :]
        
        # Pass through fully connected layer
        out = self.fc(out)
        
        return out

# Hyperparameters
input_size = 10
hidden_size = 64
num_layers = 2
num_classes = 2
learning_rate = 0.001
num_epochs = 20

# Initialize models
rnn_model = SimpleRNN(input_size, hidden_size, num_layers, num_classes)
lstm_model = LSTM(input_size, hidden_size, num_layers, num_classes)
gru_model = GRU(input_size, hidden_size, num_layers, num_classes)

# Move models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rnn_model = rnn_model.to(device)
lstm_model = lstm_model.to(device)
gru_model = gru_model.to(device)
print(f"Using device: {device}")

# Loss function and optimizers
criterion = nn.CrossEntropyLoss()
rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)
gru_optimizer = optim.Adam(gru_model.parameters(), lr=learning_rate)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    train_losses = []
    train_accuracies = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    
    return train_losses, train_accuracies

# Testing function
def test_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Calculate statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

# Train RNN model
print("\nTraining RNN model:")
rnn_train_losses, rnn_train_accuracies = train_model(
    rnn_model, train_loader, criterion, rnn_optimizer, num_epochs, device
)

# Test RNN model
rnn_test_loss, rnn_test_acc, rnn_preds, rnn_labels = test_model(
    rnn_model, test_loader, criterion, device
)
print(f'RNN Test Loss: {rnn_test_loss:.4f}, Test Accuracy: {rnn_test_acc:.4f}')

# Train LSTM model
print("\nTraining LSTM model:")
lstm_train_losses, lstm_train_accuracies = train_model(
    lstm_model, train_loader, criterion, lstm_optimizer, num_epochs, device
)

# Test LSTM model
lstm_test_loss, lstm_test_acc, lstm_preds, lstm_labels = test_model(
    lstm_model, test_loader, criterion, device
)
print(f'LSTM Test Loss: {lstm_test_loss:.4f}, Test Accuracy: {lstm_test_acc:.4f}')

# Train GRU model
print("\nTraining GRU model:")
gru_train_losses, gru_train_accuracies = train_model(
    gru_model, train_loader, criterion, gru_optimizer, num_epochs, device
)

# Test GRU model
gru_test_loss, gru_test_acc, gru_preds, gru_labels = test_model(
    gru_model, test_loader, criterion, device
)
print(f'GRU Test Loss: {gru_test_loss:.4f}, Test Accuracy: {gru_test_acc:.4f}')

# Plot training losses
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(rnn_train_losses, label='RNN')
plt.title('RNN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(lstm_train_losses, label='LSTM')
plt.title('LSTM Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(gru_train_losses, label='GRU')
plt.title('GRU Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot training accuracies
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(rnn_train_accuracies, label='RNN')
plt.title('RNN Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(lstm_train_accuracies, label='LSTM')
plt.title('LSTM Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(gru_train_accuracies, label='GRU')
plt.title('GRU Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

plt.tight_layout()
plt.show()

# Compare model performance
models = ['RNN', 'LSTM', 'GRU']
test_losses = [rnn_test_loss, lstm_test_loss, gru_test_loss]
test_accuracies = [rnn_test_acc, lstm_test_acc, gru_test_acc]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(models, test_losses)
plt.title('Test Loss Comparison')
plt.ylabel('Loss')
plt.grid(True, axis='y')

plt.subplot(1, 2, 2)
plt.bar(models, test_accuracies)
plt.title('Test Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid(True, axis='y')

plt.tight_layout()
plt.show()

# Print classification reports
print("\nRNN Classification Report:")
print(classification_report(rnn_labels, rnn_preds))

print("\nLSTM Classification Report:")
print(classification_report(lstm_labels, lstm_preds))

print("\nGRU Classification Report:")
print(classification_report(gru_labels, gru_preds))

# Save the best model
best_model_idx = np.argmax(test_accuracies)
best_model_name = models[best_model_idx]
best_model = [rnn_model, lstm_model, gru_model][best_model_idx]

torch.save(best_model.state_dict(), f'{best_model_name.lower()}_model.pth')
print(f"\nBest model ({best_model_name}) saved to {best_model_name.lower()}_model.pth")
```

## Common Problems and Advancements

### 1. Problem: Vanishing and Exploding Gradients

**Issue:**
In deep neural networks, especially in RNNs, gradients can become extremely small (vanish) or extremely large (explode) during backpropagation. Vanishing gradients make it difficult for the network to learn long-range dependencies, while exploding gradients can cause the training to diverge.

**How advancements solved this:**
- **Gradient clipping**: Limiting the maximum value of gradients to prevent explosion
- **Better initialization techniques**: Using methods like Xavier/Glorot and He initialization
- **Batch normalization**: Normalizing the inputs of each layer to reduce internal covariate shift
- **Advanced RNN architectures**: Using LSTM and GRU units that are designed to mitigate vanishing gradients
- **Residual connections**: Allowing gradients to flow directly through skip connections

```python
# Problem: Vanishing and Exploding Gradients
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Create a simple deep network to demonstrate vanishing/exploding gradients
class DeepNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(DeepNetwork, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_size, output_size))
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # No activation on output layer
                x = torch.sigmoid(x)  # Using sigmoid which is prone to vanishing gradients
        return x

# Create a deeper network
input_size = 10
hidden_size = 64
output_size = 2
num_layers = 20  # Deep network to demonstrate the problem

model = DeepNetwork(input_size, hidden_size, output_size, num_layers)

# Create dummy data
X = torch.randn(64, input_size)
y = torch.randint(0, output_size, (64,))

# Define loss function
criterion = nn.CrossEntropyLoss()

# Forward pass
outputs = model(X)
loss = criterion(outputs, y)

# Backward pass
loss.backward()

# Collect gradients
gradients = []
for name, param in model.named_parameters():
    if 'weight' in name and 'layers' in name:
        gradients.append(param.grad.abs().mean().item())

# Plot gradient magnitudes
plt.figure(figsize=(12, 6))
plt.plot(range(len(gradients)), gradients, 'o-')
plt.title('Gradient Magnitudes in a Deep Network (Vanishing Gradients)')
plt.xlabel('Layer Depth')
plt.ylabel('Average Gradient Magnitude')
plt.yscale('log')
plt.grid(True)
plt.show()

# Enhancement 1: Gradient clipping
# Create a new model
model_clip = DeepNetwork(input_size, hidden_size, output_size, num_layers)

# Define optimizer
optimizer = optim.SGD(model_clip.parameters(), lr=0.01)

# Forward pass
outputs = model_clip(X)
loss = criterion(outputs, y)

# Backward pass with gradient clipping
optimizer.zero_grad()
loss.backward()

# Apply gradient clipping
torch.nn.utils.clip_grad_norm_(model_clip.parameters(), max_norm=1.0)

# Collect gradients
gradients_clip = []
for name, param in model_clip.named_parameters():
    if 'weight' in name and 'layers' in name:
        gradients_clip.append(param.grad.abs().mean().item())

# Plot gradient magnitudes with clipping
plt.figure(figsize=(12, 6))
plt.plot(range(len(gradients)), gradients, 'o-', label='Without Clipping')
plt.plot(range(len(gradients_clip)), gradients_clip, 'o-', label='With Clipping')
plt.title('Effect of Gradient Clipping')
plt.xlabel('Layer Depth')
plt.ylabel('Average Gradient Magnitude')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

# Enhancement 2: Better initialization
# Create a new model with better initialization
class DeepNetworkBetterInit(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(DeepNetworkBetterInit, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer with Xavier initialization
        self.layers.append(nn.Linear(input_size, hidden_size))
        nn.init.xavier_uniform_(self.layers[-1].weight)
        
        # Hidden layers with Xavier initialization
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            nn.init.xavier_uniform_(self.layers[-1].weight)
        
        # Output layer with Xavier initialization
        self.layers.append(nn.Linear(hidden_size, output_size))
        nn.init.xavier_uniform_(self.layers[-1].weight)
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # No activation on output layer
                x = torch.sigmoid(x)
        return x

# Create a model with better initialization
model_better_init = DeepNetworkBetterInit(input_size, hidden_size, output_size, num_layers)

# Forward pass
outputs = model_better_init(X)
loss = criterion(outputs, y)

# Backward pass
loss.backward()

# Collect gradients
gradients_better_init = []
for name, param in model_better_init.named_parameters():
    if 'weight' in name and 'layers' in name:
        gradients_better_init.append(param.grad.abs().mean().item())

# Plot gradient magnitudes with better initialization
plt.figure(figsize=(12, 6))
plt.plot(range(len(gradients)), gradients, 'o-', label='Default Initialization')
plt.plot(range(len(gradients_better_init)), gradients_better_init, 'o-', label='Xavier Initialization')
plt.title('Effect of Better Initialization')
plt.xlabel('Layer Depth')
plt.ylabel('Average Gradient Magnitude')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

# Enhancement 3: Batch normalization
# Create a new model with batch normalization
class DeepNetworkWithBN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(DeepNetworkWithBN, self).__init__()
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.bn_layers.append(nn.BatchNorm1d(hidden_size))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.bn_layers.append(nn.BatchNorm1d(hidden_size))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_size, output_size))
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # No BN on output layer
                x = self.bn_layers[i](x)
                x = torch.sigmoid(x)
        return x

# Create a model with batch normalization
model_bn = DeepNetworkWithBN(input_size, hidden_size, output_size, num_layers)

# Forward pass
outputs = model_bn(X)
loss = criterion(outputs, y)

# Backward pass
loss.backward()

# Collect gradients
gradients_bn = []
for name, param in model_bn.named_parameters():
    if 'weight' in name and 'layers' in name:
        gradients_bn.append(param.grad.abs().mean().item())

# Plot gradient magnitudes with batch normalization
plt.figure(figsize=(12, 6))
plt.plot(range(len(gradients)), gradients, 'o-', label='Without BN')
plt.plot(range(len(gradients_bn)), gradients_bn, 'o-', label='With BN')
plt.title('Effect of Batch Normalization')
plt.xlabel('Layer Depth')
plt.ylabel('Average Gradient Magnitude')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

# Enhancement 4: Residual connections
# Create a new model with residual connections
class DeepNetworkWithResidual(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(DeepNetworkWithResidual, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        
        # Hidden layers
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_size, output_size))
        
    def forward(self, x):
        # Input layer
        x = torch.sigmoid(self.layers[0](x))
        
        # Hidden layers with residual connections
        for i in range(1, len(self.layers) - 1):
            identity = x
            x = torch.sigmoid(self.layers[i](x))
            
            # Add residual connection if dimensions match
            if identity.shape == x.shape:
                x = x + identity
        
        # Output layer
        x = self.layers[-1](x)
        
        return x

# Create a model with residual connections
model_residual = DeepNetworkWithResidual(input_size, hidden_size, output_size, num_layers)

# Forward pass
outputs = model_residual(X)
loss = criterion(outputs, y)

# Backward pass
loss.backward()

# Collect gradients
gradients_residual = []
for name, param in model_residual.named_parameters():
    if 'weight' in name and 'layers' in name:
        gradients_residual.append(param.grad.abs().mean().item())

# Plot gradient magnitudes with residual connections
plt.figure(figsize=(12, 6))
plt.plot(range(len(gradients)), gradients, 'o-', label='Without Residual')
plt.plot(range(len(gradients_residual)), gradients_residual, 'o-', label='With Residual')
plt.title('Effect of Residual Connections')
plt.xlabel('Layer Depth')
plt.ylabel('Average Gradient Magnitude')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

# Enhancement 5: LSTM for sequence models
# Create a simple RNN and an LSTM to compare gradient flow
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Create sequence data
seq_length = 50
batch_size = 16
input_size = 10
output_size = 2

X_seq = torch.randn(batch_size, seq_length, input_size)
y_seq = torch.randint(0, output_size, (batch_size,))

# Create models
rnn_model = SimpleRNN(input_size, hidden_size, output_size, num_layers=3)
lstm_model = SimpleLSTM(input_size, hidden_size, output_size, num_layers=3)

# Function to collect gradients during backpropagation
def collect_gradients(model, X, y):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Collect gradients
    gradients = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            gradients.append(param.grad.abs().mean().item())
    
    return gradients

# Collect gradients for RNN
rnn_gradients = collect_gradients(rnn_model, X_seq, y_seq)

# Collect gradients for LSTM
lstm_gradients = collect_gradients(lstm_model, X_seq, y_seq)

# Plot gradient magnitudes
plt.figure(figsize=(12, 6))
plt.plot(range(len(rnn_gradients)), rnn_gradients, 'o-', label='RNN')
plt.plot(range(len(lstm_gradients)), lstm_gradients, 'o-', label='LSTM')
plt.title('Gradient Magnitudes: RNN vs LSTM')
plt.xlabel('Parameter Index')
plt.ylabel('Average Gradient Magnitude')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()
```

### 2. Problem: Overfitting

**Issue:**
Neural networks, especially deep ones with many parameters, are prone to overfitting the training data. This means they perform well on the training data but poorly on unseen test data, indicating that they have memorized the training examples rather than learning generalizable patterns.

**How advancements solved this:**
- **Regularization techniques**: Adding penalties to the loss function to discourage complex models
- **Dropout**: Randomly dropping units during training to prevent co-adaptation of features
- **Data augmentation**: Artificially increasing the size and diversity of the training data
- **Early stopping**: Monitoring validation performance and stopping training when it starts to degrade
- **Batch normalization**: Reducing internal covariate shift and acting as a regularizer

```python
# Problem: Overfitting in Neural Networks
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Generate synthetic data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2 of original data

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define a neural network without regularization
class OverfittingNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(OverfittingNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Hyperparameters
input_size = X.shape[1]
hidden_size = 128
output_size = 2
learning_rate = 0.01
num_epochs = 100

# Initialize the model without regularization
model_overfit = OverfittingNet(input_size, hidden_size, output_size)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_overfit.parameters(), lr=learning_rate)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
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
        
        # Validation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# Train the model without regularization
print("Training model without regularization:")
train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model_overfit, train_loader, val_loader, criterion, optimizer, num_epochs
)

# Plot training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss (Without Regularization)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy (Without Regularization)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Enhancement 1: L2 regularization (weight decay)
# Define a neural network with L2 regularization
class RegularizedNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegularizedNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize the model with L2 regularization
model_l2 = RegularizedNet(input_size, hidden_size, output_size)

# Optimizer with weight decay (L2 regularization)
optimizer_l2 = optim.Adam(model_l2.parameters(), lr=learning_rate, weight_decay=0.01)  # weight_decay is the L2 regularization parameter

# Train the model with L2 regularization
print("\nTraining model with L2 regularization:")
train_losses_l2, val_losses_l2, train_accuracies_l2, val_accuracies_l2 = train_model(
    model_l2, train_loader, val_loader, criterion, optimizer_l2, num_epochs
)

# Plot training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss (No Reg)')
plt.plot(val_losses, label='Validation Loss (No Reg)')
plt.plot(train_losses_l2, label='Training Loss (L2)')
plt.plot(val_losses_l2, label='Validation Loss (L2)')
plt.title('Training and Validation Loss (With L2 Regularization)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy (No Reg)')
plt.plot(val_accuracies, label='Validation Accuracy (No Reg)')
plt.plot(train_accuracies_l2, label='Training Accuracy (L2)')
plt.plot(val_accuracies_l2, label='Validation Accuracy (L2)')
plt.title('Training and Validation Accuracy (With L2 Regularization)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Enhancement 2: Dropout
# Define a neural network with dropout
class DropoutNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(DropoutNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

# Initialize the model with dropout
model_dropout = DropoutNet(input_size, hidden_size, output_size, dropout_rate=0.5)

# Optimizer
optimizer_dropout = optim.Adam(model_dropout.parameters(), lr=learning_rate)

# Train the model with dropout
print("\nTraining model with dropout:")
train_losses_dropout, val_losses_dropout, train_accuracies_dropout, val_accuracies_dropout = train_model(
    model_dropout, train_loader, val_loader, criterion, optimizer_dropout, num_epochs
)

# Plot training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss (No Reg)')
plt.plot(val_losses, label='Validation Loss (No Reg)')
plt.plot(train_losses_dropout, label='Training Loss (Dropout)')
plt.plot(val_losses_dropout, label='Validation Loss (Dropout)')
plt.title('Training and Validation Loss (With Dropout)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy (No Reg)')
plt.plot(val_accuracies, label='Validation Accuracy (No Reg)')
plt.plot(train_accuracies_dropout, label='Training Accuracy (Dropout)')
plt.plot(val_accuracies_dropout, label='Validation Accuracy (Dropout)')
plt.title('Training and Validation Accuracy (With Dropout)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Enhancement 3: Early stopping
# Define a neural network for early stopping
class EarlyStoppingNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EarlyStoppingNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize the model for early stopping
model_early = EarlyStoppingNet(input_size, hidden_size, output_size)

# Optimizer
optimizer_early = optim.Adam(model_early.parameters(), lr=learning_rate)

# Early stopping parameters
patience = 10  # Number of epochs to wait for improvement
min_delta = 0.001  # Minimum change to qualify as improvement
best_val_loss = float('inf')
counter = 0
best_model_state = None

# Training with early stopping
print("\nTraining model with early stopping:")
train_losses_early = []
val_losses_early = []
train_accuracies_early = []
val_accuracies_early = []

for epoch in range(num_epochs):
    model_early.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        # Forward pass
        outputs = model_early(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer_early.zero_grad()
        loss.backward()
        optimizer_early.step()
        
        # Calculate statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses_early.append(train_loss)
    train_accuracies_early.append(train_accuracy)
    
    # Validation
    model_early.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model_early(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_accuracy = correct / total
    val_losses_early.append(val_loss)
    val_accuracies_early.append(val_accuracy)
    
    # Check for improvement
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        counter = 0
        best_model_state = model_early.state_dict()
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

# Load the best model
model_early.load_state_dict(best_model_state)

# Plot training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss (No Reg)')
plt.plot(val_losses, label='Validation Loss (No Reg)')
plt.plot(train_losses_early, label='Training Loss (Early Stop)')
plt.plot(val_losses_early, label='Validation Loss (Early Stop)')
plt.title('Training and Validation Loss (With Early Stopping)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy (No Reg)')
plt.plot(val_accuracies, label='Validation Accuracy (No Reg)')
plt.plot(train_accuracies_early, label='Training Accuracy (Early Stop)')
plt.plot(val_accuracies_early, label='Validation Accuracy (Early Stop)')
plt.title('Training and Validation Accuracy (With Early Stopping)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Enhancement 4: Batch normalization
# Define a neural network with batch normalization
class BatchNormNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BatchNormNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

# Initialize the model with batch normalization
model_bn = BatchNormNet(input_size, hidden_size, output_size)

# Optimizer
optimizer_bn = optim.Adam(model_bn.parameters(), lr=learning_rate)

# Train the model with batch normalization
print("\nTraining model with batch normalization:")
train_losses_bn, val_losses_bn, train_accuracies_bn, val_accuracies_bn = train_model(
    model_bn, train_loader, val_loader, criterion, optimizer_bn, num_epochs
)

# Plot training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss (No Reg)')
plt.plot(val_losses, label='Validation Loss (No Reg)')
plt.plot(train_losses_bn, label='Training Loss (BN)')
plt.plot(val_losses_bn, label='Validation Loss (BN)')
plt.title('Training and Validation Loss (With Batch Normalization)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy (No Reg)')
plt.plot(val_accuracies, label='Validation Accuracy (No Reg)')
plt.plot(train_accuracies_bn, label='Training Accuracy (BN)')
plt.plot(val_accuracies_bn, label='Validation Accuracy (BN)')
plt.title('Training and Validation Accuracy (With Batch Normalization)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Enhancement 5: Combined regularization
# Define a neural network with combined regularization
class CombinedRegNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super(CombinedRegNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

# Initialize the model with combined regularization
model_combined = CombinedRegNet(input_size, hidden_size, output_size, dropout_rate=0.3)

# Optimizer with weight decay
optimizer_combined = optim.Adam(model_combined.parameters(), lr=learning_rate, weight_decay=0.001)

# Train the model with combined regularization
print("\nTraining model with combined regularization:")
train_losses_combined, val_losses_combined, train_accuracies_combined, val_accuracies_combined = train_model(
    model_combined, train_loader, val_loader, criterion, optimizer_combined, num_epochs
)

# Plot training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss (No Reg)')
plt.plot(val_losses, label='Validation Loss (No Reg)')
plt.plot(train_losses_combined, label='Training Loss (Combined)')
plt.plot(val_losses_combined, label='Validation Loss (Combined)')
plt.title('Training and Validation Loss (With Combined Regularization)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy (No Reg)')
plt.plot(val_accuracies, label='Validation Accuracy (No Reg)')
plt.plot(train_accuracies_combined, label='Training Accuracy (Combined)')
plt.plot(val_accuracies_combined, label='Validation Accuracy (Combined)')
plt.title('Training and Validation Accuracy (With Combined Regularization)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Compare all models on test set
def evaluate_model(model, test_loader, criterion):
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
    
    return test_loss, test_accuracy

# Evaluate all models
models = {
    'No Regularization': model_overfit,
    'L2 Regularization': model_l2,
    'Dropout': model_dropout,
    'Early Stopping': model_early,
    'Batch Normalization': model_bn,
    'Combined Regularization': model_combined
}

test_results = {}
for name, model in models.items():
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
    test_results[name] = {'test_loss': test_loss, 'test_accuracy': test_accuracy}
    print(f"{name} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Plot test results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(test_results.keys(), [results['test_loss'] for results in test_results.values()])
plt.title('Test Loss Comparison')
plt.ylabel('Loss')
plt.xticks(rotation=45)
plt.grid(True, axis='y')

plt.subplot(1, 2, 2)
plt.bar(test_results.keys(), [results['test_accuracy'] for results in test_results.values()])
plt.title('Test Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.grid(True, axis='y')

plt.tight_layout()
plt.show()
```

### 3. Problem: Computational Efficiency

**Issue:**
Training deep neural networks can be computationally expensive, especially with large datasets and complex architectures. This can lead to long training times and high resource requirements, making it difficult to iterate quickly or deploy models in resource-constrained environments.

**How advancements solved this:**
- **GPU acceleration**: Leveraging parallel processing capabilities of GPUs
- **Distributed training**: Spreading the workload across multiple devices or machines
- **Mixed precision training**: Using lower precision (float16) for most operations while maintaining float32 where needed
- **Model compression**: Techniques like pruning, quantization, and knowledge distillation
- **Efficient architectures**: Designing networks that are computationally efficient

```python
# Problem: Computational Efficiency in PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create a larger model to demonstrate computational challenges
class LargeModel(nn.Module):
    def __init__(self, num_classes=10):
        super(LargeModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Create a more efficient model
class EfficientModel(nn.Module):
    def __init__(self, num_classes=10):
        super(EfficientModel, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 5
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize(224),  # Resize to match VGG input size
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize models
large_model = LargeModel(num_classes=10).to(device)
efficient_model = EfficientModel(num_classes=10).to(device)

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

large_params = count_parameters(large_model)
efficient_params = count_parameters(efficient_model)

print(f"Large model parameters: {large_params:,}")
print(f"Efficient model parameters: {efficient_params:,}")
print(f"Parameter reduction: {(large_params - efficient_params) / large_params * 100:.2f}%")

# Measure inference time
def measure_inference_time(model, data_loader, device, num_batches=10):
    model.eval()
    total_time = 0.0
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            if i >= num_batches:
                break
                
            inputs = inputs.to(device)
            
            # Warm-up
            if i == 0:
                _ = model(inputs)
                continue
                
            # Measure time
            start_time = time.time()
            _ = model(inputs)
            end_time = time.time()
            
            total_time += (end_time - start_time)
    
    avg_time = total_time / (num_batches - 1)
    return avg_time

# Measure inference time for both models
large_inference_time = measure_inference_time(large_model, test_loader, device)
efficient_inference_time = measure_inference_time(efficient_model, test_loader, device)

print(f"\nLarge model average inference time per batch: {large_inference_time:.4f} seconds")
print(f"Efficient model average inference time per batch: {efficient_inference_time:.4f} seconds")
print(f"Inference time reduction: {(large_inference_time - efficient_inference_time) / large_inference_time * 100:.2f}%")

# Enhancement 1: GPU acceleration
# Compare CPU vs GPU performance
if torch.cuda.is_available():
    # Create a smaller model for quick comparison
    small_model = EfficientModel(num_classes=10)
    
    # Measure CPU inference time
    cpu_time = measure_inference_time(small_model, test_loader, 'cpu')
    
    # Measure GPU inference time
    small_model = small_model.to(device)
    gpu_time = measure_inference_time(small_model, test_loader, device)
    
    print(f"\nCPU inference time per batch: {cpu_time:.4f} seconds")
    print(f"GPU inference time per batch: {gpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")

# Enhancement 2: Mixed precision training
# Define training function with mixed precision support
def train_model_mixed_precision(model, train_loader, criterion, optimizer, device, use_amp=False, epochs=1):
    model.train()
    
    # Create GradScaler if using mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    total_time = 0.0
    correct = 0
    total = 0
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # Backward pass with mixed precision
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = correct / total
    return accuracy

# Compare standard precision vs. mixed precision
if torch.cuda.is_available():
    # Create models for comparison
    model_fp32 = EfficientModel(num_classes=10).to(device)
    model_fp16 = EfficientModel(num_classes=10).to(device)
    
    # Create optimizers
    optimizer_fp32 = optim.Adam(model_fp32.parameters(), lr=0.001)
    optimizer_fp16 = optim.Adam(model_fp16.parameters(), lr=0.001)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Train with standard precision
    print("\nTraining with standard precision (FP32)...")
    start_time = time.time()
    fp32_accuracy = train_model_mixed_precision(
        model_fp32, train_loader, criterion, optimizer_fp32, device, use_amp=False, epochs=1
    )
    fp32_time = time.time() - start_time
    
    # Train with mixed precision
    print("Training with mixed precision (FP16)...")
    start_time = time.time()
    fp16_accuracy = train_model_mixed_precision(
        model_fp16, train_loader, criterion, optimizer_fp16, device, use_amp=True, epochs=1
    )
    fp16_time = time.time() - start_time
    
    print(f"\nFP32 training time: {fp32_time:.2f} seconds, Accuracy: {fp32_accuracy:.4f}")
    print(f"FP16 training time: {fp16_time:.2f} seconds, Accuracy: {fp16_accuracy:.4f}")
    print(f"Speedup: {fp32_time / fp16_time:.2f}x")

# Enhancement 3: Model quantization
# Apply post-training quantization to the efficient model
print("\nApplying post-training quantization...")

# Create a quantized version of the efficient model
quantized_model = torch.quantization.quantize_dynamic(
    efficient_model,
    {nn.Linear, nn.Conv2d},
    dtype=torch.qint8
)

# Measure inference time for quantized model
quantized_inference_time = measure_inference_time(quantized_model, test_loader, device)

print(f"Original model inference time per batch: {efficient_inference_time:.4f} seconds")
print(f"Quantized model inference time per batch: {quantized_inference_time:.4f} seconds")
print(f"Speedup: {efficient_inference_time / quantized_inference_time:.2f}x")

# Compare model sizes
def get_model_size(model):
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

original_size = get_model_size(efficient_model)
quantized_size = get_model_size(quantized_model)

print(f"\nOriginal model size: {original_size:.2f} MB")
print(f"Quantized model size: {quantized_size:.2f} MB")
print(f"Size reduction: {(original_size - quantized_size) / original_size * 100:.2f}%")

# Enhancement 4: Profiling to identify bottlenecks
if torch.cuda.is_available():
    print("\nProfiling model execution...")
    
    # Create a simple model for profiling
    profile_model = EfficientModel(num_classes=10).to(device)
    
    # Get a sample input
    inputs, _ = next(iter(test_loader))
    inputs = inputs.to(device)
    
    # Profile the model
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            _ = profile_model(inputs)
    
    # Print profiling results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Enhancement 5: Pruning to reduce model size
# Apply pruning to the efficient model
print("\nApplying model pruning...")

# Create a copy of the efficient model for pruning
pruned_model = EfficientModel(num_classes=10).to(device)

# Apply global pruning
parameters_to_prune = []
for name, module in pruned_model.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        parameters_to_prune.append((module, 'weight'))

# Global pruning
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,  # Prune 20% of connections
)

# Remove pruning masks to make the pruning permanent
for module, param_name in parameters_to_prune:
    prune.remove(module, param_name)

# Count parameters after pruning
pruned_params = count_parameters(pruned_model)
print(f"Original model parameters: {efficient_params:,}")
print(f"Pruned model parameters: {pruned_params:,}")
print(f"Parameter reduction: {(efficient_params - pruned_params) / efficient_params * 100:.2f}%")

# Measure inference time for pruned model
pruned_inference_time = measure_inference_time(pruned_model, test_loader, device)
print(f"Original model inference time per batch: {efficient_inference_time:.4f} seconds")
print(f"Pruned model inference time per batch: {pruned_inference_time:.4f} seconds")

# Compare all optimization techniques
optimization_techniques = ['Original', 'Efficient Architecture', 'Quantized', 'Pruned']
model_sizes = [large_params, efficient_params, quantized_params, pruned_params]
inference_times = [large_inference_time, efficient_inference_time, quantized_inference_time, pruned_inference_time]

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.bar(optimization_techniques, model_sizes)
plt.title('Model Parameter Count')
plt.ylabel('Number of Parameters')
plt.yscale('log')

plt.subplot(1, 2, 2)
plt.bar(optimization_techniques, inference_times)
plt.title('Inference Time per Batch')
plt.ylabel('Time (seconds)')

plt.tight_layout()
plt.show()
```

## Project: Regression with PyTorch

Now, let's implement a project that demonstrates the PyTorch basics we've covered:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Generate synthetic regression data
X, y = make_regression(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    noise=0.1,
    random_state=42
)

# Reshape y to be a column vector
y = y.reshape(-1, 1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2 of original data

# Standardize the features
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_val = scaler_y.transform(y_val)
y_test = scaler_y.transform(y_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the regression model
class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        x = torch.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.batch_norm3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# Hyperparameters
input_size = X.shape[1]
hidden_size = 64
output_size = 1
learning_rate = 0.001
num_epochs = 100

# Initialize the model
model = RegressionModel(input_size, hidden_size, output_size)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using device: {device}")

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate statistics
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item()
        
        epoch_val_loss = running_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        # Update learning rate
        scheduler.step(epoch_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
    
    return train_losses, val_losses

# Train the model
print("Training the regression model:")
train_losses, val_losses = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs
)

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model on the test set
model.eval()
test_loss = 0.0
predictions = []
actuals = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        
        # Store predictions and actuals for visualization
        predictions.extend(outputs.cpu().numpy())
        actuals.extend(targets.cpu().numpy())

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4f}')

# Convert predictions and actuals back to original scale
predictions = scaler_y.inverse_transform(np.array(predictions))
actuals = scaler_y.inverse_transform(np.array(actuals))

# Calculate R-squared
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

r2 = r_squared(actuals, predictions)
print(f'R-squared: {r2:.4f}')

# Visualize predictions vs. actuals
plt.figure(figsize=(10, 6))
plt.scatter(actuals, predictions, alpha=0.5)
plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.grid(True)
plt.show()

# Visualize residuals
residuals = actuals - predictions
plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)
plt.show()

# Save the model
torch.save(model.state_dict(), 'regression_model.pth')
print("Model saved to regression_model.pth")

# Load the model
new_model = RegressionModel(input_size, hidden_size, output_size)
new_model.load_state_dict(torch.load('regression_model.pth'))
new_model = new_model.to(device)
print("Model loaded from regression_model.pth")

# Make predictions on new data
# Create some new data points
new_data = np.random.randn(5, X.shape[1])
new_data_scaled = scaler_X.transform(new_data)
new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32).to(device)

# Make predictions
new_model.eval()
with torch.no_grad():
    predictions_scaled = new_model(new_data_tensor)
    predictions = scaler_y.inverse_transform(predictions_scaled.cpu().numpy())

print("\nPredictions for new data:")
for i, pred in enumerate(predictions):
    print(f"Data point {i+1}: {pred[0]:.4f}")

# Feature importance analysis using permutation importance
def permutation_importance(model, X, y, criterion, device, n_repeats=5):
    model.eval()
    baseline_loss = 0.0
    
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        outputs = model(X_tensor)
        baseline_loss = criterion(outputs, y_tensor).item()
    
    feature_importances = []
    
    for feature_idx in range(X.shape[1]):
        X_permuted = X.copy()
        
        # Permute the feature
        X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])
        
        # Calculate loss with permuted feature
        permuted_loss = 0.0
        
        for _ in range(n_repeats):
            with torch.no_grad():
                X_permuted_tensor = torch.tensor(X_permuted, dtype=torch.float32).to(device)
                outputs = model(X_permuted_tensor)
                permuted_loss += criterion(outputs, y_tensor).item()
        
        permuted_loss /= n_repeats
        
        # Feature importance is the increase in loss when the feature is permuted
        importance = permuted_loss - baseline_loss
        feature_importances.append(importance)
    
    return np.array(feature_importances)

# Calculate feature importance
print("\nCalculating feature importance...")
feature_importances = permutation_importance(
    model, X_test, y_test, criterion, device, n_repeats=5
)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), feature_importances)
plt.title('Feature Importance')
plt.xlabel('Feature Index')
plt.ylabel('Importance (Increase in MSE)')
plt.grid(True, axis='y')
plt.show()

# Print feature importance ranking
feature_ranking = np.argsort(feature_importances)[::-1]
print("\nFeature importance ranking:")
for i, idx in enumerate(feature_ranking):
    print(f"{i+1}. Feature {idx}: {feature_importances[idx]:.4f}")

# Enhancement: Implement early stopping
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Reinitialize the model
model = RegressionModel(input_size, hidden_size, output_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize early stopping
early_stopping = EarlyStopping(patience=10, min_delta=0.001)

# Training with early stopping
print("\nTraining the model with early stopping:")
train_losses = []
val_losses = []
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate statistics
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    
    # Validation
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    
    epoch_val_loss = running_loss / len(val_loader)
    val_losses.append(epoch_val_loss)
    
    # Check for early stopping
    early_stopping(epoch_val_loss)
    
    # Save the best model
    if epoch_val_loss < early_stopping.best_loss:
        best_model_state = model.state_dict()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
    
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Load the best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Evaluate the model with early stopping
model.eval()
test_loss = 0.0
predictions = []
actuals = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        
        # Store predictions and actuals for visualization
        predictions.extend(outputs.cpu().numpy())
        actuals.extend(targets.cpu().numpy())

test_loss /= len(test_loader)
print(f'Test Loss (with early stopping): {test_loss:.4f}')

# Convert predictions and actuals back to original scale
predictions = scaler_y.inverse_transform(np.array(predictions))
actuals = scaler_y.inverse_transform(np.array(actuals))

# Calculate R-squared
r2 = r_squared(actuals, predictions)
print(f'R-squared (with early stopping): {r2:.4f}')

# Compare training with and without early stopping
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss (with Early Stopping)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(actuals, predictions, alpha=0.5)
plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values (with Early Stopping)')
plt.grid(True)

plt.tight_layout()
plt.show()
```

This project demonstrates how to implement a regression model using PyTorch, including data preprocessing, model training, evaluation, and visualization. We've also covered advanced techniques like early stopping and feature importance analysis.

## Summary

PyTorch is a powerful and flexible deep learning framework that provides essential tools for building and training neural networks. In this lesson, we've covered:

1. **Tensors**: The fundamental data structure in PyTorch, similar to NumPy arrays but with GPU acceleration support.

2. **Autograd and Computation Graphs**: PyTorch's automatic differentiation engine that enables gradient-based optimization.

3. **Neural Network Module (nn.Module)**: The base class for creating neural network models in PyTorch.

4. **Optimizers**: Algorithms that adjust model parameters to minimize the loss function.

5. **Datasets and DataLoaders**: Utilities for efficiently loading and preprocessing data for training.

We've also explored how these concepts are applied in real-world scenarios through practical examples:
- Building a simple neural network for classification
- Implementing a CNN for image classification
- Implementing an RNN for sequence classification
- Implementing a regression model with evaluation and visualization

Additionally, we've addressed common challenges in deep learning and how advancements in PyTorch have helped solve them:
- Vanishing and exploding gradients
- Overfitting
- Computational efficiency

Understanding these PyTorch basics is crucial for anyone looking to dive into deep learning, as they form the foundation for building more complex models and applications.