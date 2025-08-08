# Day 6 – Linear Algebra with PyTorch

## What is Linear Algebra and Why is it Important for Deep Learning?

Linear algebra is a branch of mathematics that deals with vectors, matrices, and linear transformations. It's the mathematical foundation of deep learning and neural networks, as these systems essentially perform a series of linear transformations on input data to produce outputs.

In deep learning, we use linear algebra to:
- Represent data as vectors and matrices
- Perform operations that transform this data
- Optimize model parameters through operations like matrix multiplication
- Understand the geometric interpretation of what neural networks are learning

## Essential Subtopics

### 1. Vectors in PyTorch

**What are vectors?**
Vectors are one-dimensional arrays of numbers that can represent data points, features, or directions in space. In deep learning, vectors often represent input features, embeddings, or model outputs.

**PyTorch Implementation:**
```python
import torch

# Creating vectors
v1 = torch.tensor([1.0, 2.0, 3.0])  # A 3D vector
v2 = torch.tensor([4.0, 5.0, 6.0])

# Vector operations
addition = v1 + v2  # Element-wise addition
subtraction = v1 - v2  # Element-wise subtraction
scalar_mult = 2.0 * v1  # Scalar multiplication
dot_product = torch.dot(v1, v2)  # Dot product
norm = torch.norm(v1)  # Vector magnitude (L2 norm)
```

**When and where we use vectors:**
- Input features in machine learning models
- Word embeddings in NLP (e.g., Word2Vec, GloVe)
- Representing intermediate activations in neural networks
- Output predictions and gradients

**Real-world examples:**
- In recommendation systems, user preferences can be represented as vectors
- In computer vision, image features extracted by CNNs are vectors
- In NLP, words or sentences are converted to vector representations (embeddings)

### 2. Matrices in PyTorch

**What are matrices?**
Matrices are two-dimensional arrays of numbers arranged in rows and columns. In deep learning, matrices can represent datasets, weight parameters, or transformations.

**PyTorch Implementation:**
```python
import torch

# Creating matrices
A = torch.tensor([[1.0, 2.0, 3.0], 
                  [4.0, 5.0, 6.0]])  # 2x3 matrix
B = torch.tensor([[7.0, 8.0], 
                  [9.0, 10.0], 
                  [11.0, 12.0]])  # 3x2 matrix

# Matrix operations
addition = A + A.T  # Addition with transpose
matrix_mult = torch.matmul(A, B)  # Matrix multiplication
elementwise_mult = A * A  # Element-wise multiplication
transpose = A.T  # Transpose
inverse = torch.inverse(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))  # Matrix inverse
determinant = torch.det(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))  # Determinant
```

**When and where we use matrices:**
- Representing datasets where rows are samples and columns are features
- Weight matrices in neural network layers
- Covariance matrices in statistics and machine learning
- Transformation matrices for geometric operations

**Real-world examples:**
- In image processing, images are represented as matrices of pixel values
- In collaborative filtering, user-item interactions are stored in a matrix
- Weight matrices in neural networks transform input features to output predictions

### 3. Tensors and Broadcasting

**What are tensors?**
Tensors are generalizations of vectors and matrices to higher dimensions. A scalar is a 0D tensor, a vector is a 1D tensor, a matrix is a 2D tensor, and so on. In deep learning, tensors are the fundamental data structure used to represent all data.

**What is broadcasting?**
Broadcasting is a mechanism that allows PyTorch to perform operations on tensors of different shapes. When possible, the smaller tensor is "broadcast" across the larger tensor to make their shapes compatible.

**PyTorch Implementation:**
```python
import torch

# Creating tensors of different dimensions
scalar = torch.tensor(3.0)  # 0D tensor (scalar)
vector = torch.tensor([1.0, 2.0, 3.0])  # 1D tensor (vector)
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2D tensor (matrix)
tensor_3d = torch.randn(2, 3, 4)  # 3D tensor

# Broadcasting examples
A = torch.ones(3, 1)  # Shape: (3, 1)
B = torch.ones(1, 3)  # Shape: (1, 3)
C = A + B  # Result shape: (3, 3) due to broadcasting

# More complex broadcasting
D = torch.ones(5, 1, 4)  # Shape: (5, 1, 4)
E = torch.ones(   3, 4)  # Shape: (    3, 4)
F = D + E  # Result shape: (5, 3, 4) due to broadcasting
```

**When and where we use tensors and broadcasting:**
- Representing multi-dimensional data like images (3D: height × width × channels)
- Batches of data (4D: batch_size × height × width × channels)
- Video data (5D: batch_size × time × height × width × channels)
- Efficiently performing operations across different dimensions without explicit loops

**Real-world examples:**
- In computer vision, a batch of RGB images is represented as a 4D tensor
- In NLP, a batch of sentences represented as word embeddings is a 3D tensor
- Broadcasting is used to efficiently add biases to all outputs in a neural network layer

### 4. Linear Transformations

**What are linear transformations?**
A linear transformation is a function between two vector spaces that preserves the operations of vector addition and scalar multiplication. In deep learning, neural network layers perform linear transformations on input data.

**PyTorch Implementation:**
```python
import torch

# Linear transformation: y = Wx + b
# W is a weight matrix, x is input vector, b is bias vector

# Define a linear transformation layer
linear = torch.nn.Linear(in_features=3, out_features=2)  # Maps 3D to 2D

# Input vector
x = torch.tensor([1.0, 2.0, 3.0])

# Apply linear transformation
y = linear(x)
print(y)  # Output is a 2D vector

# Manual implementation
W = torch.tensor([[1.0, 2.0, 3.0], 
                  [4.0, 5.0, 6.0]])  # 2x3 weight matrix
b = torch.tensor([0.5, -0.5])  # 2D bias vector

y_manual = torch.matmul(W, x) + b
print(y_manual)
```

**When and where we use linear transformations:**
- In fully connected layers of neural networks
- In the attention mechanism of transformers
- For dimensionality reduction (e.g., PCA)
- In feature extraction and engineering

**Real-world examples:**
- In image classification, linear transformations map image features to class scores
- In word embedding models like Word2Vec, linear transformations map words to vector representations
- In recommendation systems, linear transformations map user and item features to predicted ratings

### 5. Eigenvalues and Eigenvectors

**What are eigenvalues and eigenvectors?**
For a square matrix A, an eigenvector v is a non-zero vector that, when multiplied by A, yields a scalar multiple of itself. That scalar is the eigenvalue λ. In other words, Av = λv.

**PyTorch Implementation:**
```python
import torch

# Create a square matrix
A = torch.tensor([[1.0, 2.0], 
                  [2.0, 1.0]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = torch.linalg.eig(A)

# Since PyTorch returns complex tensors for eigenvalues,
# we convert to real if the imaginary part is negligible
eigenvalues = eigenvalues.real
eigenvectors = eigenvectors.real

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)

# Verify: Av = λv
v = eigenvectors[:, 0]  # First eigenvector
lambda_val = eigenvalues[0]  # First eigenvalue
Av = torch.matmul(A, v)
lambda_v = lambda_val * v
print("Av equals λv:", torch.allclose(Av, lambda_v))
```

**When and where we use eigenvalues and eigenvectors:**
- Principal Component Analysis (PCA) for dimensionality reduction
- Understanding the behavior of linear transformations
- In spectral clustering for graph-based machine learning
- In the analysis of neural network dynamics

**Real-world examples:**
- In facial recognition, PCA (which uses eigenvectors) is used for feature extraction
- In Google's PageRank algorithm, eigenvectors determine the importance of web pages
- In quantum mechanics, eigenvectors represent possible states of a system

## Applications in Deep Learning

### 1. Neural Network Operations

Neural networks are essentially a series of linear transformations followed by non-linear activations:

```python
import torch
import torch.nn as nn

# A simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(784, 128)  # Linear transformation: 784 → 128
        self.activation = nn.ReLU()        # Non-linear activation
        self.layer2 = nn.Linear(128, 10)   # Linear transformation: 128 → 10
        
    def forward(self, x):
        x = self.layer1(x)  # Matrix multiplication + bias addition
        x = self.activation(x)  # Non-linear activation
        x = self.layer2(x)  # Another linear transformation
        return x

model = SimpleNN()
input = torch.randn(1, 784)  # Random input
output = model(input)  # Forward pass
```

### 2. Gradient Computation

Linear algebra is essential for computing gradients during backpropagation:

```python
import torch

# Create a tensor with requires_grad=True to track computations
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Define a computation involving linear algebra operations
y = x.sum()  # Sum of elements
z = torch.matmul(x, x)  # Dot product of x with itself

# Compute gradients
z.backward()  # Computes dz/dx

print(x.grad)  # Prints the gradient of z with respect to x
```

### 3. Batch Processing

Linear algebra enables efficient processing of multiple data points simultaneously:

```python
import torch

# Batch of 64 images, each flattened to 784 dimensions
batch = torch.randn(64, 784)

# Weight matrix for a linear layer
weights = torch.randn(784, 128)

# Bias vector
bias = torch.randn(128)

# Apply linear transformation to the entire batch
output = torch.matmul(batch, weights) + bias  # Shape: (64, 128)
```

## Common Problems and Advancements

### 1. Problem: Numerical Instability

**Issue:**
When performing linear algebra operations, especially with very large or very small numbers, we can encounter numerical instability issues like overflow, underflow, or loss of precision.

**How advancements solved this:**
- **Normalization techniques**: Batch Normalization, Layer Normalization help stabilize the values flowing through networks
- **Precision control**: Using appropriate numerical precision (float32, float16) based on the requirements
- **Numerically stable implementations**: Libraries like PyTorch use optimized algorithms for operations like matrix inversion and eigenvalue computation

```python
# Example of batch normalization
batch_norm = nn.BatchNorm1d(num_features=128)
normalized_output = batch_norm(output)  # Stabilizes the values
```

### 2. Problem: Computational Complexity

**Issue:**
Matrix multiplication has a time complexity of O(n³) for square matrices of size n, which becomes computationally expensive for large matrices.

**How advancements solved this:**
- **GPU acceleration**: Leveraging parallel processing capabilities of GPUs
- **Strassen's algorithm**: Reduces complexity to approximately O(n^2.807)
- **Distributed computing**: Splitting computations across multiple devices
- **Approximation methods**: Using techniques like low-rank approximation

```python
# Check if GPU is available and move tensors to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch = batch.to(device)
weights = weights.to(device)

# Now the matrix multiplication will be accelerated on GPU if available
output = torch.matmul(batch, weights) + bias
```

### 3. Problem: Memory Constraints

**Issue:**
Large tensors and deep networks require significant memory, which can exceed available GPU memory.

**How advancements solved this:**
- **Gradient checkpointing**: Trading computation for memory by recomputing activations during backward pass
- **Mixed precision training**: Using lower precision (float16) for most operations while maintaining float32 where needed
- **Model parallelism**: Distributing different parts of the model across multiple devices
- **Memory-efficient optimizers**: Techniques like 8-bit Adam

```python
# Example of gradient checkpointing
from torch.utils.checkpoint import checkpoint

class MemoryEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, 10)
        
    def forward(self, x):
        x = self.layer1(x)
        # Use checkpointing for memory-intensive layers
        x = checkpoint(self.layer2, x)
        x = self.layer3(x)
        return x
```

### 4. Problem: Vanishing/Exploding Gradients

**Issue:**
In deep networks, gradients can become very small (vanish) or very large (explode) during backpropagation, making training unstable or impossible.

**How advancements solved this:**
- **Careful initialization**: Techniques like Xavier/Glorot and He initialization
- **Normalization layers**: Batch Normalization, Layer Normalization
- **Gradient clipping**: Limiting the maximum value of gradients
- **Residual connections**: Allowing gradients to flow directly through skip connections

```python
# Example of gradient clipping
model = SimpleNN()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# During training
for inputs, targets in dataloader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    
    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
```

## Project: PyTorch Matrix Calculator

Now, let's implement a simple matrix calculator using PyTorch that demonstrates the linear algebra concepts we've covered:

```python
import torch

class MatrixCalculator:
    def __init__(self):
        pass
    
    def add(self, A, B):
        """Matrix addition"""
        return A + B
    
    def subtract(self, A, B):
        """Matrix subtraction"""
        return A - B
    
    def multiply(self, A, B):
        """Matrix multiplication"""
        return torch.matmul(A, B)
    
    def transpose(self, A):
        """Matrix transpose"""
        return A.T
    
    def inverse(self, A):
        """Matrix inverse (for square matrices)"""
        return torch.inverse(A)
    
    def determinant(self, A):
        """Matrix determinant (for square matrices)"""
        return torch.det(A)
    
    def eigen(self, A):
        """Eigenvalues and eigenvectors (for square matrices)"""
        eigenvalues, eigenvectors = torch.linalg.eig(A)
        return eigenvalues.real, eigenvectors.real
    
    def solve(self, A, b):
        """Solve linear system Ax = b"""
        return torch.linalg.solve(A, b)
    
    def norm(self, A, p=None):
        """Matrix norm"""
        return torch.norm(A, p=p)
    
    def rank(self, A):
        """Matrix rank"""
        return torch.linalg.matrix_rank(A)

# Example usage
calculator = MatrixCalculator()

# Create matrices
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
b = torch.tensor([9.0, 10.0])

# Perform operations
print("A + B:")
print(calculator.add(A, B))

print("\nA × B:")
print(calculator.multiply(A, B))

print("\nA^T:")
print(calculator.transpose(A))

print("\nA^-1:")
print(calculator.inverse(A))

print("\nDeterminant of A:")
print(calculator.determinant(A))

print("\nEigenvalues and eigenvectors of A:")
eigenvalues, eigenvectors = calculator.eigen(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)

print("\nSolution to Ax = b:")
print(calculator.solve(A, b))

print("\nFrobenius norm of A:")
print(calculator.norm(A))

print("\nRank of A:")
print(calculator.rank(A))
```

This matrix calculator demonstrates the fundamental linear algebra operations that form the building blocks of deep learning algorithms. Understanding these operations and their properties is crucial for developing a deep intuition about how neural networks work and how to optimize them.

## Summary

Linear algebra is the mathematical foundation of deep learning. In this lesson, we've covered:

1. **Vectors and their operations**: Essential for representing data points and features
2. **Matrices and their operations**: Used for representing datasets and weight parameters
3. **Tensors and broadcasting**: Multi-dimensional arrays that are the core data structure in PyTorch
4. **Linear transformations**: The fundamental operation performed by neural network layers
5. **Eigenvalues and eigenvectors**: Important for understanding the behavior of linear transformations

We've also explored how these concepts are applied in deep learning, common problems encountered, and how advancements in the field have addressed these challenges.

The matrix calculator project provides a practical implementation of these concepts using PyTorch, demonstrating how these mathematical operations form the building blocks of deep learning algorithms.

Understanding linear algebra is crucial for anyone looking to master deep learning, as it provides the mathematical framework for understanding how neural networks process and transform data.