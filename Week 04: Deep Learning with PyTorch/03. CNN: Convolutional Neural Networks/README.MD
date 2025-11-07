# Day 18 – CNN: Convolutional Neural Networks

## Convolution

### What is it?

Convolution is a mathematical operation that applies a small matrix (called a kernel or filter) to an input image or feature map to extract features like edges, textures, or patterns. The kernel slides across the input, performing element-wise multiplication and summation at each position, creating a new representation of the input that highlights specific features.

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Basic convolution implementation
class ConvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvolutionLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        return self.conv(x)

# Example usage
def demonstrate_convolution():
    # Create a sample 5x5 input with 1 channel
    input_tensor = torch.randn(1, 1, 5, 5)
    print(f"Input shape: {input_tensor.shape}")
    
    # Create convolution layer
    conv_layer = ConvolutionLayer(1, 3, kernel_size=3, stride=1, padding=1)
    output = conv_layer(input_tensor)
    
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in conv_layer.parameters())}")
    
    return conv_layer, input_tensor, output

conv_layer, input_tensor, output = demonstrate_convolution()

# Manual convolution calculation to understand the process
def manual_convolution():
    # Simple 3x3 input
    input_data = torch.tensor([[[[1., 2., 3.],
                                [4., 5., 6.],
                                [7., 8., 9.]]]], dtype=torch.float32)
    
    # Simple 2x2 kernel
    kernel = torch.tensor([[[[1., 0.],
                           [0., 1.]]]], dtype=torch.float32)
    
    # Perform convolution manually
    result = F.conv2d(input_data, kernel, stride=1, padding=0)
    print(f"Manual convolution result: {result}")
    
    return result

manual_result = manual_convolution()
```

### When and where we use it

Convolution is used in:
- **Image processing**: Feature extraction, edge detection, texture analysis
- **Computer vision**: Object detection, image classification, segmentation
- **Time series analysis**: Pattern recognition in sequential data
- **Audio processing**: Spectrogram analysis, feature extraction

```python
# When to use convolution - Image classification example
def when_to_use_convolution():
    # For image classification task
    class ImageClassifier(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            # Convolution layers extract spatial features
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Extract low-level features
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Extract mid-level features
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # Extract high-level features
            
            # Classification layers
            self.classifier = nn.Sequential(
                nn.Linear(128 * 4 * 4, 256),  # Assuming 32x32 input -> 4x4 after convolutions
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)  # Reduce spatial dimensions
            
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2)
            
            x = x.view(x.size(0), -1)  # Flatten
            x = self.classifier(x)
            return x
    
    return ImageClassifier()

image_classifier = when_to_use_convolution()
```

### Real-world examples

- **Medical imaging**: Detecting tumors in X-rays, MRI scans
- **Autonomous vehicles**: Object detection, lane detection
- **Face recognition**: Feature extraction for identity verification
- **Satellite imagery**: Land use classification, weather prediction

```python
# Real-world example: Edge detection
def edge_detection_example():
    # Create a simple edge detection kernel
    edge_kernel = torch.tensor([[[[-1., -1., -1.],
                                 [-1.,  8., -1.],
                                 [-1., -1., -1.]]]], dtype=torch.float32)
    
    # Simulate an image with edges
    sample_image = torch.randn(1, 1, 8, 8)
    edge_detected = F.conv2d(sample_image, edge_kernel, padding=1)
    
    print(f"Edge detection applied. Output shape: {edge_detected.shape}")
    return edge_detected

edge_output = edge_detection_example()
```

### Problems and Advancements

**Problem**: Computational complexity increases with kernel size and number of channels.
**Advancement**: Separable convolutions (depthwise + pointwise) reduce parameters.

```python
# Problem: Standard convolution - many parameters
class StandardConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
    
    def forward(self, x):
        return self.conv(x)

# Solution: Depthwise separable convolution - fewer parameters
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # Depthwise convolution: apply one filter per input channel
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                  groups=in_channels, padding=1)
        # Pointwise convolution: combine channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Compare parameter counts
def compare_convolutions():
    in_channels, out_channels, kernel_size = 64, 128, 3
    input_tensor = torch.randn(1, 64, 32, 32)
    
    # Standard convolution
    std_conv = StandardConv(in_channels, out_channels, kernel_size)
    std_output = std_conv(input_tensor)
    
    # Depthwise separable convolution
    sep_conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size)
    sep_output = sep_conv(input_tensor)
    
    std_params = sum(p.numel() for p in std_conv.parameters())
    sep_params = sum(p.numel() for p in sep_conv.parameters())
    
    print(f"Standard Conv - Parameters: {std_params}, Output shape: {std_output.shape}")
    print(f"Separable Conv - Parameters: {sep_params}, Output shape: {sep_output.shape}")
    print(f"Parameter reduction: {(std_params - sep_params) / std_params * 100:.2f}%")

compare_convolutions()
```

## Pooling

### What is it?

Pooling is a down-sampling operation that reduces the spatial dimensions of feature maps while retaining the most important information. Common types include max pooling (selects maximum value) and average pooling (computes average value) within local regions.

### PyTorch Implementation

```python
# Pooling operations
class PoolingLayers(nn.Module):
    def __init__(self, pool_type='max', kernel_size=2, stride=2):
        super(PoolingLayers, self).__init__()
        if pool_type == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride)
        elif pool_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride)
        else:
            raise ValueError("pool_type must be 'max' or 'avg'")
    
    def forward(self, x):
        return self.pool(x)

# Demonstrate different pooling operations
def demonstrate_pooling():
    # Create sample input
    input_tensor = torch.randn(1, 3, 8, 8)
    print(f"Original input shape: {input_tensor.shape}")
    
    # Max pooling
    max_pool = PoolingLayers('max', kernel_size=2, stride=2)
    max_pooled = max_pool(input_tensor)
    print(f"Max pooled shape: {max_pooled.shape}")
    
    # Average pooling
    avg_pool = PoolingLayers('avg', kernel_size=2, stride=2)
    avg_pooled = avg_pool(input_tensor)
    print(f"Avg pooled shape: {avg_pooled.shape}")
    
    return max_pool, avg_pool

max_pool, avg_pool = demonstrate_pooling()

# Adaptive pooling (output size is fixed regardless of input size)
def adaptive_pooling_example():
    input_tensor = torch.randn(1, 64, 16, 16)
    
    # Adaptive pooling to fixed output size
    adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
    output = adaptive_pool(input_tensor)
    
    print(f"Adaptive pooling - Input: {input_tensor.shape}, Output: {output.shape}")
    return output

adaptive_output = adaptive_pooling_example()
```

### When and where we use it

Pooling is used to:
- **Reduce computational complexity**: Smaller feature maps require fewer operations
- **Control overfitting**: Reduces model capacity by reducing spatial dimensions
- **Achieve translation invariance**: Makes model less sensitive to small shifts
- **Build hierarchical representations**: Create multi-scale features

```python
# When to use pooling - Building hierarchical features
def hierarchical_feature_extraction():
    class HierarchicalCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            # Level 1: Low-level features
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.pool1 = nn.MaxPool2d(2)  # Halve spatial dimensions
            
            # Level 2: Mid-level features
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool2 = nn.MaxPool2d(2)  # Further reduction
            
            # Level 3: High-level features
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool3 = nn.AdaptiveAvgPool2d((1, 1))  # Fixed size regardless of input
            
            self.classifier = nn.Linear(128, num_classes)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool1(x)  # 32x32 -> 16x16
            
            x = F.relu(self.conv2(x))
            x = self.pool2(x)  # 16x16 -> 8x8
            
            x = F.relu(self.conv3(x))
            x = self.pool3(x)  # 8x8 -> 1x1
            
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    return HierarchicalCNN()

hierarchical_model = hierarchical_feature_extraction()
```

### Real-world examples

- **Image classification**: Reducing feature map size before final classification
- **Object detection**: Creating feature pyramids at different scales
- **Medical diagnosis**: Extracting multi-scale features from scans
- **Video analysis**: Processing frames at different temporal scales

```python
# Real-world example: Multi-scale feature extraction
def multi_scale_feature_extraction():
    class MultiScaleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            
            # Different pooling sizes for multi-scale features
            self.pool_small = nn.MaxPool2d(2)    # 2x2 pooling
            self.pool_medium = nn.MaxPool2d(4)   # 4x4 pooling
            self.pool_large = nn.MaxPool2d(8)    # 8x8 pooling
            
            self.classifier = nn.Linear(64 * 3, 10)  # Combine all scales
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            
            # Extract features at different scales
            feat_small = self.pool_small(x).view(x.size(0), -1)
            feat_medium = self.pool_medium(x).view(x.size(0), -1)
            feat_large = self.pool_large(x).view(x.size(0), -1)
            
            # Concatenate multi-scale features
            combined = torch.cat([feat_small, feat_medium, feat_large], dim=1)
            output = self.classifier(combined)
            return output
    
    return MultiScaleCNN()

multi_scale_model = multi_scale_feature_extraction()
```

### Problems and Advancements

**Problem**: Information loss during max pooling (only keeps maximum value).
**Advancement**: Fractional max pooling, stochastic pooling, and attention-based pooling.

```python
# Problem: Information loss in max pooling
def information_loss_example():
    # Original feature map with important details
    feature_map = torch.tensor([[[[1., 2., 3., 4.],
                                 [5., 6., 7., 8.],
                                 [9., 10., 11., 12.],
                                 [13., 14., 15., 16.]]]], dtype=torch.float32)
    
    max_pooled = F.max_pool2d(feature_map, 2)
    avg_pooled = F.avg_pool2d(feature_map, 2)
    
    print(f"Original: {feature_map[0, 0]}")
    print(f"Max pooled: {max_pooled[0, 0]}")
    print(f"Avg pooled: {avg_pooled[0, 0]}")
    print("Max pooling loses information about other values in each region")

information_loss_example()

# Solution: Attention-based pooling
class AttentionPooling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Compute attention weights
        attention_weights = self.attention(x)
        
        # Apply attention and then global average pooling
        attended_features = x * attention_weights
        pooled = F.adaptive_avg_pool2d(attended_features, (1, 1))
        return pooled

def attention_pooling_example():
    input_tensor = torch.randn(2, 64, 8, 8)
    attention_pool = AttentionPooling(64)
    output = attention_pool(input_tensor)
    print(f"Attention pooling output shape: {output.shape}")
    return output

attention_output = attention_pooling_example()
```

## Flattening

### What is it?

Flattening is the process of converting multi-dimensional feature maps (3D: height x width x channels) into a 1D vector. This is necessary to connect convolutional layers to fully connected (linear) layers for classification or regression tasks.

### PyTorch Implementation

```python
# Flattening implementation
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    
    def forward(self, x):
        # Flatten all dimensions except batch dimension
        return x.view(x.size(0), -1)

# Demonstrate flattening
def demonstrate_flattening():
    # Sample feature maps from convolutional layers
    feature_maps = torch.randn(32, 64, 7, 7)  # Batch=32, Channels=64, Height=7, Width=7
    print(f"Before flattening: {feature_maps.shape}")
    
    # Manual flattening
    flattened = feature_maps.view(feature_maps.size(0), -1)
    print(f"After manual flattening: {flattened.shape}")
    
    # Using custom flatten layer
    flatten_layer = FlattenLayer()
    flattened_custom = flatten_layer(feature_maps)
    print(f"After custom flatten layer: {flattened_custom.shape}")
    
    # Using PyTorch's built-in flatten
    flattened_builtin = torch.flatten(feature_maps, 1)  # Flatten from dimension 1 (keep batch)
    print(f"After built-in flatten: {flattened_builtin.shape}")
    
    return flattened

flattened_output = demonstrate_flattening()
```

### When and where we use it

Flattening is used:
- **Before fully connected layers**: To connect convolutional features to classification heads
- **Feature vector creation**: To create fixed-size representations
- **Input preparation**: To prepare features for traditional ML algorithms
- **Embedding creation**: To create vector representations for similarity tasks

```python
# When to use flattening - Complete CNN architecture
def complete_cnn_with_flattening():
    class CompleteCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            # Convolutional feature extractor
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            
            # Flattening layer
            self.flatten = nn.Flatten(start_dim=1)  # Keep batch dimension
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(128 * 4 * 4, 256),  # 128 channels * 4x4 spatial size
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.flatten(x)  # Flatten before fully connected layers
            x = self.classifier(x)
            return x
    
    return CompleteCNN()

complete_model = complete_cnn_with_flattening()
```

### Real-world examples

- **Image classification**: Flattening features before softmax classification
- **Object detection**: Flattening region features for bounding box classification
- **Image retrieval**: Creating flattened embeddings for similarity search
- **Transfer learning**: Flattening pre-trained features for new tasks

```python
# Real-world example: Transfer learning with flattening
def transfer_learning_example():
    class TransferLearningCNN(nn.Module):
        def __init__(self, pretrained_features, num_classes=1000):
            super().__init__()
            # Use pre-trained convolutional features
            self.features = pretrained_features
            
            # Add custom classification head
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # Ensure consistent size
                nn.Flatten(start_dim=1),       # Flatten for linear layer
                nn.Linear(512, 128),           # Reduce dimensions
                nn.ReLU(),
                nn.Linear(128, num_classes)    # Final classification
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    
    # Simulate pre-trained features (like ResNet backbone)
    pretrained_backbone = nn.Sequential(
        nn.Conv2d(3, 64, 7, stride=2, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(3, stride=2, padding=1),
        nn.Conv2d(64, 512, 3, padding=1)
    )
    
    return TransferLearningCNN(pretrained_backbone)

transfer_model = transfer_learning_example()
```

### Problems and Advancements

**Problem**: Flattening loses spatial relationships and hierarchical structure.
**Advancement**: Global pooling, attention mechanisms, and transformer-based approaches.

```python
# Problem: Spatial information loss during flattening
def spatial_information_loss():
    # Feature maps with spatial patterns
    feature_maps = torch.randn(1, 4, 8, 8)  # 4 channels, 8x8 spatial
    
    # Traditional flattening loses spatial structure
    flattened = feature_maps.view(1, -1)
    print(f"Flattened shape: {flattened.shape}")
    print("Spatial relationships are lost in traditional flattening")

spatial_information_loss()

# Solution: Global pooling preserves some spatial information
class GlobalPoolingFlatten(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
    
    def forward(self, x):
        # Global average pooling
        avg_pooled = self.global_avg_pool(x)
        # Global max pooling
        max_pooled = self.global_max_pool(x)
        
        # Combine both (concatenate)
        combined = torch.cat([avg_pooled, max_pooled], dim=1)
        return combined.view(x.size(0), -1)

def global_pooling_flatten_example():
    input_tensor = torch.randn(2, 64, 8, 8)
    global_pool_flatten = GlobalPoolingFlatten(64)
    output = global_pool_flatten(input_tensor)
    print(f"Global pooling flatten output: {output.shape}")
    return output

global_output = global_pooling_flatten_example()
```

## 🧪 Exercise: CNN on MNIST

```python
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def mnist_cnn_exercise():
    # Define CNN architecture for MNIST
    class MNISTCNN(nn.Module):
        def __init__(self):
            super(MNISTCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
            self.fc1 = nn.Linear(64 * 14 * 14, 128)  # After pooling: 28/2=14
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.pool(x)  # 28x28 -> 14x14
            x = self.dropout1(x)
            x = torch.flatten(x, 1)  # Flattening
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    
    return MNISTCNN()

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Training function
def train_mnist_cnn():
    model = mnist_cnn_exercise()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
        
        if batch_idx >= 500:  # Limit for demonstration
            break
    
    return model

# Run the exercise
print("Training MNIST CNN...")
trained_model = train_mnist_cnn()
print("MNIST CNN training completed!")
```

## 📌 Project: Handwritten Digit Recognition

```python
class DigitRecognitionSystem:
    def __init__(self, model_path=None):
        self.model = self._build_model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
    
    def _build_model(self):
        class DigitCNN(nn.Module):
            def __init__(self):
                super().__init__()
                # Feature extraction layers
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(64),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(0.25),
                    
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(128),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(256),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(0.25)
                )
                
                # Classification head
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256 * 7 * 7, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, 10)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        return DigitCNN()
    
    def train(self, train_loader, test_loader, epochs=10):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Evaluate on test set
            test_accuracy = self._evaluate(test_loader)
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%')
    
    def _evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return 100 * correct / total
    
    def predict_digit(self, image_tensor):
        """Predict digit from input image"""
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor.unsqueeze(0))
            predicted = output.argmax(dim=1, keepdim=True)
            confidence = torch.softmax(output, dim=1).max().item()
        return predicted.item(), confidence
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

# Initialize and train the digit recognition system
digit_system = DigitRecognitionSystem()

# Train the model
print("Starting digit recognition system training...")
digit_system.train(train_loader, test_loader, epochs=5)
print("Training completed!")

# Example prediction
sample_image, sample_label = test_dataset[0]
predicted_digit, confidence = digit_system.predict_digit(sample_image)
print(f"Predicted: {predicted_digit}, Actual: {sample_label}, Confidence: {confidence:.2f}")
```