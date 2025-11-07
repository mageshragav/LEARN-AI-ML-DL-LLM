# Day 19 – Saving & Loading Models

## torch.save()

### What is it?

`torch.save()` is PyTorch's primary function for serializing and saving objects to disk. It can save various PyTorch objects including model parameters, entire models, optimizer states, and other Python objects like dictionaries or lists. The function uses Python's `pickle` module under the hood, making it versatile for saving complex data structures.

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from datetime import datetime

# Define a sample neural network
class SampleModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super(SampleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create model instance
model = SampleModel()

# Method 1: Save entire model (architecture + parameters)
def save_entire_model(model, filepath):
    """Save the entire model including architecture"""
    torch.save(model, filepath)
    print(f"Entire model saved to {filepath}")

# Method 2: Save only the state dictionary (recommended)
def save_state_dict(model, filepath):
    """Save only the model's state dictionary"""
    torch.save(model.state_dict(), filepath)
    print(f"Model state dictionary saved to {filepath}")

# Method 3: Save multiple objects
def save_multiple_objects(model, optimizer, epoch, loss, filepath):
    """Save model, optimizer, epoch, and loss together"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

# Example usage
model = SampleModel()

# Save entire model
save_entire_model(model, 'entire_model.pth')

# Save state dictionary
save_state_dict(model, 'model_weights.pth')

# Save checkpoint with optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
save_multiple_objects(model, optimizer, epoch=10, loss=0.5, filepath='checkpoint.pth')

print("All save methods demonstrated!")
```

### When and where we use torch.save()

- **Model persistence**: Saving trained models for later use
- **Checkpointing**: Saving training progress to resume later
- **Model sharing**: Distributing trained models to others
- **Experiment tracking**: Saving different model versions for comparison
- **Production deployment**: Storing models for inference in production systems

```python
# When to use torch.save() - Training checkpoint example
def save_training_checkpoint(model, optimizer, epoch, loss, is_best=False):
    """Save training checkpoint with metadata"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save current checkpoint
    torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
    
    # Save best model if this is the best checkpoint
    if is_best:
        torch.save(checkpoint, 'best_model_checkpoint.pth')
    
    print(f"Checkpoint saved for epoch {epoch}")

# Example usage in training loop
model = SampleModel()
optimizer = optim.Adam(model.parameters())

# Simulate saving during training
for epoch in range(5):
    # ... training code here ...
    loss = 1.0 / (epoch + 1)  # Simulated decreasing loss
    save_training_checkpoint(model, optimizer, epoch, loss, is_best=(epoch == 0))
```

### Real-world examples

- **Model versioning**: Keeping different versions of models in production
- **Transfer learning**: Saving pre-trained models for fine-tuning
- **Model compression**: Saving pruned or quantized models
- **Multi-stage training**: Saving intermediate results for complex pipelines

```python
# Real-world example: Model versioning system
class ModelVersionManager:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def save_model_version(self, model, version, metadata=None):
        """Save model with version number and metadata"""
        filepath = os.path.join(self.model_dir, f'model_v{version}.pth')
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        torch.save(save_dict, filepath)
        print(f"Model version {version} saved to {filepath}")
        return filepath
    
    def list_versions(self):
        """List all saved model versions"""
        versions = []
        for file in os.listdir(self.model_dir):
            if file.startswith('model_v') and file.endswith('.pth'):
                version = file.split('_v')[1].split('.')[0]
                versions.append(version)
        return sorted(versions)

# Usage example
version_manager = ModelVersionManager()
model = SampleModel()

# Save different versions
version_manager.save_model_version(model, '1.0', {'accuracy': 0.95, 'dataset': 'MNIST'})
version_manager.save_model_version(model, '1.1', {'accuracy': 0.97, 'dataset': 'MNIST', 'augmentation': True})

print(f"Available versions: {version_manager.list_versions()}")
```

### Problems and Advancements

**Problem**: Saving entire models can be inefficient and may break with code changes.
**Advancement**: State dictionary approach is more flexible and portable.

```python
# Problem: Saving entire model can cause issues
class OldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        # Old implementation had additional attributes that might break

# Solution: Use state_dict for better compatibility
def save_model_safely(model, filepath):
    """Save model state in a way that's more robust to code changes"""
    torch.save({
        'state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'architecture': str(model)
    }, filepath)

def load_model_safely(model_class, filepath):
    """Load model with error handling"""
    checkpoint = torch.load(filepath)
    
    # Create new model instance
    model = model_class()
    
    # Load state dict
    model.load_state_dict(checkpoint['state_dict'])
    return model

# Example with error handling
def safe_save_load_example():
    model = SampleModel()
    
    # Save safely
    save_model_safely(model, 'safe_model.pth')
    
    # Load safely
    loaded_model = load_model_safely(SampleModel, 'safe_model.pth')
    print("Model loaded safely using state dict approach")
    
    return loaded_model

safe_model = safe_save_load_example()
```

## state_dict

### What is it?

`state_dict` is a Python dictionary containing the learnable parameters (weights and biases) of a PyTorch model. It maps each layer's parameter name to its corresponding tensor. This dictionary is crucial for saving and loading model weights, transferring learning between models, and inspecting model parameters.

### PyTorch Implementation

```python
# Understanding state_dict
def demonstrate_state_dict():
    model = SampleModel()
    
    # Get the state dictionary
    state_dict = model.state_dict()
    
    print("Model state dictionary keys:")
    for key, tensor in state_dict.items():
        print(f"{key}: {tensor.shape}")
    
    # Access specific parameters
    print(f"\nFirst layer weights shape: {state_dict['fc1.weight'].shape}")
    print(f"First layer bias shape: {state_dict['fc1.bias'].shape}")
    
    # Show parameter values
    print(f"First few weights of fc1: {state_dict['fc1.weight'][:3, :3]}")
    
    return state_dict

state_dict = demonstrate_state_dict()

# Working with state_dict for transfer learning
def transfer_learning_example():
    # Source model
    source_model = SampleModel(input_size=784, hidden_size=128, output_size=10)
    
    # Target model (different output size)
    target_model = SampleModel(input_size=784, hidden_size=128, output_size=5)
    
    # Get source model's state dict
    source_state_dict = source_model.state_dict()
    target_state_dict = target_model.state_dict()
    
    # Transfer compatible layers
    for key in target_state_dict.keys():
        if key in source_state_dict and target_state_dict[key].shape == source_state_dict[key].shape:
            target_state_dict[key] = source_state_dict[key]
            print(f"Transferred {key}")
    
    # Load the updated state dict
    target_model.load_state_dict(target_state_dict, strict=False)
    print("Transfer learning completed!")
    
    return target_model

transferred_model = transfer_learning_example()
```

### When and where we use state_dict

- **Model saving/loading**: The recommended way to save/load model weights
- **Transfer learning**: Transferring pre-trained weights to new models
- **Model inspection**: Examining learned parameters
- **Parameter initialization**: Setting specific initial values
- **Model comparison**: Comparing weights between different models

```python
# When to use state_dict - Model inspection example
def inspect_model_parameters(model):
    """Inspect model parameters using state_dict"""
    state_dict = model.state_dict()
    
    total_params = 0
    trainable_params = 0
    
    for name, param in state_dict.items():
        num_params = param.numel()
        total_params += num_params
        
        print(f"{name}: {param.shape}, {num_params:,} parameters")
        
        # Check if parameter is trainable
        if param.requires_grad:
            trainable_params += num_params
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Memory usage: {total_params * 4 / 1024 / 1024:.2f} MB (assuming float32)")

# Example usage
model = SampleModel()
inspect_model_parameters(model)
```

### Real-world examples

- **Pre-trained model loading**: Loading weights from models like ResNet, BERT
- **Fine-tuning**: Adapting pre-trained models to new tasks
- **Model ensembling**: Combining weights from multiple models
- **Parameter analysis**: Studying learned representations

```python
# Real-world example: Pre-trained model loading
def load_pretrained_weights(model, pretrained_path):
    """Load pre-trained weights with proper error handling"""
    try:
        # Load the checkpoint
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if model was trained with DataParallel
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '') if key.startswith('module.') else key
            new_state_dict[new_key] = value
        
        # Load state dict
        model.load_state_dict(new_state_dict, strict=False)
        print("Pre-trained weights loaded successfully!")
        
    except FileNotFoundError:
        print(f"Pretrained model file {pretrained_path} not found")
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
    
    return model

# Example usage
model = SampleModel()
# load_pretrained_weights(model, 'pretrained_model.pth')  # Uncomment when file exists
```

### Problems and Advancements

**Problem**: Mismatched layer names or shapes when loading state_dict.
**Advancement**: Flexible loading with `strict=False` and name mapping.

```python
# Problem: State dict key mismatch
class ModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 1)

class ModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(10, 5)  # Different name
        self.classifier = nn.Linear(5, 1)  # Different name

# Solution: Key mapping for compatibility
def load_with_key_mapping(source_state_dict, target_model):
    """Load state dict with key name mapping"""
    target_state_dict = target_model.state_dict()
    
    # Define key mappings
    key_mapping = {
        'layer1.weight': 'backbone.weight',
        'layer1.bias': 'backbone.bias',
        'layer2.weight': 'classifier.weight',
        'layer2.bias': 'classifier.bias'
    }
    
    # Map keys
    mapped_state_dict = {}
    for source_key, value in source_state_dict.items():
        target_key = key_mapping.get(source_key, source_key)
        if target_key in target_state_dict:
            mapped_state_dict[target_key] = value
    
    # Load with strict=False to allow partial loading
    target_model.load_state_dict(mapped_state_dict, strict=False)
    print("State dict loaded with key mapping")
    
    return target_model

# Example
model_v1 = ModelV1()
model_v2 = ModelV2()

# Transfer weights using key mapping
model_v2 = load_with_key_mapping(model_v1.state_dict(), model_v2)
```

## 📌 Project: Streamlit app with trained model

```python
# Create a complete Streamlit app with model loading
import streamlit as st
import numpy as np
from PIL import Image
import io

# First, let's create and save a model for the Streamlit app
class StreamlitModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super(StreamlitModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Function to create and save a demo model
def create_and_save_demo_model():
    """Create and save a demo model for the Streamlit app"""
    model = StreamlitModel()
    
    # Save the model's state dict
    torch.save(model.state_dict(), 'demo_model.pth')
    print("Demo model saved as 'demo_model.pth'")
    
    return model

# Create the demo model
demo_model = create_and_save_demo_model()

# Now create the Streamlit app code as a string
streamlit_app_code = '''
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io

# Define the same model architecture
class StreamlitModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super(StreamlitModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Function to load the model
@st.cache_resource
def load_model():
    """Load the trained model"""
    model = StreamlitModel()
    
    # Try to load the model
    try:
        # Load the state dict
        state_dict = torch.load("demo_model.pth", map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()  # Set to evaluation mode
        st.success("Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please ensure 'demo_model.pth' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to preprocess image
def preprocess_image(image):
    """Preprocess the uploaded image for the model"""
    # Convert to grayscale if needed
    if image.mode != "L":
        image = image.convert("L")
    
    # Resize to 28x28 (for MNIST-like model)
    image = image.resize((28, 28))
    
    # Convert to numpy array and normalize
    img_array = np.array(image)
    img_array = img_array.astype(np.float32) / 255.0
    
    # Invert if needed (make digit white on black background)
    if np.mean(img_array) > 0.5:  # If background is lighter
        img_array = 1 - img_array
    
    # Add batch and channel dimensions
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 28, 28]
    
    return img_tensor

# Main Streamlit app
def main():
    st.title("🧠 PyTorch Model Demo with Streamlit")
    st.markdown("---")
    
    # Load model
    model = load_model()
    
    if model is not None:
        st.success("Model is ready for predictions!")
        
        # File uploader for image
        uploaded_file = st.file_uploader(
            "Upload an image for prediction", 
            type=["png", "jpg", "jpeg"],
            help="Upload a grayscale image (ideally 28x28 pixels)"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess image
            with st.spinner("Processing image..."):
                processed_image = preprocess_image(image)
            
            # Make prediction
            with st.spinner("Making prediction..."):
                with torch.no_grad():
                    output = model(processed_image)
                    probabilities = torch.exp(output).squeeze().numpy()
                    predicted_class = torch.argmax(output, dim=1).item()
            
            # Display results
            st.subheader("Prediction Results")
            st.write(f"**Predicted Class:** {predicted_class}")
            st.write(f"**Confidence:** {probabilities[predicted_class]:.3f}")
            
            # Show all probabilities
            st.subheader("Class Probabilities")
            prob_dict = {f"Class {i}": prob for i, prob in enumerate(probabilities)}
            st.bar_chart(prob_dict)
            
            # Show probability breakdown
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Top 3 Predictions:**")
                top3_indices = np.argsort(probabilities)[-3:][::-1]
                for i, idx in enumerate(top3_indices):
                    st.write(f"{i+1}. Class {idx}: {probabilities[idx]:.3f}")
            
            with col2:
                st.write("**Model Confidence:**")
                confidence_percent = probabilities[predicted_class] * 100
                st.progress(float(confidence_percent / 100))
                st.write(f"{confidence_percent:.1f}%")
    
    # Model information
    st.markdown("---")
    st.subheader("Model Information")
    st.write("- This is a simple neural network trained on MNIST-like data")
    st.write("- Input: 28x28 grayscale image")
    st.write("- Output: 10 classes (digits 0-9)")
    st.write("- Architecture: Linear -> ReLU -> Dropout -> Linear -> Softmax")

if __name__ == "__main__":
    main()
'''

# Save the Streamlit app to a file
with open('streamlit_app.py', 'w') as f:
    f.write(streamlit_app_code)

print("Streamlit app created as 'streamlit_app.py'")

# Instructions for running the Streamlit app
print("\nTo run the Streamlit app:")
print("1. Install Streamlit: pip install streamlit")
print("2. Run the app: streamlit run streamlit_app.py")
print("3. The app will load the saved model and allow image predictions")

# Advanced model saving with metadata
def save_model_with_metadata(model, filepath, metadata=None):
    """Save model with additional metadata"""
    if metadata is None:
        metadata = {}
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'model_type': model.__class__.__name__,
            'input_size': getattr(model, 'input_size', None),
            'output_size': getattr(model, 'output_size', None),
            **metadata
        }
    }
    
    torch.save(save_dict, filepath)
    print(f"Model with metadata saved to {filepath}")

# Example with metadata
model_with_metadata = SampleModel()
save_model_with_metadata(
    model_with_metadata, 
    'model_with_metadata.pth',
    metadata={
        'accuracy': 0.95,
        'dataset': 'MNIST',
        'epochs_trained': 10,
        'description': 'Simple feedforward network for digit classification'
    }
)

# Load model with metadata
def load_model_with_metadata(filepath):
    """Load model and return with metadata"""
    checkpoint = torch.load(filepath)
    
    if 'model_state_dict' in checkpoint:
        # Model was saved with metadata
        model_state_dict = checkpoint['model_state_dict']
        metadata = checkpoint.get('metadata', {})
        
        # Create model instance (you'd need to know the architecture)
        # For this example, we'll create a SampleModel
        model = SampleModel()
        model.load_state_dict(model_state_dict)
        
        return model, metadata
    else:
        # Model was saved without metadata
        print("Model saved without metadata")
        model = SampleModel()
        model.load_state_dict(checkpoint)
        return model, {}

# Test loading with metadata
loaded_model, metadata = load_model_with_metadata('model_with_metadata.pth')
print(f"Loaded model with metadata: {metadata}")
```

