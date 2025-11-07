# Day 46 – Model Exporting

## What is Model Exporting?

Model exporting is the process of converting a trained machine learning model from its native framework format (like PyTorch, TensorFlow, or JAX) into a standardized format that can be used in different environments, deployment scenarios, or inference engines. This process is crucial for deploying models to production, edge devices, or cloud services.

## Essential Subtopics

### 1. PyTorch to ONNX Conversion

**What it is:**
ONNX (Open Neural Network Exchange) is an open format for representing machine learning models. Converting PyTorch models to ONNX allows them to be run on various ONNX-compatible runtimes and hardware accelerators, independent of the original training framework.

**Explanation:**
The PyTorch to ONNX conversion process involves:
1. Tracing or scripting the PyTorch model to capture its operations
2. Exporting the captured computation graph to ONNX format
3. Verifying the exported model produces the same outputs as the original
4. Optimizing the ONNX model for inference

This conversion enables models to be deployed in environments where PyTorch might not be available or optimal, such as on edge devices or with specialized inference accelerators.

**Coding Example:**
```python
import torch
import torch.nn as nn
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from typing import Tuple

# Define a simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class PyTorchToONNXConverter:
    """
    A class to convert PyTorch models to ONNX format.
    """
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...]):
        """
        Initialize the converter.
        
        Args:
            model: PyTorch model to convert
            input_shape: Shape of the input tensor
        """
        self.model = model
        self.input_shape = input_shape
        self.model.eval()  # Set model to evaluation mode
    
    def export_to_onnx(self, output_path: str, opset_version: int = 11) -> None:
        """
        Export the PyTorch model to ONNX format.
        
        Args:
            output_path: Path to save the ONNX model
            opset_version: ONNX opset version to use
        """
        # Create a dummy input for tracing
        dummy_input = torch.randn(self.input_shape)
        
        # Export the model
        torch.onnx.export(
            self.model,               # Model to export
            dummy_input,             # Model input
            output_path,             # File path to save the ONNX model
            export_params=True,       # Store the trained parameter weights inside the model file
            opset_version=opset_version,  # ONNX opset version
            do_constant_folding=True,  # Optimize constant folding
            input_names=['input'],   # Model's input names
            output_names=['output'], # Model's output names
            dynamic_axes={'input': {0: 'batch_size'},  # Variable length axes
                         'output': {0: 'batch_size'}}
        )
        
        print(f"Model exported to {output_path}")
    
    def verify_onnx_model(self, onnx_path: str) -> bool:
        """
        Verify that the ONNX model produces the same outputs as the PyTorch model.
        
        Args:
            onnx_path: Path to the ONNX model
            
        Returns:
            True if verification passes, False otherwise
        """
        # Load the ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(onnx_path)
        
        # Create test input
        test_input = torch.randn(self.input_shape)
        
        # Get PyTorch model output
        with torch.no_grad():
            pytorch_output = self.model(test_input)
        
        # Get ONNX model output
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]
        
        # Convert to numpy for comparison
        pytorch_output_np = pytorch_output.numpy()
        
        # Compare outputs
        are_close = np.allclose(pytorch_output_np, ort_output, rtol=1e-3, atol=1e-5)
        
        if are_close:
            print("ONNX model verification passed!")
        else:
            print("ONNX model verification failed!")
            print(f"Max difference: {np.max(np.abs(pytorch_output_np - ort_output))}")
        
        return are_close
    
    def optimize_onnx_model(self, input_path: str, output_path: str) -> None:
        """
        Optimize the ONNX model.
        
        Args:
            input_path: Path to the input ONNX model
            output_path: Path to save the optimized ONNX model
        """
        # Load the model
        model = onnx.load(input_path)
        
        # Apply optimizations
        from onnx import optimizer
        
        # List of optimizations to apply
        all_passes = optimizer.get_all_passes()
        passes = [
            'eliminate_unused_initializer',  # Remove unused initializers
            'fuse_add_bias_into_conv',       # Fuse add bias into convolution
            'fuse_bn_into_conv',             # Fuse batch normalization into convolution
            'fuse_conv_bn_add_relu',         # Fuse conv, batch norm, add, and relu
            'fuse_consecutive_concats',      # Fuse consecutive concat operations
            'fuse_consecutive_reduce_unsqueeze',  # Fuse consecutive reduce unsqueeze
            'fuse_consecutive_squeezes',     # Fuse consecutive squeeze operations
            'fuse_consecutive_transposes',   # Fuse consecutive transpose operations
            'fuse_matmul_add_bias',          # Fuse matmul and add bias
            'fuse_pad_into_conv',            # Fuse pad into convolution
            'gelu_approximation',           # Approximate GELU activation
            'eliminate_deadend',             # Eliminate deadend operations
            'eliminate_nop_transpose',       # Eliminate no-op transpose operations
            'eliminate_nop_pad',             # Eliminate no-op pad operations
            'eliminate_identity',            # Eliminate identity operations
            'eliminate_deadend',             # Eliminate deadend operations
            'extract_constant_to_initializer',  # Extract constants to initializers
        ]
        
        # Apply optimizations
        optimized_model = optimizer.optimize(model, passes)
        
        # Save the optimized model
        onnx.save(optimized_model, output_path)
        
        print(f"Optimized model saved to {output_path}")

# Example usage
def pytorch_to_onnx_example():
    """Example of converting a PyTorch model to ONNX."""
    # Create a simple model
    model = SimpleModel(input_size=10, hidden_size=20, output_size=5)
    
    # Create converter
    converter = PyTorchToONNXConverter(model, input_shape=(1, 10))
    
    # Export to ONNX
    converter.export_to_onnx("simple_model.onnx")
    
    # Verify the ONNX model
    is_valid = converter.verify_onnx_model("simple_model.onnx")
    
    if is_valid:
        # Optimize the ONNX model
        converter.optimize_onnx_model("simple_model.onnx", "simple_model_optimized.onnx")
        
        # Verify the optimized model
        converter.verify_onnx_model("simple_model_optimized.onnx")

# Run the example
# pytorch_to_onnx_example()
```

### 2. Dockerizing Models

**What it is:**
Dockerizing models is the process of packaging a model along with its dependencies, runtime environment, and inference code into a Docker container. This creates a portable, consistent, and scalable deployment unit that can run on any system with Docker installed.

**Explanation:**
The Dockerization process involves:
1. Creating a Dockerfile that defines the container environment
2. Installing necessary dependencies and runtime libraries
3. Copying the model and inference code into the container
4. Exposing appropriate ports for API access
5. Building and testing the container

Docker containers provide isolation, reproducibility, and scalability for model deployment, making it easier to manage models in production environments.

**Coding Example:**
```python
import os
import subprocess
import json
from typing import Dict, List, Any

class ModelDockerizer:
    """
    A class to create Docker containers for machine learning models.
    """
    def __init__(self, model_path: str, model_type: str = "onnx"):
        """
        Initialize the dockerizer.
        
        Args:
            model_path: Path to the model file
            model_type: Type of the model (onnx, pytorch, tensorflow, etc.)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model_name = os.path.splitext(os.path.basename(model_path))[0]
        self.image_name = f"{self.model_name}-service"
        self.container_name = f"{self.model_name}-container"
    
    def create_dockerfile(self, base_image: str = "python:3.9-slim", 
                         requirements: List[str] = None,
                         port: int = 8000) -> str:
        """
        Create a Dockerfile for the model service.
        
        Args:
            base_image: Base Docker image to use
            requirements: List of Python requirements
            port: Port to expose
            
        Returns:
            Path to the created Dockerfile
        """
        # Default requirements
        if requirements is None:
            requirements = [
                "fastapi",
                "uvicorn",
                "onnxruntime",
                "numpy",
                "pydantic"
            ]
        
        # Create Dockerfile content
        dockerfile_content = f"""FROM {base_image}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application code
COPY {os.path.basename(self.model_path)} .
COPY app.py .

# Expose port
EXPOSE {port}

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "{port}"]
"""
        
        # Write Dockerfile
        dockerfile_path = "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)
        
        # Create requirements.txt
        with open("requirements.txt", "w") as f:
            f.write("\n".join(requirements))
        
        print(f"Dockerfile created at {dockerfile_path}")
        return dockerfile_path
    
    def create_app_file(self, port: int = 8000) -> str:
        """
        Create a FastAPI application file for serving the model.
        
        Args:
            port: Port to run the service on
            
        Returns:
            Path to the created application file
        """
        app_content = f"""from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="{self.model_name} Service", version="1.0.0")

# Load the ONNX model
try:
    session = ort.InferenceSession("{os.path.basename(self.model_path)}")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {{e}}")
    raise

# Define input model
class ModelInput(BaseModel):
    data: List[List[float]]
    
    class Config:
        schema_extra = {{
            "example": {{
                "data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
            }}
        }}

# Define output model
class ModelOutput(BaseModel):
    prediction: List[float]
    
    class Config:
        schema_extra = {{
            "example": {{
                "prediction": [0.1, 0.2, 0.3, 0.4, 0.5]
            }}
        }}

@app.get("/")
async def root():
    return {{"message": "{self.model_name} Service is running"}}

@app.get("/health")
async def health_check():
    return {{"status": "healthy"}}

@app.post("/predict", response_model=ModelOutput)
async def predict(input_data: ModelInput):
    try:
        # Convert input to numpy array
        input_array = np.array(input_data.data, dtype=np.float32)
        
        # Run inference
        outputs = session.run(None, {{"input": input_array}})
        
        # Get the first output (assuming single output model)
        prediction = outputs[0].tolist()
        
        return ModelOutput(prediction=prediction)
    
    except Exception as e:
        logger.error(f"Prediction error: {{e}}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def model_info():
    try:
        # Get model input and output info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        return {{
            "model_name": "{self.model_name}",
            "model_type": "{self.model_type}",
            "input_shape": input_info.shape,
            "input_type": input_info.type,
            "output_shape": output_info.shape,
            "output_type": output_info.type
        }}
    
    except Exception as e:
        logger.error(f"Model info error: {{e}}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port={port})
"""
        
        # Write app file
        app_path = "app.py"
        with open(app_path, "w") as f:
            f.write(app_content)
        
        print(f"Application file created at {app_path}")
        return app_path
    
    def build_docker_image(self, dockerfile_path: str = "Dockerfile") -> bool:
        """
        Build the Docker image.
        
        Args:
            dockerfile_path: Path to the Dockerfile
            
        Returns:
            True if build succeeded, False otherwise
        """
        try:
            # Build the Docker image
            build_command = [
                "docker", "build",
                "-t", self.image_name,
                "-f", dockerfile_path,
                "."
            ]
            
            result = subprocess.run(
                build_command,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"Docker image built successfully: {self.image_name}")
            print(result.stdout)
            return True
        
        except subprocess.CalledProcessError as e:
            print(f"Docker build failed: {e.stderr}")
            return False
    
    def run_docker_container(self, port: int = 8000, 
                           host_port: int = 8000) -> bool:
        """
        Run the Docker container.
        
        Args:
            port: Port inside the container
            host_port: Port on the host machine
            
        Returns:
            True if container started successfully, False otherwise
        """
        try:
            # Run the Docker container
            run_command = [
                "docker", "run",
                "-d",  # Detached mode
                "--name", self.container_name,
                "-p", f"{host_port}:{port}",
                self.image_name
            ]
            
            result = subprocess.run(
                run_command,
                capture_output=True,
                text=True,
                check=True
            )
            
            container_id = result.stdout.strip()
            print(f"Docker container started: {self.container_name} (ID: {container_id})")
            print(f"Service available at: http://localhost:{host_port}")
            return True
        
        except subprocess.CalledProcessError as e:
            print(f"Docker run failed: {e.stderr}")
            return False
    
    def stop_docker_container(self) -> bool:
        """
        Stop the Docker container.
        
        Returns:
            True if container stopped successfully, False otherwise
        """
        try:
            # Stop the Docker container
            stop_command = ["docker", "stop", self.container_name]
            
            subprocess.run(
                stop_command,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"Docker container stopped: {self.container_name}")
            return True
        
        except subprocess.CalledProcessError as e:
            print(f"Docker stop failed: {e.stderr}")
            return False
    
    def test_docker_service(self, host_port: int = 8000) -> bool:
        """
        Test the Docker service.
        
        Args:
            host_port: Port on the host machine
            
        Returns:
            True if test passed, False otherwise
        """
        try:
            import requests
            
            # Test health endpoint
            response = requests.get(f"http://localhost:{host_port}/health")
            if response.status_code == 200:
                print("Health check passed")
                
                # Test prediction endpoint
                test_data = {"data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]}
                response = requests.post(
                    f"http://localhost:{host_port}/predict",
                    json=test_data
                )
                
                if response.status_code == 200:
                    print("Prediction test passed")
                    print(f"Prediction result: {response.json()}")
                    return True
                else:
                    print(f"Prediction test failed: {response.status_code}")
                    return False
            else:
                print(f"Health check failed: {response.status_code}")
                return False
        
        except Exception as e:
            print(f"Service test failed: {e}")
            return False

# Example usage
def dockerize_model_example():
    """Example of dockerizing a model."""
    # First, create a simple ONNX model
    model = SimpleModel(input_size=10, hidden_size=20, output_size=5)
    converter = PyTorchToONNXConverter(model, input_shape=(1, 10))
    converter.export_to_onnx("simple_model.onnx")
    
    # Dockerize the model
    dockerizer = ModelDockerizer("simple_model.onnx")
    
    # Create Dockerfile and app file
    dockerizer.create_dockerfile()
    dockerizer.create_app_file()
    
    # Build Docker image
    if dockerizer.build_docker_image():
        # Run Docker container
        if dockerizer.run_docker_container():
            # Test the service
            dockerizer.test_docker_service()
            
            # Stop the container
            dockerizer.stop_docker_container()

# Run the example
# dockerize_model_example()
```

## When and Where We Use Model Exporting

### When to Use Model Exporting:
- When deploying models to production environments
- When models need to run on edge devices with limited resources
- When integrating models into applications written in different languages
- When using specialized inference accelerators or hardware
- When sharing models with others who might not have the same framework

### When to Use Dockerization:
- When deploying models to cloud platforms
- When models need to be scaled horizontally
- When ensuring consistent runtime environments across different machines
- When managing multiple models with different dependencies
- When implementing CI/CD pipelines for ML models

### Where to Use Model Exporting:
- In mobile and web applications that run models locally
- On edge devices like IoT sensors, cameras, and drones
- In microservices architectures where models are separate services
- In serverless functions that need to load models quickly
- In embedded systems with specific hardware requirements

### Where to Use Dockerization:
- In Kubernetes clusters for orchestration and scaling
- In cloud services like AWS ECS, Google Cloud Run, or Azure Container Instances
- In on-premises data centers with container orchestration
- In development environments to ensure consistency
- In CI/CD pipelines for testing and deployment

**Coding Example:**
```python
class ModelDeploymentPipeline:
    """
    A complete pipeline for model deployment, including exporting and dockerization.
    """
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...]):
        """
        Initialize the deployment pipeline.
        
        Args:
            model: PyTorch model to deploy
            input_shape: Shape of the input tensor
        """
        self.model = model
        self.input_shape = input_shape
        self.model_name = "deployed_model"
        
        # Create directories
        os.makedirs("dist", exist_ok=True)
        os.makedirs("docker", exist_ok=True)
    
    def prepare_model(self) -> str:
        """
        Prepare the model for deployment.
        
        Returns:
            Path to the prepared model
        """
        # Convert to ONNX
        converter = PyTorchToONNXConverter(self.model, self.input_shape)
        onnx_path = f"dist/{self.model_name}.onnx"
        converter.export_to_onnx(onnx_path)
        
        # Verify the model
        if not converter.verify_onnx_model(onnx_path):
            raise Exception("Model verification failed")
        
        # Optimize the model
        optimized_path = f"dist/{self.model_name}_optimized.onnx"
        converter.optimize_onnx_model(onnx_path, optimized_path)
        
        return optimized_path
    
    def create_deployment_package(self, model_path: str) -> Dict[str, str]:
        """
        Create a complete deployment package.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Dictionary with paths to created files
        """
        # Initialize dockerizer
        dockerizer = ModelDockerizer(model_path)
        dockerizer.model_name = self.model_name
        dockerizer.image_name = f"{self.model_name}-service"
        dockerizer.container_name = f"{self.model_name}-container"
        
        # Create Dockerfile in docker directory
        os.chdir("docker")
        dockerfile_path = dockerizer.create_dockerfile()
        app_path = dockerizer.create_app_file()
        
        # Copy model to docker directory
        import shutil
        shutil.copy(f"../{model_path", ".")
        
        os.chdir("..")
        
        return {
            "model_path": model_path,
            "dockerfile_path": f"docker/{dockerfile_path}",
            "app_path": f"docker/{app_path}"
        }
    
    def deploy(self, test: bool = True) -> bool:
        """
        Deploy the model.
        
        Args:
            test: Whether to test the deployment
            
        Returns:
            True if deployment succeeded, False otherwise
        """
        try:
            # Prepare model
            model_path = self.prepare_model()
            
            # Create deployment package
            package = self.create_deployment_package(model_path)
            
            # Build Docker image
            dockerizer = ModelDockerizer(model_path)
            dockerizer.model_name = self.model_name
            dockerizer.image_name = f"{self.model_name}-service"
            dockerizer.container_name = f"{self.model_name}-container"
            
            os.chdir("docker")
            
            if not dockerizer.build_docker_image():
                return False
            
            # Run container
            if not dockerizer.run_docker_container():
                return False
            
            # Test service
            if test:
                if not dockerizer.test_docker_service():
                    return False
            
            os.chdir("..")
            
            print("Model deployed successfully!")
            return True
        
        except Exception as e:
            print(f"Deployment failed: {e}")
            return False

# Example usage
def deployment_pipeline_example():
    """Example of using the deployment pipeline."""
    # Create a model
    model = SimpleModel(input_size=10, hidden_size=20, output_size=5)
    
    # Create deployment pipeline
    pipeline = ModelDeploymentPipeline(model, input_shape=(1, 10))
    
    # Deploy the model
    success = pipeline.deploy()
    
    if success:
        print("Model deployment pipeline completed successfully!")
    else:
        print("Model deployment pipeline failed!")

# Run the example
# deployment_pipeline_example()
```

## Real-World Applications and Examples

### 1. Edge AI Camera

**Application:**
An edge AI camera that runs object detection models locally for real-time analysis, with the model exported to ONNX and deployed in a Docker container.

**Coding Example:**
```python
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

class EdgeAICamera:
    """
    An edge AI camera that runs object detection models.
    """
    def __init__(self, model_path: str, labels_path: str):
        """
        Initialize the edge AI camera.
        
        Args:
            model_path: Path to the ONNX model
            labels_path: Path to the labels file
        """
        # Load the ONNX model
        self.session = ort.InferenceSession(model_path)
        
        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        
        # Get model input and output details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        
        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a frame for inference.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        # Resize frame to match model input shape
        input_height, input_width = self.input_shape[2:4]
        resized_frame = cv2.resize(frame, (input_width, input_height))
        
        # Convert to RGB (if it's BGR)
        if len(resized_frame.shape) == 3 and resized_frame.shape[2] == 3:
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and add batch dimension
        preprocessed = resized_frame.astype(np.float32) / 255.0
        preprocessed = np.expand_dims(preprocessed, axis=0)
        
        # If model expects NCHW format, transpose from NHWC
        if self.input_shape[1] == 3:  # NCHW format
            preprocessed = np.transpose(preprocessed, (0, 3, 1, 2))
        
        return preprocessed
    
    def detect_objects(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Detect objects in a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (detections, processed_frame)
        """
        # Preprocess frame
        preprocessed = self.preprocess_frame(frame)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: preprocessed})
        
        # Process outputs
        detections = self._process_outputs(outputs[0])
        
        # Draw detections on frame
        processed_frame = self._draw_detections(frame, detections)
        
        return detections, processed_frame
    
    def _process_outputs(self, outputs: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process model outputs to extract detections.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            List of detection dictionaries
        """
        # This is a simplified example - actual processing depends on the model
        # For YOLO-style models, outputs would have shape [batch, num_detections, 5+num_classes]
        
        detections = []
        
        # Assuming outputs is [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, confidence, class_id]
        for detection in outputs[0]:
            if detection[4] > self.confidence_threshold:
                x1, y1, x2, y2, confidence, class_id = detection
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(confidence),
                    "class_id": int(class_id),
                    "class_name": self.labels[int(class_id)] if int(class_id) < len(self.labels) else "Unknown"
                })
        
        return detections
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detection bounding boxes and labels on a frame.
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            
        Returns:
            Frame with detections drawn
        """
        processed_frame = frame.copy()
        
        for detection in detections:
            # Extract bounding box
            x1, y1, x2, y2 = detection["bbox"]
            
            # Convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            cv2.putText(processed_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return processed_frame
    
    def run(self):
        """Run the edge AI camera."""
        try:
            while True:
                # Read frame from camera
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                # Detect objects
                detections, processed_frame = self.detect_objects(frame)
                
                # Display frame
                cv2.imshow('Edge AI Camera', processed_frame)
                
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            # Release resources
            self.camera.release()
            cv2.destroyAllWindows()

# Example usage
def edge_ai_camera_example():
    """Example of using the edge AI camera."""
    # This example assumes you have an ONNX model and labels file
    # In a real scenario, you would provide actual paths
    
    # Create edge AI camera
    # camera = EdgeAICamera(
    #     model_path="yolov5s.onnx",
    #     labels_path="coco.names"
    # )
    
    # Run the camera
    # camera.run()

# Note: This is a simplified example. A real implementation would require
# actual model files and more sophisticated output processing.
```

### 2. Cloud Model Service

**Application:**
A cloud-based model service that serves multiple models through a REST API, with each model exported to ONNX and dockerized for easy deployment and scaling.

**Coding Example:**
```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import logging
import os
from typing import Dict, List, Any, Optional
import uvicorn

class ModelRegistry:
    """
    A registry for managing multiple ONNX models.
    """
    def __init__(self):
        """Initialize the model registry."""
        self.models = {}
        self.model_info = {}
    
    def register_model(self, name: str, path: str, version: str = "1.0") -> bool:
        """
        Register a model in the registry.
        
        Args:
            name: Name of the model
            path: Path to the ONNX model file
            version: Version of the model
            
        Returns:
            True if registration succeeded, False otherwise
        """
        try:
            # Load the model
            session = ort.InferenceSession(path)
            
            # Get model info
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            
            # Store model and info
            self.models[name] = session
            self.model_info[name] = {
                "path": path,
                "version": version,
                "input_shape": input_info.shape,
                "input_type": input_info.type,
                "output_shape": output_info.shape,
                "output_type": output_info.type
            }
            
            logging.info(f"Model '{name}' registered successfully")
            return True
        
        except Exception as e:
            logging.error(f"Failed to register model '{name}': {e}")
            return False
    
    def get_model(self, name: str) -> Optional[ort.InferenceSession]:
        """
        Get a model by name.
        
        Args:
            name: Name of the model
            
        Returns:
            Model session or None if not found
        """
        return self.models.get(name)
    
    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get model information by name.
        
        Args:
            name: Name of the model
            
        Returns:
            Model information or None if not found
        """
        return self.model_info.get(name)
    
    def list_models(self) -> List[str]:
        """
        List all registered models.
        
        Returns:
            List of model names
        """
        return list(self.models.keys())

# Input and output models
class ModelInput(BaseModel):
    model_name: str
    data: List[List[float]]
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "simple_model",
                "data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
            }
        }

class ModelOutput(BaseModel):
    model_name: str
    prediction: List[float]
    model_version: str
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "simple_model",
                "prediction": [0.1, 0.2, 0.3, 0.4, 0.5],
                "model_version": "1.0"
            }
        }

class ModelInfo(BaseModel):
    name: str
    version: str
    input_shape: List[int]
    input_type: str
    output_shape: List[int]
    output_type: str

# Create FastAPI app
app = FastAPI(title="Cloud Model Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model registry
model_registry = ModelRegistry()

# Register models on startup
@app.on_event("startup")
async def startup_event():
    """Register models on startup."""
    # In a real implementation, you would load models from a directory
    # For this example, we'll assume models are in a 'models' directory
    
    models_dir = "models"
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith(".onnx"):
                model_name = os.path.splitext(file)[0]
                model_path = os.path.join(models_dir, file)
                model_registry.register_model(model_name, model_path)

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Cloud Model Service is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/models", response_model=List[str])
async def list_models():
    """List all registered models."""
    return model_registry.list_models()

@app.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """Get information about a model."""
    info = model_registry.get_model_info(model_name)
    if not info:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    return ModelInfo(
        name=model_name,
        version=info["version"],
        input_shape=info["input_shape"],
        input_type=info["input_type"],
        output_shape=info["output_shape"],
        output_type=info["output_type"]
    )

@app.post("/predict", response_model=ModelOutput)
async def predict(input_data: ModelInput):
    """Make a prediction using a model."""
    # Get the model
    model = model_registry.get_model(input_data.model_name)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{input_data.model_name}' not found")
    
    try:
        # Convert input to numpy array
        input_array = np.array(input_data.data, dtype=np.float32)
        
        # Get model input name
        input_name = model.get_inputs()[0].name
        
        # Run inference
        outputs = model.run(None, {input_name: input_array})
        
        # Get the first output (assuming single output model)
        prediction = outputs[0].tolist()
        
        # Get model version
        model_info = model_registry.get_model_info(input_data.model_name)
        model_version = model_info["version"] if model_info else "unknown"
        
        return ModelOutput(
            model_name=input_data.model_name,
            prediction=prediction,
            model_version=model_version
        )
    
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Example usage
def cloud_model_service_example():
    """Example of running the cloud model service."""
    # This would typically be run with uvicorn from the command line:
    # uvicorn cloud_model_service:app --host 0.0.0.0 --port 8000
    
    # For this example, we'll just show how to run it programmatically
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Note: This is a simplified example. A real implementation would include:
# - Authentication and authorization
# - Rate limiting
# - Model versioning
# - A/B testing
# - Monitoring and logging
# - Scalability with multiple workers
```

## Common Problems and How Advancements Have Solved Them

### Problem 1: Framework Lock-in and Deployment Complexity

**Problem:**
Models trained in specific frameworks like PyTorch or TensorFlow were difficult to deploy in environments that didn't support those frameworks. This created vendor lock-in and made it challenging to deploy models across different platforms and devices.

**Solution:**
Standardized model formats like ONNX, along with containerization technologies like Docker, have solved this problem by:
1. Providing framework-agnostic model representations
2. Enabling models to run on various hardware and software platforms
3. Simplifying the deployment process through containerization
4. Allowing models to be optimized for specific inference engines

**Problem Code:**
```python
# Old approach - framework-specific deployment
def deploy_pytorch_model(model, input_shape):
    """
    Deploy a PyTorch model in a framework-specific way.
    
    Args:
        model: PyTorch model to deploy
        input_shape: Shape of the input tensor
    """
    # This approach requires PyTorch to be available in the deployment environment
    import torch
    
    # Save the model
    torch.save(model.state_dict(), "model.pth")
    
    # Create a simple Flask app that requires PyTorch
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    # Load the model (requires PyTorch)
    loaded_model = SimpleModel(*input_shape[1:])
    loaded_model.load_state_dict(torch.load("model.pth"))
    loaded_model.eval()
    
    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.json["data"]
        input_tensor = torch.tensor(data, dtype=torch.float32)
        
        with torch.no_grad():
            output = loaded_model(input_tensor)
        
        return jsonify({"prediction": output.tolist()})
    
    # Run the app
    app.run(host="0.0.0.0", port=5000)

# This approach has several limitations:
# 1. Requires PyTorch in the deployment environment
# 2. Not easily portable to different platforms
# 3. Difficult to optimize for specific hardware
```

**Enhancement Code:**
```python
# New approach - framework-agnostic deployment with ONNX and Docker
def deploy_model_agnostic(model, input_shape):
    """
    Deploy a model in a framework-agnostic way.
    
    Args:
        model: PyTorch model to deploy
        input_shape: Shape of the input tensor
    """
    # Convert to ONNX
    converter = PyTorchToONNXConverter(model, input_shape)
    converter.export_to_onnx("model.onnx")
    
    # Verify and optimize
    converter.verify_onnx_model("model.onnx")
    converter.optimize_onnx_model("model.onnx", "model_optimized.onnx")
    
    # Create Docker container
    dockerizer = ModelDockerizer("model_optimized.onnx")
    dockerizer.create_dockerfile()
    dockerizer.create_app_file()
    
    # Build and run container
    dockerizer.build_docker_image()
    dockerizer.run_docker_container()
    
    # Test the service
    dockerizer.test_docker_service()
    
    # The service can now be deployed anywhere Docker is available
    # without requiring PyTorch or any specific ML framework

# This approach provides:
# 1. Framework independence
# 2. Portability across platforms
# 3. Optimization opportunities
# 4. Easier deployment and scaling
```

### Problem 2: Inconsistent Runtime Environments

**Problem:**
Models often behaved differently in development, testing, and production environments due to differences in dependencies, library versions, and system configurations. This led to the "it works on my machine" problem and made deployments unreliable.

**Solution:**
Docker containers solve this problem by:
1. Packaging the model, dependencies, and runtime into a single, immutable unit
2. Ensuring consistent behavior across all environments
3. Simplifying dependency management
4. Enabling easy scaling and orchestration

**Problem Code:**
```python
# Old approach - environment-specific deployment
def deploy_with_venv():
    """
    Deploy a model using a virtual environment.
    
    This approach is prone to environment inconsistencies.
    """
    # Create a virtual environment
    import subprocess
    import sys
    
    # Create virtual environment
    subprocess.run([sys.executable, "-m", "venv", "venv"])
    
    # Activate virtual environment and install dependencies
    # This is platform-specific and can fail
    subprocess.run("source venv/bin/activate && pip install -r requirements.txt", shell=True)
    
    # The model service now depends on the specific environment
    # and may behave differently on different systems
```

**Enhancement Code:**
```python
# New approach - containerized deployment
def deploy_with_docker():
    """
    Deploy a model using Docker containers.
    
    This approach ensures consistent behavior across environments.
    """
    # Create a complete deployment pipeline
    model = SimpleModel(input_size=10, hidden_size=20, output_size=5)
    pipeline = ModelDeploymentPipeline(model, input_shape=(1, 10))
    
    # The pipeline handles:
    # 1. Model conversion to ONNX
    # 2. Verification and optimization
    # 3. Docker image creation
    # 4. Container deployment
    # 5. Service testing
    
    success = pipeline.deploy()
    
    if success:
        print("Model deployed successfully in a Docker container!")
        print("The container can now be run on any system with Docker,")
        print("ensuring consistent behavior across all environments.")
    
    # The container can be easily:
    # 1. Scaled horizontally
    # 2. Deployed to cloud platforms
    # 3. Managed with orchestration tools like Kubernetes
    # 4. Updated with zero downtime
```

## 🧪 Exercise: Convert Model Format

Let's create an exercise to convert a model from PyTorch to ONNX format.

```python
def model_conversion_exercise():
    """
    Exercise: Convert a PyTorch model to ONNX format.
    """
    print("Model Conversion Exercise")
    print("=" * 50)
    
    # Create a more complex model for the exercise
    class ComplexModel(nn.Module):
        def __init__(self):
            super(ComplexModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(32 * 8 * 8, 128)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.flatten(x)
            x = self.relu3(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Create the model
    model = ComplexModel()
    
    # Create a random input for testing
    input_shape = (1, 3, 32, 32)  # Batch of 1, 3 channels, 32x32 image
    test_input = torch.randn(input_shape)
    
    # Get PyTorch output
    with torch.no_grad():
        pytorch_output = model(test_input)
    
    print(f"PyTorch model output shape: {pytorch_output.shape}")
    
    # Step 1: Convert to ONNX
    print("\nStep 1: Converting to ONNX...")
    converter = PyTorchToONNXConverter(model, input_shape)
    converter.export_to_onnx("exercise_model.onnx")
    
    # Step 2: Verify the conversion
    print("\nStep 2: Verifying the conversion...")
    is_valid = converter.verify_onnx_model("exercise_model.onnx")
    
    if not is_valid:
        print("Conversion verification failed!")
        return
    
    print("Conversion verification passed!")
    
    # Step 3: Optimize the model
    print("\nStep 3: Optimizing the model...")
    converter.optimize_onnx_model("exercise_model.onnx", "exercise_model_optimized.onnx")
    
    # Step 4: Verify the optimized model
    print("\nStep 4: Verifying the optimized model...")
    is_valid = converter.verify_onnx_model("exercise_model_optimized.onnx")
    
    if not is_valid:
        print("Optimized model verification failed!")
        return
    
    print("Optimized model verification passed!")
    
    # Step 5: Compare model sizes
    print("\nStep 5: Comparing model sizes...")
    import os
    
    original_size = os.path.getsize("exercise_model.onnx")
    optimized_size = os.path.getsize("exercise_model_optimized.onnx")
    
    print(f"Original model size: {original_size / 1024:.2f} KB")
    print(f"Optimized model size: {optimized_size / 1024:.2f} KB")
    print(f"Size reduction: {(1 - optimized_size / original_size) * 100:.2f}%")
    
    # Step 6: Compare inference times
    print("\nStep 6: Comparing inference times...")
    import time
    
    # PyTorch inference time
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            model(test_input)
    pytorch_time = time.time() - start_time
    
    # ONNX inference time
    ort_session = ort.InferenceSession("exercise_model_optimized.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
    
    start_time = time.time()
    for _ in range(100):
        ort_session.run(None, ort_inputs)
    onnx_time = time.time() - start_time
    
    print(f"PyTorch inference time (100 runs): {pytorch_time:.4f} seconds")
    print(f"ONNX inference time (100 runs): {onnx_time:.4f} seconds")
    print(f"Speedup: {pytorch_time / onnx_time:.2f}x")
    
    print("\nExercise completed successfully!")

# Run the exercise
# model_conversion_exercise()
```

## 📌 Project: Containerized Model Service

Now, let's create a comprehensive project - a containerized model service that can serve multiple models through a REST API.

```python
import os
import sys
import argparse
import json
import logging
from typing import Dict, List, Any, Optional
import shutil
import tempfile

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContainerizedModelService:
    """
    A complete containerized model service project.
    """
    def __init__(self, config_path: str = None):
        """
        Initialize the containerized model service.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path) if config_path else self._default_config()
        
        # Create directories
        self._create_directories()
        
        # Initialize components
        self.model_registry = ModelRegistry()
        self.model_converter = PyTorchToONNXConverter
        self.model_dockerizer = ModelDockerizer
    
    def _default_config(self) -> Dict[str, Any]:
        """
        Create a default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "service_name": "model-service",
            "version": "1.0.0",
            "port": 8000,
            "models_dir": "models",
            "docker_base_image": "python:3.9-slim",
            "requirements": [
                "fastapi",
                "uvicorn",
                "onnxruntime",
                "numpy",
                "pydantic",
                "python-multipart"
            ],
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _create_directories(self):
        """Create necessary directories."""
        os.makedirs(self.config["models_dir"], exist_ok=True)
        os.makedirs("docker", exist_ok=True)
        os.makedirs("dist", exist_ok=True)
    
    def create_sample_model(self, name: str = "sample_model") -> str:
        """
        Create a sample PyTorch model for demonstration.
        
        Args:
            name: Name of the model
            
        Returns:
            Path to the created model
        """
        import torch
        
        # Create a simple model
        class SampleModel(torch.nn.Module):
            def __init__(self):
                super(SampleModel, self).__init__()
                self.fc1 = torch.nn.Linear(10, 20)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(20, 5)
            
            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x
        
        # Create and save the model
        model = SampleModel()
        model_path = os.path.join(self.config["models_dir"], f"{name}.pt")
        torch.save(model.state_dict(), model_path)
        
        logger.info(f"Sample model created at {model_path}")
        return model_path
    
    def convert_all_models(self):
        """Convert all PyTorch models in the models directory to ONNX."""
        models_dir = self.config["models_dir"]
        
        for file in os.listdir(models_dir):
            if file.endswith(".pt"):
                model_name = os.path.splitext(file)[0]
                pt_path = os.path.join(models_dir, file)
                onnx_path = os.path.join(models_dir, f"{model_name}.onnx")
                
                logger.info(f"Converting {file} to ONNX...")
                
                # Load the PyTorch model
                import torch
                model = SampleModel()
                model.load_state_dict(torch.load(pt_path))
                model.eval()
                
                # Convert to ONNX
                converter = self.model_converter(model, input_shape=(1, 10))
                converter.export_to_onnx(onnx_path)
                
                # Verify conversion
                if converter.verify_onnx_model(onnx_path):
                    logger.info(f"Successfully converted {file} to ONNX")
                    
                    # Optimize the model
                    optimized_path = os.path.join(models_dir, f"{model_name}_optimized.onnx")
                    converter.optimize_onnx_model(onnx_path, optimized_path)
                    
                    if converter.verify_onnx_model(optimized_path):
                        logger.info(f"Successfully optimized {file}")
                    else:
                        logger.warning(f"Failed to verify optimized model for {file}")
                else:
                    logger.error(f"Failed to convert {file} to ONNX")
    
    def create_docker_files(self):
        """Create Docker files for the service."""
        # Create Dockerfile
        dockerfile_content = f"""FROM {self.config["docker_base_image"]}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy models and application code
COPY models/ ./models/
COPY app.py .

# Expose port
EXPOSE {self.config["port"]}

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "{self.config["port"]}"]
"""
        
        with open("docker/Dockerfile", "w") as f:
            f.write(dockerfile_content)
        
        # Create requirements.txt
        with open("docker/requirements.txt", "w") as f:
            f.write("\n".join(self.config["requirements"]))
        
        # Create app.py
        app_content = f'''from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import logging
import os
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="{self.config["service_name"]}", version="{self.config["version"]}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model registry
class ModelRegistry:
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = models_dir
        self.models = {{}}
        self.model_info = {{}}
        self._load_models()
    
    def _load_models(self):
        """Load all ONNX models from the models directory."""
        for file in os.listdir(self.models_dir):
            if file.endswith(".onnx") and not file.endswith("_optimized.onnx"):
                model_name = os.path.splitext(file)[0]
                model_path = os.path.join(self.models_dir, file)
                self._load_model(model_name, model_path)
    
    def _load_model(self, name: str, path: str):
        """Load a single ONNX model."""
        try:
            session = ort.InferenceSession(path)
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            
            self.models[name] = session
            self.model_info[name] = {{
                "path": path,
                "input_shape": input_info.shape,
                "input_type": input_info.type,
                "output_shape": output_info.shape,
                "output_type": output_info.type
            }}
            
            logger.info(f"Model '{{name}}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model '{{name}}': {{e}}")
    
    def get_model(self, name: str):
        return self.models.get(name)
    
    def get_model_info(self, name: str):
        return self.model_info.get(name)
    
    def list_models(self):
        return list(self.models.keys())

# Initialize model registry
model_registry = ModelRegistry()

# Input and output models
class ModelInput(BaseModel):
    model_name: str
    data: List[List[float]]

class ModelOutput(BaseModel):
    model_name: str
    prediction: List[float]

class ModelInfo(BaseModel):
    name: str
    input_shape: List[int]
    input_type: str
    output_shape: List[int]
    output_type: str

@app.get("/")
async def root():
    return {{"message": "{self.config["service_name"]} is running"}}

@app.get("/health")
async def health_check():
    return {{"status": "healthy"}}

@app.get("/models", response_model=List[str])
async def list_models():
    return model_registry.list_models()

@app.get("/models/{{model_name}}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    info = model_registry.get_model_info(model_name)
    if not info:
        raise HTTPException(status_code=404, detail=f"Model '{{model_name}}' not found")
    
    return ModelInfo(
        name=model_name,
        input_shape=info["input_shape"],
        input_type=info["input_type"],
        output_shape=info["output_shape"],
        output_type=info["output_type"]
    )

@app.post("/predict", response_model=ModelOutput)
async def predict(input_data: ModelInput):
    model = model_registry.get_model(input_data.model_name)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{{input_data.model_name}}' not found")
    
    try:
        input_array = np.array(input_data.data, dtype=np.float32)
        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {{input_name: input_array}})
        prediction = outputs[0].tolist()
        
        return ModelOutput(
            model_name=input_data.model_name,
            prediction=prediction
        )
    except Exception as e:
        logger.error(f"Prediction error: {{e}}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port={self.config["port"]})
'''
        
        with open("docker/app.py", "w") as f:
            f.write(app_content)
        
        logger.info("Docker files created successfully")
    
    def build_docker_image(self):
        """Build the Docker image."""
        import subprocess
        
        # Copy models to docker directory
        shutil.copytree(self.config["models_dir"], "docker/models", dirs_exist_ok=True)
        
        # Build the image
        build_command = [
            "docker", "build",
            "-t", self.config["service_name"],
            "docker"
        ]
        
        try:
            result = subprocess.run(
                build_command,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Docker image built successfully: {self.config['service_name']}")
            logger.info(result.stdout)
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker build failed: {e.stderr}")
            return False
    
    def run_docker_container(self):
        """Run the Docker container."""
        import subprocess
        
        run_command = [
            "docker", "run",
            "-d",
            "--name", self.config["service_name"],
            "-p", f"{self.config['port']}:{self.config['port']}",
            self.config["service_name"]
        ]
        
        try:
            result = subprocess.run(
                run_command,
                capture_output=True,
                text=True,
                check=True
            )
            
            container_id = result.stdout.strip()
            logger.info(f"Docker container started: {self.config['service_name']} (ID: {container_id})")
            logger.info(f"Service available at: http://localhost:{self.config['port']}")
            return container_id
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker run failed: {e.stderr}")
            return None
    
    def test_service(self):
        """Test the running service."""
        import requests
        import time
        
        # Wait for service to start
        time.sleep(5)
        
        try:
            # Test health endpoint
            response = requests.get(f"http://localhost:{self.config['port']}/health")
            if response.status_code == 200:
                logger.info("Health check passed")
                
                # Test model listing
                response = requests.get(f"http://localhost:{self.config['port']}/models")
                if response.status_code == 200:
                    models = response.json()
                    logger.info(f"Available models: {models}")
                    
                    if models:
                        # Test prediction
                        test_data = {
                            "model_name": models[0],
                            "data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
                        }
                        
                        response = requests.post(
                            f"http://localhost:{self.config['port']}/predict",
                            json=test_data
                        )
                        
                        if response.status_code == 200:
                            logger.info("Prediction test passed")
                            logger.info(f"Prediction result: {response.json()}")
                            return True
                        else:
                            logger.error(f"Prediction test failed: {response.status_code}")
                    else:
                        logger.warning("No models available for testing")
                else:
                    logger.error(f"Model listing test failed: {response.status_code}")
            else:
                logger.error(f"Health check failed: {response.status_code}")
        
        except Exception as e:
            logger.error(f"Service test failed: {e}")
        
        return False
    
    def stop_docker_container(self):
        """Stop the Docker container."""
        import subprocess
        
        stop_command = ["docker", "stop", self.config["service_name"]]
        
        try:
            subprocess.run(
                stop_command,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Docker container stopped: {self.config['service_name']}")
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker stop failed: {e.stderr}")
            return False
    
    def create_project(self):
        """Create the complete containerized model service project."""
        logger.info("Creating containerized model service project...")
        
        # Step 1: Create sample model
        logger.info("Step 1: Creating sample model...")
        self.create_sample_model()
        
        # Step 2: Convert models to ONNX
        logger.info("Step 2: Converting models to ONNX...")
        self.convert_all_models()
        
        # Step 3: Create Docker files
        logger.info("Step 3: Creating Docker files...")
        self.create_docker_files()
        
        # Step 4: Build Docker image
        logger.info("Step 4: Building Docker image...")
        if not self.build_docker_image():
            logger.error("Failed to build Docker image")
            return False
        
        # Step 5: Run Docker container
        logger.info("Step 5: Running Docker container...")
        container_id = self.run_docker_container()
        if not container_id:
            logger.error("Failed to run Docker container")
            return False
        
        # Step 6: Test service
        logger.info("Step 6: Testing service...")
        if self.test_service():
            logger.info("Service test passed!")
        else:
            logger.warning("Service test failed")
        
        # Step 7: Stop container (optional)
        logger.info("Step 7: Stopping container...")
        self.stop_docker_container()
        
        logger.info("Project created successfully!")
        return True

def main():
    """Main function for the containerized model service project."""
    parser = argparse.ArgumentParser(description="Containerized Model Service")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--create-config", action="store_true", help="Create a sample configuration file")
    parser.add_argument("--create-sample-model", action="store_true", help="Create a sample model")
    parser.add_argument("--convert-models", action="store_true", help="Convert all models to ONNX")
    parser.add_argument("--create-docker", action="store_true", help="Create Docker files")
    parser.add_argument("--build-image", action="store_true", help="Build Docker image")
    parser.add_argument("--run-container", action="store_true", help="Run Docker container")
    parser.add_argument("--test-service", action="store_true", help="Test the service")
    parser.add_argument("--create-project", action="store_true", help="Create the complete project")
    
    args = parser.parse_args()
    
    # Create sample config if requested
    if args.create_config:
        config = ContainerizedModelService()._default_config()
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
        logger.info("Sample configuration created: config.json")
        return
    
    # Initialize service
    service = ContainerizedModelService(args.config)
    
    # Execute requested actions
    if args.create_sample_model:
        service.create_sample_model()
    
    if args.convert_models:
        service.convert_all_models()
    
    if args.create_docker:
        service.create_docker_files()
    
    if args.build_image:
        service.build_docker_image()
    
    if args.run_container:
        service.run_docker_container()
    
    if args.test_service:
        service.test_service()
    
    if args.create_project:
        service.create_project()

if __name__ == "__main__":
    main()
```

