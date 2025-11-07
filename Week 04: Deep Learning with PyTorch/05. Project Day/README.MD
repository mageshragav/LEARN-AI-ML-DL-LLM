

# Day 20 – Project Day: Full Image Classifier + Streamlit Deployment

## What is an End-to-End Deep Learning Project and Why is it Important?

An end-to-end deep learning project encompasses the entire lifecycle of a machine learning application, from data acquisition and preprocessing to model training, evaluation, saving/loading, and finally, deployment as an interactive application. It's not just about achieving high accuracy in a Jupyter notebook; it's about creating a functional tool that can be used by others.

Building such a project is crucial because:

-   **It Simulates Real-World Scenarios:** In industry, a model's value is only realized when it's deployed and solving a real problem. This project bridges the gap between academic exercises and practical application.
-   **It Teaches MLOps Fundamentals:** You learn essential skills beyond model architecture, such as versioning your model, creating a reproducible environment, and building an API or user interface.
-   **It Develops a Product Mindset:** You start thinking about the user experience. How will someone interact with my model? What kind of input should they provide? How can I present the output clearly?
-   **It Creates a Tangible Portfolio Piece:** A deployed application is a powerful demonstration of your skills to potential employers or collaborators. It shows you can see a project through from conception to completion.

## Essential Subtopics

### 1. Project Setup and Environment Management

**What is Project Setup and Environment Management?**
This is the foundational step of creating a clean, reproducible, and isolated environment for your project. It involves defining the project structure, managing dependencies (libraries and their versions), and ensuring that your code can run consistently on different machines.

**Explanation:**
-   **Project Structure:** Organizing your code, data, and saved models into a logical directory structure (e.g., separate folders for `data`, `src`, `models`, `app`).
-   **Virtual Environments:** Tools like `venv` or `conda` create isolated Python environments. This prevents conflicts between project-specific dependencies and system-wide packages.
-   **Dependency Management:** A `requirements.txt` file lists all the Python packages and their exact versions needed to run the project. This allows anyone (including your future self) to recreate the environment with a single command.

**Implementation (Project Structure & `requirements.txt`):**

First, let's define the structure of our project. Create the following folders and files:

```
image_classifier_project/
├── data/
│   └── raw/  # We will download the dataset here
├── models/
│   └── saved_model.pth  # Our trained model will be saved here
├── src/
│   └── train.py         # Script for training the model
├── app/
│   └── streamlit_app.py # Script for the Streamlit application
└── requirements.txt     # List of dependencies
```

Now, let's create the `requirements.txt` file. This file is critical for reproducibility.

**`requirements.txt`:**
```
# Core deep learning framework
torch==2.1.0
torchvision==0.16.0

# For image manipulation and handling
Pillow==10.0.1

# For the web application deployment
streamlit==1.28.0

# For data manipulation and visualization (useful for analysis)
numpy==1.24.3
matplotlib==3.7.2
```

To set up the environment, you would run these commands in your terminal:
```bash
# 1. Create the project directory and navigate into it
mkdir image_classifier_project
cd image_classifier_project

# 2. Create a virtual environment (using venv)
python -m venv venv

# 3. Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install the dependencies from requirements.txt
pip install -r requirements.txt
```

**When and where we use it:**
-   **At the very beginning** of any new software or data science project.
-   **When collaborating** with a team to ensure everyone has the same setup.
-   **When deploying** an application to a server or cloud platform to guarantee the correct packages are installed.

**Real-world examples:**
-   A data scientist at a healthcare company starts a new project to classify X-rays. They create a `conda` environment with specific versions of PyTorch and medical imaging libraries to ensure their results are reproducible.
-   A startup develops a new AI feature. They use a `requirements.txt` file to deploy their model to a cloud service like AWS SageMaker or Heroku, ensuring the production environment matches the development environment.

**Problems and Advancements:**
-   **Problem:** Manually managing environments and dependencies can lead to "it works on my machine" syndrome, where code runs perfectly on the developer's computer but fails elsewhere due to version conflicts.
  -   **Advancement:** **Containerization with Docker**. Docker takes environment management to the next level by packaging the application code along with its entire runtime environment (OS, system libraries, Python, packages) into a lightweight, portable container. This guarantees that the application runs exactly the same way everywhere.
    ```dockerfile
    # Example Dockerfile
    FROM python:3.9-slim
    
    WORKDIR /app
    
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    COPY . .
    
    EXPOSE 8501
    
    CMD streamlit run app/streamlit_app.py
    ```
-   **Problem:** `requirements.txt` can become messy and doesn't capture dependencies from sources other than PyPI (like Git repositories).
  -   **Advancement:** **Modern dependency tools like Poetry or Pipenv**. These tools combine the functionality of `pip` and `venv` into a single tool. They use files like `pyproject.toml` to manage dependencies more robustly, lock them to specific versions for reproducibility, and manage virtual environments automatically.

### 2. Model Training and Saving

**What is Model Training and Saving?**
This is the core process where we define our CNN architecture, feed it training data, and iteratively adjust its weights to minimize a loss function. Once the model is trained and performs well on validation data, we save its learned parameters (weights and biases) to a file. This saved file can then be loaded later for inference or deployment without needing to retrain the model from scratch.

**Explanation:**
-   **Training Loop:** The standard process involves iterating over the training data for a set number of epochs. In each epoch, the model makes predictions (forward pass), calculates the error (loss), and adjusts its weights via backpropagation and an optimizer.
-   **Validation:** During training, we periodically evaluate the model on a separate validation set (data it hasn't been trained on). This helps us monitor for overfitting and decide when to stop training.
-   **Saving the Model:** PyTorch provides two main ways to save a model:
    1.  **Saving the State Dictionary (Recommended):** This saves only the learned parameters (weights and biases) of the model. It's the most flexible and widely used method. You need the model class definition to load it.
    2.  **Saving the Entire Model:** This saves the model object itself, including its architecture and state dict. It's less flexible but can be convenient for quick deployment.

**Implementation (`src/train.py`):**

This script will download the CIFAR-10 dataset, define and train a CNN, and save the best performing model.

**`src/train.py`:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import os

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Create models directory if it doesn't exist
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "cifar10_cnn.pth")

# Hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.001

# --- 1. Data Loading and Preprocessing ---
# Define transformations for training and testing
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Download and load the CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='../data/raw', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='../data/raw', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# CIFAR-10 classes
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# --- 2. Define the CNN Model ---
class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CIFAR10_CNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 3. Training and Evaluation Functions ---
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

# --- 4. The Main Training Loop ---
best_accuracy = 0.0

for epoch in range(NUM_EPOCHS):
    print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
    train(model, DEVICE, train_loader, optimizer, criterion, epoch)
    accuracy = test(model, DEVICE, test_loader, criterion)

    # Save the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        # Save only the model's state dictionary
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

print("Training finished.")
print(f"Best model saved to {MODEL_PATH} with {best_accuracy:.2f}% accuracy.")
```

**When and where we use it:**
-   **During the development phase** of any machine learning project to create a predictive model.
-   **In research** to experiment with different architectures and hyperparameters.
-   **In production pipelines** where models might be retrained periodically on new data to maintain their accuracy.

**Real-world examples:**
-   A social media company trains a CNN to classify images uploaded by users into categories (e.g., food, travel, pets) to improve content recommendation and ad targeting. The trained model is saved and served by an inference API.
-   A manufacturing company trains a model to detect defects in products on an assembly line. The model is retrained nightly with new images and the best version is saved and loaded into the factory's inspection system each morning.

**Problems and Advancements:**
-   **Problem:** Training a deep learning model can be extremely time-consuming and computationally expensive, requiring powerful GPUs.
  -   **Advancement:** **Transfer Learning**. Instead of training a model from scratch, we can take a powerful model that has already been trained on a massive dataset (like ImageNet) and fine-tune it for our specific task. This drastically reduces training time and data requirements while often achieving state-of-the-art results.
    ```python
    # Example of Transfer Learning with a pre-trained ResNet
    import torchvision.models as models
    
    # Load a pre-trained ResNet-18 model
    model = models.resnet18(pretrained=True)
    
    # Freeze the early layers (optional)
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully-connected layer for our 10-class problem
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10) # CIFAR-10 has 10 classes
    
    # Now, only the parameters of the new layer will be trained by default
    # model = model.to(DEVICE)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE) # Optimize only the new layer
    ```
-   **Problem:** Managing different versions of models, tracking which hyperparameters and data versions were used to train them, and reproducing results can be chaotic.
  -   **Advancement:** **Experiment Tracking Tools (MLflow, Weights & Biases)**. These tools provide a structured way to log every aspect of your training run: hyperparameters, metrics (loss, accuracy), model artifacts, and even visualizations. This makes it easy to compare experiments, reproduce the best ones, and deploy the correct model version.

### 3. Building a Streamlit Web App

**What is Streamlit?**
Streamlit is an open-source Python library that makes it incredibly fast and easy to create custom, interactive web applications for machine learning and data science projects. With just a few lines of Python code, you can build a UI that includes elements like file uploaders, sliders, buttons, and data visualizations, without needing to know HTML, CSS, or JavaScript.

**Explanation:**
-   **Declarative Syntax:** You write a Python script, and Streamlit automatically renders it as a web app. When you update the script and save it, the app updates in real-time.
-   **Widgets:** Streamlit provides simple functions to add interactive elements like `st.title()`, `st.header()`, `st.text_input()`, `st.slider()`, `st.button()`, and crucially for us, `st.file_uploader()`.
-   **Data Flow:** Streamlit apps run from top to bottom. When a user interacts with a widget (e.g., uploads a file), the script is re-executed from top to bottom, and the UI is updated with the new results.

**Implementation (`app/streamlit_app.py`):**

This script will load our saved model, provide an interface for users to upload an image, and then display the image and the model's prediction.

**`app/streamlit_app.py`:**
```python
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import sys
import os

# --- Configuration ---
# Add the src directory to the path to import the model class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from train import CIFAR10_CNN, CLASSES

# Set up the page
st.set_page_config(page_title="CIFAR-10 Image Classifier", layout="centered")

# --- Load Model ---
@st.cache_resource
def load_model(model_path):
    """Loads the trained model from the .pth file."""
    model = CIFAR10_CNN()
    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval() # Set the model to evaluation mode
    return model

MODEL_PATH = "../models/cifar10_cnn.pth"
model = load_model(MODEL_PATH)

# --- Image Preprocessing ---
def preprocess_image(image):
    """Preprocesses the uploaded image to match the model's input requirements."""
    # Define the same transformations as in training
    preprocess = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Apply transformations
    image_tensor = preprocess(image)
    # Add a batch dimension (model expects a batch of images)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

# --- Prediction Function ---
def predict(image_tensor, model):
    """Makes a prediction on the preprocessed image."""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    return predicted_class.item(), confidence.item()

# --- Streamlit App UI ---
st.title("🚀 CIFAR-10 Image Classifier")
st.write("Upload an image and the model will predict which class it belongs to.")

st.sidebar.header("About")
st.sidebar.info("""
This is a simple image classification app built with a Convolutional Neural Network (CNN).
The model is trained on the CIFAR-10 dataset and can classify images into 10 categories:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck
""")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    st.write("Classifying...")
    
    # Preprocess the image and make a prediction
    image_tensor = preprocess_image(image)
    prediction, confidence = predict(image_tensor, model)
    
    # Display the prediction
    st.subheader("Prediction:")
    st.markdown(f"### {CLASSES[prediction]}")
    st.markdown(f"**Confidence:** {confidence:.2%}")
    
    # Display a progress bar for confidence
    st.progress(confidence)
    
    # Display top 3 predictions
    st.subheader("Top 3 Predictions:")
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        top_3_probs, top_3_classes = torch.topk(probabilities, 3)
    
    for i in range(3):
        class_name = CLASSES[top_3_classes[0][i]]
        prob = top_3_probs[0][i].item()
        st.write(f"**{i+1}. {class_name}**: {prob:.2%}")

```

**When and where we use it:**
-   **For rapid prototyping** of ML applications to get stakeholder feedback quickly.
-   **To create internal tools** for data scientists, engineers, or business analysts who need to interact with a model without writing code.
-   **For educational purposes** to demonstrate how a model works in an interactive way.
-   **To build simple public-facing demos** of AI capabilities.

**Real-world examples:**
-   A data scientist at a retail company builds a Streamlit app to let the marketing team upload images of potential products and get an instant prediction on which demographic the product would appeal to most.
-   A research team creates a Streamlit app to visualize the activations of different layers in their CNN, helping them understand what the model is learning and debug its behavior.

**Problems and Advancements:**
-   **Problem:** Streamlit apps re-run the entire script on every interaction, which can be slow if you're loading large models or datasets on every run.
  -   **Advancement:** **Caching with `@st.cache_resource`**. As shown in the code, this decorator tells Streamlit to run the decorated function only once and then cache its result in memory. On subsequent runs, the cached result is returned instantly, which is perfect for loading models or other expensive objects.
-   **Problem:** For complex, multi-page applications with user authentication, databases, and complex state management, Streamlit's simplicity can become a limitation.
  -   **Advancement:** **More Flexible Frameworks (Flask, FastAPI, Django)**. For building production-grade, scalable, and highly customizable web applications, traditional web frameworks are more suitable. They offer fine-grained control over routing, state, and backend logic. A common pattern is to build the model inference API with FastAPI (known for its speed and automatic API documentation) and build a separate frontend (e.g., with React or Vue.js) that communicates with this API. Streamlit often serves as the bridge between a simple prototype and a full-stack application.

### 4. Running and Deploying the App

**What is Deployment?**
Deployment is the process of making your application available to users over the internet or a private network. It involves taking your code and its dependencies and running them on a server that is accessible to your intended audience.

**Explanation:**
-   **Local Running:** The first step is to run the Streamlit app on your local machine to ensure everything works as expected.
-   **Cloud Deployment:** To share the app with others, you need to deploy it to a cloud platform. Streamlit Community Cloud is the easiest way to deploy Streamlit apps for free. Other options include Heroku, AWS, Google Cloud, or Azure, which offer more control and scalability.

**Implementation (Running and Deploying):**

**Step 1: Run the App Locally**

First, make sure you have trained your model by running the `src/train.py` script. Once the `models/cifar10_cnn.pth` file is created, you can run the Streamlit app.

Open your terminal (with the virtual environment activated), navigate to the project's root directory (`image_classifier_project/`), and run:

```bash
streamlit run app/streamlit_app.py
```

This will automatically open a new tab in your web browser with your application running. You can now upload images and see the predictions.

**Step 2: Deploy to Streamlit Community Cloud**

This is the simplest way to share your app publicly.

1.  **Create a GitHub Repository:**
    *   If you haven't already, initialize a Git repository in your project folder.
    *   Create a new repository on GitHub.
    *   Link your local repository to the remote GitHub repository and push all your code.
    *   **Important:** Make sure your `requirements.txt` file is in the root of your repository and is up-to-date. Streamlit Cloud uses this to install dependencies.

2.  **Deploy on Streamlit Cloud:**
    *   Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign up for a free account.
    *   Click on "New app".
    *   In the "Deploy an app" page, select your GitHub repository from the dropdown.
    *   Select the branch (usually `main` or `master`).
    *   Specify the path to your main app file: `app/streamlit_app.py`.
    *   Click "Deploy!".

Streamlit Cloud will now pull your code from GitHub, create an environment based on your `requirements.txt`, and launch your application. In a few minutes, you'll have a public URL that you can share with anyone!

**When and where we use it:**
-   **When you need to share your work** with colleagues, clients, or the public.
-   **For creating live demos** for presentations, blog posts, or portfolios.
-   **To build internal tools** that are accessible to everyone in an organization, regardless of their local machine setup.

**Real-world examples:**
-   A student finishes a final project for a machine learning course. They deploy it to Streamlit Cloud and submit the live URL as part of their project submission.
-   A freelance data scientist builds a custom NLP analysis tool for a client. They deploy it on Heroku, giving the client a secure URL to use the tool without needing to install any software.

**Problems and Advancements:**
-   **Problem:** Free deployment tiers often have limitations on resources (CPU, memory), uptime, and customizability. They may not be suitable for commercial applications with high traffic.
  -   **Advancement:** **Scalable Cloud Services (AWS SageMaker, Google AI Platform, Azure Machine Learning)**. These are enterprise-grade platforms designed specifically for deploying, managing, and scaling machine learning models. They offer features like auto-scaling (handling traffic spikes), A/B testing different models, detailed monitoring, and robust security, making them ideal for production-critical applications.
-   **Problem:** Deploying a model that requires a GPU for inference can be complex and expensive on many platforms.
  -   **Advancement:** **Serverless GPU Inference and Model Optimization**. Services like AWS Lambda (with custom containers) or specialized platforms like Replicate and Banana.dev allow you to run GPU-powered models in a serverless fashion, meaning you only pay for the compute time you use. Furthermore, model optimization techniques like **quantization** (reducing the precision of model weights, e.g., from 32-bit floats to 8-bit integers) and **pruning** (removing less important weights) can drastically reduce model size and inference time, often allowing complex models to run efficiently even on CPUs.