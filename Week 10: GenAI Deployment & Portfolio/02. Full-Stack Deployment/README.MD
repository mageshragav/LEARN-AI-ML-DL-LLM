Excellent! Let's dive into **Day 47 – Full-Stack Deployment**, focusing on **FastAPI + Streamlit**. This is a critical step in taking your trained deep learning models and making them accessible via web applications.

---

## 🧠 What is Full-Stack Deployment?

**Full-stack deployment** refers to the process of deploying both the **backend** (server-side logic, often an API) and the **frontend** (user interface) of an application so that users can interact with your AI model via a web interface.

- **FastAPI**: A modern, fast (high-performance), web framework for building APIs with Python. It's designed to be easy to use and to automatically generate interactive API documentation.
- **Streamlit**: An open-source app framework used to create beautiful, interactive web apps for machine learning and data science projects with minimal code.

---

## 📚 Essential Subtopics

### 1. **Building a REST API with FastAPI**
FastAPI allows you to expose your machine learning model as a REST endpoint so that other applications (like Streamlit or mobile apps) can send data and receive predictions.

#### Code Example: Basic FastAPI App

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputData):
    # Dummy model prediction (replace with actual model)
    result = f"Prediction for: {data.text}"
    return {"result": result}
```

Run with:
```bash
uvicorn main:app --reload
```

Access interactive docs at `http://127.0.0.1:8000/docs`.

---

### 2. **Creating a Frontend with Streamlit**
Streamlit provides an easy way to build UIs that interact with your FastAPI backend.

#### Code Example: Streamlit Frontend

```python
import streamlit as st
import requests

st.title("AI Prediction App")

user_input = st.text_input("Enter text for prediction")

if st.button("Predict"):
    response = requests.post("http://127.0.0.1:8000/predict", json={"text": user_input})
    prediction = response.json()["result"]
    st.success(f"Model says: {prediction}")
```

Run with:
```bash
streamlit run frontend.py
```

---

### 3. **Model Serialization and Loading**
Before deploying, you need to save your trained model (e.g., using `pickle`, `joblib`, or `torch.save`).

#### Example: Saving and Loading a Model

```python
import joblib

# Save model
joblib.dump(model, "model.pkl")

# Load model
model = joblib.load("model.pkl")
```

---

## 🛠️ When and Where We Use It

### Use Cases:
- **Model serving**: Expose your model via REST API for real-time predictions.
- **Interactive demos**: Use Streamlit to build quick UIs for model testing and showcasing.
- **Internal tools**: Build apps for internal teams to interact with models without needing to know the backend.

### Coding Example: Calling FastAPI from Streamlit

Already shown above, but the key part is the `requests.post()` call that sends user input to FastAPI and displays the result.

---

## 🌍 Real-World Applications and Examples

### 1. **Healthcare Diagnosis App**
- **Backend**: FastAPI serves a medical image classification model.
- **Frontend**: Streamlit allows doctors to upload images and get diagnostic predictions.

### 2. **Customer Support Chatbot**
- **Backend**: FastAPI serves a conversational AI model.
- **Frontend**: Streamlit simulates a chat interface for users to interact with the bot.

### 3. **Sentiment Analysis Dashboard**
- **Backend**: FastAPI serves a sentiment analysis model.
- **Frontend**: Streamlit dashboard lets users enter reviews and get sentiment scores.

---

## ⚠️ Common Problems and How Advancements Solved Them

### Problem 1: **Cold Start Latency**
- **Issue**: Serverless or containerized apps may have high latency on the first request after inactivity.
- **Solution**: Use **AWS Lambda Provisioned Concurrency** or keep containers warm with periodic health checks.

#### Problem Code:
```python
# Model loads on every request
@app.post("/predict")
def predict(data: InputData):
    model = joblib.load("model.pkl")  # Slow!
    return {"result": model.predict([data.text])}
```

#### Enhancement Code:
```python
# Load model once at startup
model = joblib.load("model.pkl")

@app.post("/predict")
def predict(data: InputData):
    return {"result": model.predict([data.text])}
```

---

### Problem 2: **Scalability**
- **Issue**: Single-instance FastAPI can't handle high traffic.
- **Solution**: Use **container orchestration (Docker + Kubernetes)** or serverless platforms.

#### Dockerfile Example:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn streamlit requests
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
```

---

### Problem 3: **Lack of Authentication**
- **Issue**: Public endpoints can be abused.
- **Solution**: Use **OAuth2**, **API keys**, or **JWT tokens**.

#### FastAPI Security Example:
```python
from fastapi import Depends, HTTPException, security

oauth2_scheme = security.HTTPBearer()

def get_current_user(token: str = Depends(oauth2_scheme)):
    if token.credentials != "secret-token-123":
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

@app.post("/predict")
def predict(data: InputData, user: str = Depends(get_current_user)):
    return {"result": "Authenticated prediction"}
```

---

## 🧪 Exercise: Serve Model via REST

**Task**: Create a FastAPI endpoint that serves a trained model (e.g., a text classifier). Then, build a Streamlit frontend that sends input to this endpoint and shows predictions.

---

## 📌 Project: Complete AI Web App

Combine everything into a **fully deployed AI web app**:
- Train a model (e.g., sentiment analysis).
- Expose it via FastAPI.
- Build a Streamlit frontend.
- Add authentication and Dockerize the app.
- Deploy it on a cloud platform (e.g., Heroku, AWS, GCP).

---

Awesome! Let's now walk through building and deploying a **complete AI web app** using **FastAPI + Streamlit**, step by step. This will include:

1. **Training a simple model** (Sentiment Analysis)
2. **Building a FastAPI backend** to serve the model
3. **Creating a Streamlit frontend** to interact with the API
4. **Adding authentication** (optional but recommended)
5. **Dockerizing the app**
6. **Deploying to the cloud** (we’ll use Heroku as an example)

---

## 🧠 Step 1: Train a Simple Model (Sentiment Analysis)

We'll use the `IMDb` movie review dataset and train a basic `LogisticRegression` model using `scikit-learn`.

### Code: `train_model.py`

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

# Load dataset (replace with your own dataset)
data = pd.read_csv("IMDB Dataset.csv")  # Columns: review, sentiment

# Preprocess labels
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# Build pipeline
model = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression()
)

# Train
model.fit(data['review'], data['sentiment'])

# Save model
joblib.dump(model, "sentiment_model.pkl")
```

Run this script once to generate `sentiment_model.pkl`.

---

## 🚀 Step 2: Build FastAPI Backend

We'll create a FastAPI app that loads the trained model and exposes a `/predict` endpoint.

### Code: `main.py`

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load model
model = joblib.load("sentiment_model.pkl")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    prediction = model.predict([input.text])[0]
    sentiment = "positive" if prediction == 1 else "negative"
    return {"text": input.text, "sentiment": sentiment}
```

Run with:
```bash
uvicorn main:app --reload
```

You can test it at `http://127.0.0.1:8000/docs`.

---

## 🎨 Step 3: Build Streamlit Frontend

Now let's build a simple UI using Streamlit that sends user input to the FastAPI backend and shows the prediction.

### Code: `frontend.py`

```python
import streamlit as st
import requests

st.title("🎬 Movie Review Sentiment Analysis")

user_input = st.text_area("Enter a movie review:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"text": user_input}
        )
        result = response.json()
        sentiment = result["sentiment"]
        st.success(f"Sentiment: **{sentiment.upper()}**")
```

Run with:
```bash
streamlit run frontend.py
```

---

## 🔐 Step 4: Add Authentication (Optional)

Let's secure our FastAPI endpoint with a simple API key check.

### Code: `main.py` (Enhanced)

```python
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

app = FastAPI()
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != "my-secret-key":
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.post("/predict")
async def predict(input: TextInput, api_key: str = Depends(get_api_key)):
    prediction = model.predict([input.text])[0]
    sentiment = "positive" if prediction == 1 else "negative"
    return {"text": input.text, "sentiment": sentiment}
```

Now you must include the header `X-API-KEY: my-secret-key` in requests.

---

## 🐳 Step 5: Dockerize the App

We'll create a `Dockerfile` and `docker-compose.yml` to containerize both FastAPI and Streamlit.

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### requirements.txt

```
fastapi
uvicorn
streamlit
requests
scikit-learn
pandas
joblib
python-multipart
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - API_KEY=my-secret-key

  frontend:
    build: .
    command: streamlit run frontend.py --server.port=8501
    ports:
      - "8501:8501"
    depends_on:
      - api
```

Run with:
```bash
docker-compose up
```

---

## ☁️ Step 6: Deploy to Heroku

### 1. Install Heroku CLI and login

```bash
heroku login
```

### 2. Create Heroku app

```bash
heroku create your-app-name
```

### 3. Add `Procfile` for Heroku

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### 4. Push to Heroku

```bash
git init
git add .
git commit -m "Deploy AI app"
heroku git:remote -a your-app-name
git push heroku main
```

### 5. Open the app

```bash
heroku open
```

---

## 🧪 Exercise: Try It Yourself

- Train your own model (e.g., image classifier, spam detector).
- Replace the `predict` logic in FastAPI.
- Modify the Streamlit UI to accept different input types (e.g., image upload).
- Try deploying on Render, Railway, or AWS instead of Heroku.

---

