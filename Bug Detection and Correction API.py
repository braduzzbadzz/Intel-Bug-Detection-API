# Bug Detection and Correction API - FastAPI App with Public Tunneling Support

# Required packages: fastapi, uvicorn, joblib, scikit-learn, pyngrok, localtunnel, datasets, jinja2, nest_asyncio
# Install packages with:
# pip install fastapi uvicorn joblib scikit-learn pyngrok datasets jinja2 nest_asyncio

import joblib
import uvicorn
import numpy as np
import nest_asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from pydantic import BaseModel
import re
from typing import Optional
import os
import subprocess
import time

# Apply nest_asyncio to allow event loop within Jupyter/interactive environments
nest_asyncio.apply()

# Define FastAPI and CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Template directory setup
if not os.path.exists("templates"):
    os.makedirs("templates")

# Basic HTML UI
with open("templates/index.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html>
<head><title>Bug Detection API</title></head>
<body>
    <h1>Bug Detection and Correction API</h1>
    <p>Enter your code below to check for bugs and get suggestions for fixing it:</p>
</body>
</html>
""")

templates = Jinja2Templates(directory="templates")

# Pydantic models
class CodeInput(BaseModel):
    code: str

class CodeOutput(BaseModel):
    prediction: str
    confidence: float
    has_error: bool
    fixed_code: Optional[str] = None
    error_details: Optional[str] = None

# Load or train model
def load_or_train_model():
    try:
        model = joblib.load("bug_fix_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        print("✅ Loaded existing model and vectorizer.")
    except FileNotFoundError:
        print("🔧 Training new model...")
        dataset = load_dataset("google/code_x_glue_cc_clone_detection_big_clone_bench")
        df = dataset["train"].to_pandas().sample(n=10000, random_state=42)
        X = df["func1"] + " " + df["func2"]
        y = df["label"].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        vectorizer = CountVectorizer(max_features=10000)
        X_train_vec = vectorizer.fit_transform(X_train)
        model = LinearSVC(dual=False)
        model.fit(X_train_vec, y_train)
        joblib.dump(model, "bug_fix_model.pkl")
        joblib.dump(vectorizer, "vectorizer.pkl")
        accuracy = model.score(vectorizer.transform(X_test), y_test)
        print(f"📈 Model trained with accuracy: {accuracy:.4f}")
    return model, vectorizer

# Basic error fixer
common_errors = [
    (r'(\w+)\s*=\s*(\w+)\s*\+\s*(\d+);', r'\1 = \2 + \3'),
    # Add more regex rules as needed
]

def fix_code(code):
    fixed_code = code
    error_details = []
    for pattern, replacement in common_errors:
        matches = re.findall(pattern, fixed_code)
        if matches:
            fixed_code = re.sub(pattern, replacement, fixed_code)
            error_details.append(f"Fixed pattern: {pattern}")
    return fixed_code, error_details

# Public URL exposure using localtunnel
def setup_tunnel():
    try:
        print("🌐 Setting up public URL via localtunnel...")
        process = subprocess.Popen(
            ["npx", "localtunnel", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        time.sleep(5)
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if "url is:" in output.lower() or "your url is:" in output.lower():
                url = output.strip().split()[-1]
                print(f"\n🔗 ACCESS YOUR API HERE: {url}\n")
                break
    except Exception as e:
        print(f"❌ Error setting up tunnel: {e}")

# Optional ngrok setup (requires auth token)
def setup_ngrok():
    try:
        from pyngrok import ngrok
        # ngrok.set_auth_token("YOUR_AUTH_TOKEN")  # Set your token here
        public_url = ngrok.connect(8000).public_url
        print(f"🔗 Public API URL via ngrok: {public_url}")
        return public_url
    except Exception as e:
        print(f"❌ Error setting up ngrok: {e}")
        return None

# FastAPI routes
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=CodeOutput)
async def predict(code_input: CodeInput):
    fixed_code, error_details = fix_code(code_input.code)
    transformed_code = vectorizer.transform([code_input.code])
    prediction = model.predict(transformed_code)[0]
    confidence = abs(model.decision_function(transformed_code)[0])
    result = "Buggy Code" if prediction == 1 else "Fixed Code"
    return CodeOutput(
        prediction=result,
        confidence=float(confidence),
        has_error=prediction == 1,
        fixed_code=fixed_code if prediction == 1 or error_details else code_input.code,
        error_details=", ".join(error_details) if error_details else None
    )

# Run server
def start_server():
    setup_tunnel()  # You can switch to setup_ngrok() if needed
    print("🚀 Starting Bug Detection API at http://localhost:8000")
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    model, vectorizer = load_or_train_model()
    start_server()
