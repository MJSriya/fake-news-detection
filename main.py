from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import os

app = FastAPI()

# Mount static folder
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class NewsRequest(BaseModel):
    text: str

# Load ML components
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except Exception as e:
    print(f"Failed to load model: {e}")
    model, vectorizer = None, None

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict_news(request: NewsRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="ML Model not loaded.")
    
    if not request.text or request.text.strip() == "":
        raise HTTPException(status_code=400, detail="Empty text provided.")
    
    try:
        vec_input = vectorizer.transform([request.text])
        prediction = model.predict(vec_input)[0]
        
        return {
            "prediction": prediction.capitalize(),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
