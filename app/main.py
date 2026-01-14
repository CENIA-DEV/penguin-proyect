from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from src.models.predict import predict_species
from src.models.train import train_api
from src.models.evaluate import evaluate_model


app = FastAPI(
    title="Penguin Species Prediction API",
    description="API para predecir la especie de pingüinos basada en características físicas.",
    version="1.0.0"
)

# ruta raíz
@app.get("/")
def read_root():
    return {
        "message": "Bienvenido a Penguin Species Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "train": "/train",
            "test": "/test"
        }
    }

# crear clase de modelo de datos para la entrada
class PenguinFeatures(BaseModel):
    culmen_length_mm: float
    culmen_depth_mm: float
    flipper_length_mm: float
    body_mass_g: int
    sex: str

@app.get("/health")
def health_check():
    return {"status": "API is running"}

# definir endpoint de predicción
# definir endpoint de entrenamiento
@app.post("/train")
def train_model():
    try:
        results = train_api()
        return {"message": "modelo entrenado exitosamente", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# agregar el endpoint a test
@app.get("/test")
def test_api():
    try:
        metrics = evaluate_model()
        return {"message": "evaluación completada exitosamente", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    
    

@app.post("/predict")
def predict_penguin_species(features: PenguinFeatures):
    try:
        result = predict_species(features.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
