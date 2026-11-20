import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Spotify Popularity ML API",
    description="API de inferencia (XGBoost) y métricas de equidad para predicción de popularidad en Spotify.",
    version="1.0.0"
)

# Rutas de los artefactos generados por 02_train.py
RUTA_MODELO = "models/pipeline_xgboost.joblib"
RUTA_METRICAS = "models/metricas_modelo.json"

# Cargar el pipeline en memoria al iniciar
if not os.path.exists(RUTA_MODELO):
    raise RuntimeError(f"No se encontró el modelo en {RUTA_MODELO}. Ejecuta src/02_train.py primero.")

modelo = joblib.load(RUTA_MODELO)

# DTO de entrada (mismas columnas que usó 02_train.py)
class CancionInput(BaseModel):
    danceability: float = Field(..., ge=0.0, le=1.0, example=0.73)
    energy: float = Field(..., ge=0.0, le=1.0, example=0.81)
    loudness: float = Field(..., example=-5.2)
    speechiness: float = Field(..., ge=0.0, le=1.0, example=0.06)
    acousticness: float = Field(..., ge=0.0, le=1.0, example=0.12)
    instrumentalness: float = Field(..., ge=0.0, le=1.0, example=0.0)
    liveness: float = Field(..., ge=0.0, le=1.0, example=0.15)
    valence: float = Field(..., ge=0.0, le=1.0, example=0.65)
    tempo: float = Field(..., gt=0.0, example=122.0)
    duration_ms: int = Field(..., gt=0, example=210000)
    time_signature: int = Field(..., ge=3, le=7, example=4)
    explicit: int = Field(..., ge=0, le=1, example=0)
    track_genre: str = Field(..., example="pop")

@app.get("/")
def health_check():
    return {
        "estado": "OK",
        "mensaje": "API de Spotify ML funcionando correctamente. Ve a /docs para probar los endpoints."
    }

@app.get("/metricas")
def obtener_metricas():
    if not os.path.exists(RUTA_METRICAS):
        raise HTTPException(status_code=404, detail="No se encontró metricas_modelo.json")
    with open(RUTA_METRICAS, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/predict")
def predecir_popularidad(cancion: CancionInput):
    # Convertir el JSON validado a DataFrame con las columnas exactas
    df_entrada = pd.DataFrame([cancion.model_dump()])
    
    prediccion = float(modelo.predict(df_entrada)[0])
    # Asegurar que quede en el rango real de Spotify (0 a 100)
    prediccion = round(max(0.0, min(100.0, prediccion)), 2)
    
    return {
        "popularidad_predicha": prediccion,
        "es_hit_potencial": prediccion >= 70.0,
        "genero": cancion.track_genre
    }