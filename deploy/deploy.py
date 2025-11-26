from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
import os
import sys
# Esto asegura que al momento de importar desde deploy.py, el punto de partida sea el directorio raíz del proyecto
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Directorio raíz del proyecto
sys.path.insert(0, ROOT_DIR) 
from src.predict import predict_new_patient
from src.model_utils import load_last_model

# Cargar modelo automáticamente
model = load_last_model("models/")

app = FastAPI(
    title="Heart Disease Prediction API", 
    description="API para predecir riesgo de enfermedad cardíaca usando modelo ML entrenado.",
    version="1.0"
)

# HearthData hereda BaseModel de Pydantic para validación de datos (aseugra que los datos de entrada sean correctos)
# FastAPI funciona igual si no uso esta validación, pero esto asegura
# La petición recibe un JSON con estos campos y tipos de datos 
# y al tipar el parámetro 'data' en la función predict, FastAPI valida automáticamente la entrada (con Pydantic)
class HeartData(BaseModel):
    Age: float
    Sex: str
    ChestPainType: str
    RestingBP: float
    Cholesterol: float
    FastingBS: int
    RestingECG: str
    MaxHR: float
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str


@app.get("/")
def home():
    return {"status": "ok", "message": "API funcionando correctamente."}

# Si falta algun campo o el es tipo es incorrecto, FastAPI devuelve un error 422 y no se ejecuta la función
# Ya que intenta convertir el JSON de entrada a un objeto HeartData y si falla, devuelve el error
@app.post("/predict")
def predict(data: HeartData):

    # Convierto pydantic model a diccionario
    dataToDict = data.model_dump()
    # Convierto diccionario a DataFrame (ya que el modelo espera un DataFrame)
    DictToDataframe = pd.DataFrame([dataToDict])
    # Realizo la predicción
    result = predict_new_patient(data)


    print(f"Predicción: {result['prediction']}, Probabilidad: {result['probability']}")

    return result


ejemplo_input = {
        "Age": 45,
        "Sex": "M",
        "ChestPainType": "ATA",
        "RestingBP": 130,
        "Cholesterol": 233,
        "FastingBS": 1,
        "RestingECG": "Normal",
        "MaxHR": 150,
        "ExerciseAngina": "N",
        "Oldpeak": 2.3,
        "ST_Slope": "Flat"
    }

data_obj = HeartData(**ejemplo_input)

predict(data_obj)

# Para correr la app: uvicorn deploy.deploy:app --reload