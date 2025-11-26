import os
import pandas as pd
import joblib
from src.model_utils import load_last_model, load_last_model, load_preprocessor
# BaseModel es la clase base para crear modelos de datos con validación en Pydantic




# Realiza una predicción para un nuevo paciente dado un diccionario con sus datos
def predict_new_patient(patient_data: pd.DataFrame):

    quantity_preprocessors = len([f for f in os.listdir("preprocessors/") if f.startswith("v") and f.endswith("preprocessor.joblib")])
    quantity_models = len([f for f in os.listdir("models/") if f.startswith("v") and f.endswith(".joblib")])
    if quantity_models == 0 or quantity_preprocessors == 0:
        raise FileNotFoundError("No se encontraron modelos o preprocesadores guardados para hacer predicciones.")
    else:
        preprocessor = load_preprocessor() # Si no se especifica versión, carga la última
        model = load_last_model()


    # Preprocesar
    df_processed = preprocessor.transform(patient_data)

    # Predicción
    prediction = int(model.predict(df_processed)[0])

    # Probabilidad
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(df_processed)[0][1])
    else:
        probability = None  # SVC podría no tenerlo

    return {
        "prediction": prediction,
        "probability": probability
    }
