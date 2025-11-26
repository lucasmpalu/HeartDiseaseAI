# src/train_pipeline.py

import pandas as pd
import os

from data_prep import clean_data, preprocess_data
from train import evaluate_models_crossval, select_champion
from model_utils import save_model

#############################################
# 1. CONFIGURACIONES DEL PIPELINE
#############################################

# Debo ejecutar este pipeline desde la raíz del proyecto, sino, la ruta relativa no funciona
# SIEMPRE que vamos a usar un nuevo dataset raw, hay que cargar manualmente el archivo en data/raw/ con el formato y v
quantity_datasets = len([f for f in os.listdir("data/raw/") if f.startswith("df_original_v")])
RAW_DATA_PATH = f"data/raw/df_original_v{quantity_datasets}.csv"


# Modelos y grids para GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

models = {
    "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
    "SVC": SVC(probability=True, random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42)
}

param_grid = {
    "LogisticRegression": {
        "C": [0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"],
        "random_state": [42]
    },
    "SVC": {
        "C": [0.5, 1, 10],
        "kernel": ["linear", "rbf"],
        "random_state": [42]
    },
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "random_state": [42]
    },
}

#############################################
# 2. FUNCIÓN PRINCIPAL DEL PIPELINE
#############################################

def train_pipeline(use_ensemble=False):
    print("\n==============================")
    print(" 🚀 INICIANDO PIPELINE ML")
    print("==============================\n")

    # ------------------------------
    # A. Cargar datos crudos
    # ------------------------------
    print("📥 Cargando dataset...")
    df = pd.read_csv(RAW_DATA_PATH)

    # ------------------------------
    # B. Limpiar datos
    # ------------------------------
    print("🧹 Limpiando datos...")
    df_clean = clean_data(df, irrelevant_features=['PatientID'])

    # ------------------------------
    # C. Preprocesar (split + imputación + escalado + encoding)
    #   y guardar datos limpios antes de procesar y procesados
    # ------------------------------
    print("⚙️ Preprocesando y guardando datos...")
    X_train, X_test, y_train, y_test = preprocess_data(df_clean)

    # ------------------------------
    # D. Entrenar modelos con GridSearchCV
    # ------------------------------
    print("🤖 Entrenando modelos con GridSearchCV...")
    evaluated_models = evaluate_models_crossval(models, param_grid, X_train, y_train)

    # ------------------------------
    # E. Seleccionar campeón (o ensemble)
    # ------------------------------
    print("🏆 Seleccionando modelo campeón...")
    best_name, best_model = select_champion(
        evaluated_models,
        X_test, y_test,
        ensemble=use_ensemble,
        X_train=X_train,
        y_train=y_train
    )

    # ------------------------------
    # F. Guardar modelo final
    # ------------------------------
    print(f"💾 Guardando modelo final: {best_name}...")
    save_model(best_model, best_name)

    print("\n==============================")
    print(" 🎯 PIPELINE COMPLETO")
    print("==============================\n")

    return best_name, best_model


#############################################
# 3. EJECUCIÓN DIRECTA
#############################################
if __name__ == "__main__":
    # Ejecutar el pipeline completo (en este caso sin ensemble)
    train_pipeline(use_ensemble=False)
    print("Pipeline ejecutado correctamente.")
