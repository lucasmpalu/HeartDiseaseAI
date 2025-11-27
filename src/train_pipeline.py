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

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier  # si lo tenés instalado

models = {
    "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
    "SVC": SVC(probability=True, random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "ExtraTrees": ExtraTreesClassifier(random_state=42),
}


# Grilla de hiperparámetros ampliada
param_grid = {
    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs", "liblinear"]
    },
    "SVC": {
        "C": [0.1, 0.5, 1, 5, 10],
        "kernel": ["linear", "rbf", "poly"],
        "degree": [2, 3, 4],
        "gamma": ["scale", "auto"]
    },
    "RandomForest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "GradientBoosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4, 5],
        "subsample": [0.8, 1]
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200, 400],
        "learning_rate": [0.01, 0.1, 1.0]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 11, 15, 21],
        "weights": ["uniform", "distance"],
        "p": [1, 2]  # Manhattan o Euclidiana
    },
    "ExtraTrees": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10]
    }
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
    evaluated_models = evaluate_models_crossval(models, param_grid, X_train, y_train, folds=5)

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
