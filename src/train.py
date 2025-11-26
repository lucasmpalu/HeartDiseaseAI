# En este script vamos a entrenar un modelo de machine learning para predecir enfermedades del corazón.
# Usaremos los datos preprocesados guardados en data/processed/ y los conjuntos de entrenamiento y prueba guardados en data/processed/.
import pandas as pd
import joblib as jl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier

import os

# Esta función evalúa múltiples modelos usando validación cruzada y multiples hiperparámetros
# GridSearchCV permite hacer cross-validation internamente mientras busca los mejores hiperparámetros
# Es decir, cada modelo será evaluado con 5 folds (5 pasadas de diferentes conjuntos de entrenamiento y validación)
# Un diccionario para los modelos y otro para los parametros a evaluar
# Los keys de ambos diccionarios deben coincidir para que funcione correctamente
# El cross

def evaluate_models_crossval(models: dict, param_grid: dict, 
                             X_train: pd.DataFrame, y_train: pd.Series, 
                             folds: int = 5) -> list:

    results = []

    # Evaluar cada modelo con GridSearchCV
    for name, model in models.items():

        # GridSearchCV prueba TODAS las combinaciones de hiperparámetros
        # y evalúa cada una con validación cruzada (K-Fold).
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid[name],
            cv=folds,
            scoring='accuracy'
        )

        # Esto entrena MUCHOS modelos internamente (uno por fold y por combinación).
        # Al final, entrena de nuevo el modelo ganador con TODO X_train.
        grid.fit(X_train, y_train)

        # Modelo final ya reentrenado con TODO el conjunto X_train
        best_model = grid.best_estimator_

        # Accuracy promedio del mejor conjunto de hiperparámetros durante la CV
        best_score = grid.best_score_

        # Hiperparámetros ganadores
        best_params = grid.best_params_

        # Guardar todo
        results.append({
            "name": name,
            "model": best_model,
            "best_score": best_score,
            "best_params": best_params
        })

        print(f"Modelo: {name}")
        print(f"  Mejor Accuracy (CV): {best_score:.4f}")
        print(f"  Mejores Hiperparámetros: {best_params}")
        print("-" * 40)

    return results


# Esta función devuelve el mejor modelo o un modelo ensemble si se especifica
# evaluated_models: lista de diccionarios con 'name', 'model', 'best_score' y 'best_params'
# X_test, y_test: conjunto de prueba para evaluar la performance real
# ensemble: si es True, construye un ensemble con los dos mejores modelos
# X_train, y_train: necesarios SOLO si ensemble=True para reentrenar el ensemble
def select_champion(evaluated_models: list, X_test: pd.DataFrame, y_test: pd.Series, ensemble=False, X_train=None, y_train=None):

    # Calcular performance de todos los modelos con el conjunto de prueba, esta es la performance REAL
    performance = []
    # evaluated_models es una lista de diccionarios con 'name', 'model', 'best_score' y 'best_params' 
    for item in evaluated_models:
        name = item["name"]
        model = item["model"] # model es un modelo ya entrenado/fit
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        performance.append((name, model, acc))

        # Guardar las metricas totales finales en reports/metrics/ por número de versión de modelo, contar los modelos existentes en models/
        existing_models = [f for f in os.listdir("models/") if f.startswith("v") and "_model_" in f and f.endswith(".joblib")]
        version = len(existing_models) + 1
        metrics_path = f"reports/metrics/v{version}_metrics_{name}.txt"
        with open(metrics_path, 'w') as f:
            f.write(f"Classification Report for {name}:\n") 
            f.write(classification_report(y_test, y_pred))
            f.write("\nConfusion Matrix:\n")
            f.write(str(confusion_matrix(y_test, y_pred)))

    print("Resultados en test:")
    for name, model, acc in performance:
        print(f"{name}: {acc:.4f}")

    # Ordenar por accuracy descendente
    performance.sort(key=lambda x: x[2], reverse=True)

    # --- ENSEMBLE --- Si hay más de un modelo, construir un ensemble con los mejores dos o más y devolverlo
    if ensemble and len(performance) >= 2:
        top2 = performance[:2]   # tomar los dos mejores
        print(f"Construyendo ensemble con: {top2[0][0]} y {top2[1][0]}")

        models_dict = {name: model for name, model, acc in top2}
    
        ensemble_model = build_ensemble(models_dict, X_train, y_train)
        return "Ensemble", ensemble_model

    # --- CHAMPION NORMAL --- Devuelvo el modelo ya entrenado con mejor performance en test
    best_name, best_model, acc = performance[0]
    return best_name, best_model


# Esta función construye un modelo ensemble usando VotingClassifier, combinando múltiples modelos ya entrenados
def build_ensemble(models_dict: list, X_train, y_train):
    estimators = [(name, model) for name, model in models_dict.items()]
    ensemble = VotingClassifier(estimators=estimators, voting='soft')   
    ensemble.fit(X_train, y_train)  # Reentrenar el ensemble con todo el conjunto de entrenamiento
    return ensemble

