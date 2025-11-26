# En esta etapa vamos a preparar los datos para el modelado. 
# Esto incluye: 
#  - Manejar valores nulos y duplicados, eliminar features irrelevantes, etc.
#  - Codificar variables categóricas (one-hot encoding, label encoding, etc.)
#  - Escalar características numéricas
#  - Dividir los datos en conjuntos de entrenamiento y prueba.
#  - Evitar data leakage asegurándonos de que las transformaciones se ajusten solo en el conjunto de entrenamiento 
#    (usando la media/moda del entrenamiento para imputar nulos en el test, por ejemplo).
#  - Guardar los datos preprocesados y los conjuntos de entrenamiento y prueba.
# 
# 1. Guardar el dataset original (raw data, que ya podría estar guardado en data/raw/)
# 2. También vamos a guardar los datos preprocesados para su uso posterior en el modelado.
# 3. Y por otro lado guardaremos los datasets de entrenamiento y prueba por separado.

import pandas as pd
import numpy as np
import joblib as jl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer # Para transformaciones específicas de columnas, en vez de hacer manualmente el Encoding y el Scaler
from model_utils import save_cleaned_data
import os

# Devuelve nulos por columna y total de filas duplicadas
def resumen_nulos_duplicados(df):
    nulos = df.isnull().sum()
    duplicados = df.duplicated().sum()
    print(f"Nulos por columna:\n{nulos}\n")
    print(f"Total de filas duplicadas: {duplicados}\n")

    return nulos, duplicados


# Limpiamos los datos: eliminar duplicados y eliminar columnas irrelevantes, SIN IMPUTAR NULOS PARA EVITAR DATA LEAKAGE
# Esta la usamos en el pipeline
def clean_data(df, file_name='heart_disease_v1', irrelevant_features=None, output_dir='data/interim/'):


    # Eliminar filas duplicadas
    df = df.drop_duplicates()

    #  Eliminar columnas irrelevantes, aunque esto también lo analizaríamos en EDA, esto sería para un pipeline automático
    # Llegado el caso, ajustar los nombres de irrelevant_columns según el dataset específico
    if irrelevant_features is not None:
        for col in irrelevant_features:
            if col in df.columns:
                df = df.drop(columns=[col])
        
    return df # Devuelve el DataFrame limpio

# Esta función será utilizada dentro de preprocesar_datos para imputar nulos una vez separados los conjuntos de entrenamiento y prueba
def imputar_nulos(X_train, X_test, output_dir='data/interim/'):
    df_train = X_train.copy()
    df_test = X_test.copy()
    # Imputar nulos en columnas numéricas con la media
    num_cols = df_train.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_train[col].isnull().sum() > 0:
            mean_value = df_train[col].mean() # Calcular la media en el conjunto de entrenamiento
            df_train[col].fillna(mean_value, inplace=True) # Imputar en conjunto de entrenamiento
            df_test[col].fillna(mean_value, inplace=True) # Imputar en conjunto de prueba con la media del entrenamiento (MUY IMPORTANTE PARA EVITAR DATA LEAKAGE)

    # Imputar nulos en columnas categóricas con la moda
    cat_cols = df_train.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df_train[col].isnull().sum() > 0:
            mode_value = df_train[col].mode()[0] # Moda del conjunto de entrenamiento (sería el valor más frecuente)
            df_train[col].fillna(mode_value, inplace=True) # Imputar en conjunto de entrenamiento
            df_test[col].fillna(mode_value, inplace=True) # Imputar en conjunto de prueba con la moda del entrenamiento (MUY IMPORTANTE PARA EVITAR DATA LEAKAGE)
    
    return df_train, df_test

# Preprocesar los datos: codificar categóricas, escalar numéricas
# Será usada en el train_pipeline
def preprocess_data(cleaned_df, target_column='HeartDisease'):

    X = cleaned_df.drop(columns=[target_column])
    y = cleaned_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Imputación sin leakage
    X_train, X_test = imputar_nulos(X_train, X_test)

    # Guardar cleaned previo al encoding
    cleaned_df = pd.concat([X_train, y_train], axis=1)
    save_cleaned_data(cleaned_df, "cleaned_data.csv", output_dir="data/interim/")

    # Columnas numéricas y categóricas
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

    # Definir el preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
        ]
    )

    # Transformaciones finales (fit_transform en train, transform en test, sino es DATA LEAKAGE)
    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)

    # Convertir a DataFrames de nuevo (ya que ColumnTransformer devuelve arrays numpy)
    X_train_final = pd.DataFrame(X_train_final, columns=preprocessor.get_feature_names_out())
    X_test_final = pd.DataFrame(X_test_final, columns=preprocessor.get_feature_names_out())

    # Guardar procesados
    save_preprocessed_data(
        X_train_final, X_test_final, y_train, y_test,
        preprocessor,
        output_dir="data/processed/"
    )

    return X_train_final, X_test_final, y_train, y_test


def save_preprocessed_data(X_train, X_test, y_train, y_test, preprocessor, output_dir='data/processed/', preprocessor_dir='preprocessors/'):
    os.makedirs(output_dir, exist_ok=True)

    existing = os.listdir(output_dir)
    version_preprocessed_data = sum(1 for f in existing if f.startswith("v") and "_X_train" in f) + 1

    X_train.to_csv(os.path.join(output_dir, f'v{version_preprocessed_data}_X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, f'v{version_preprocessed_data}_X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, f'v{version_preprocessed_data}_y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, f'v{version_preprocessed_data}_y_test.csv'), index=False)

    # joblib para guardar el preprocessor
    os.makedirs(preprocessor_dir, exist_ok=True) # Si el directorio no existe, joblib no puede guardar el archivo allí.
    version_preprocessor = len([f for f in os.listdir(preprocessor_dir) if f.startswith("v") and f.endswith("_preprocessor.joblib")]) + 1
    jl.dump(preprocessor, os.path.join(preprocessor_dir, f'v{version_preprocessor}_preprocessor.joblib'))

