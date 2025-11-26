import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

###############################################
# 1. CARGAR ÚLTIMO DATASET RAW 
###############################################

# Esto 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Esto trae la carpeta actual: notebooks/
PROJECT_ROOT = os.path.dirname(BASE_DIR)               # Esto trae la carpeta padre: raiz del proyecto
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")    # Esto trae la carpeta donde está el dataset original


files = [f for f in os.listdir(RAW_DIR) if f.startswith("df_original_v")]
files.sort()

if not files:
    raise FileNotFoundError("No se encontró ningún archivo 'df_original_v' en ../data/raw/")

latest_file = files[-1]
data_path = os.path.join(RAW_DIR, latest_file)

print(f"📂 Cargando dataset: {latest_file}")
data = pd.read_csv(data_path)

###############################################
# 2. CREAR DIRECTORIO DE FIGURAS Y REPORTES 
###############################################

FIG_DIR = os.path.join(PROJECT_ROOT, "reports", "figures", "exploratory", f"dataset_v{len(files)}")
os.makedirs(FIG_DIR, exist_ok=True)

REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "info_data", f"info_dataset_v{len(files)}")
os.makedirs(REPORT_DIR, exist_ok=True)

REPORT_PATH = os.path.join(REPORT_DIR, "eda_summary.txt")


###############################################
# 3. GUARDAR INFORMACIÓN GENERAL DEL DATASET
###############################################

with open(REPORT_PATH, "w", encoding="utf-8") as f:

    f.write("===== EDA SUMMARY =====")
    f.write(f"Dataset cargado: {latest_file}")

    f.write("=== Primeras filas ===")
    f.write(str(data.head(10)))

    f.write("=== Info ===")
    # info() no devuelve string, hay que capturarla
    # capturar el info() en un buffer de texto
    buffer = StringIO()
    data.info(buf=buffer)
    info_str = buffer.getvalue()

    f.write("=== Info ===")
    f.write(info_str)
    f.write("=== Describe ===")
    f.write(str(data.describe(include='all')))

    f.write("=== Cantidad de nulos ===")
    f.write(str(data.isnull().sum()))

    f.write("=== Filas duplicadas ===")
    f.write(str(data.duplicated().sum()))

    f.write("=== Tipos de datos ===")
    f.write(str(data.dtypes))

    if "Age" in data.columns:
        f.write("=== Rango de Age (outliers) ===")
        f.write(str(data['Age'].describe()))

print(f"📄 Archivo generado: {REPORT_PATH}")

###############################################
# 4. FEATURE ENGINEERING 
###############################################

if "Age" in data.columns:
    data['Age_Range'] = pd.cut(
        data['Age'], bins=[29, 40, 55, 77],
        labels=['Young', 'Middle_Aged', 'Senior']
    )

if "Cholesterol" in data.columns:
    data['Cholesterol_Level'] = pd.cut(
        data['Cholesterol'], bins=[0, 200, 240, 600],
        labels=['Desirable', 'Borderline High', 'High']
    )

###############################################
# 5. FUNCIÓN PARA GRAFICAR 
###############################################

def plot_feature_and_target(data, feature, target):

    is_categorical = (data[feature].dtype == 'object') or (data[feature].nunique() <= 10)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # distribución
    if is_categorical:
        sns.countplot(x=feature, data=data, ax=axes[0])
    else:
        sns.histplot(data[feature].dropna(), kde=True, ax=axes[0])

    axes[0].set_title(f'Distribución de {feature}')

    # feature vs target
    if is_categorical:
        sns.countplot(x=feature, hue=target, data=data, ax=axes[1])
    else:
        sns.boxplot(x=target, y=feature, data=data, ax=axes[1])

    axes[1].set_title(f'{feature} vs {target}')

    plt.tight_layout()

    clean_name = feature.replace("/", "_").replace(" ", "_")

    plt.savefig(f"{FIG_DIR}/{clean_name}_analysis.png")
    plt.close()

###############################################
# 6. GENERAR GRÁFICOS
###############################################

if "HeartDisease" not in data.columns:
    raise ValueError("El dataset no contiene la columna 'HeartDisease'.")

for col in data.columns:
    if col != "HeartDisease":
        print(f"📊 Graficando: {col}")
        plot_feature_and_target(data, col, "HeartDisease")

print("\n✅ EDA COMPLETO")
print(f"🖼️ Gráficos guardados en: {FIG_DIR}")

