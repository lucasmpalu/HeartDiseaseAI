# Esta función guarda el modelo entrenado en disco usando joblib
import os
import joblib as jl

def save_model(trained_model, algoritm_name: str, output_dir: str = 'models/'):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    existing_files = os.listdir(output_dir)
    model_path = os.path.join(output_dir, f'v{len(existing_files)+1}_model_{algoritm_name}.joblib')
    jl.dump(trained_model, model_path) # Modelo en disco y nombre del archivo
    print(f"Modelo guardado en: {model_path}")

def load_last_model(input_dir="models/"):

    existing_files = [
        f for f in os.listdir(input_dir)
        if f.startswith("v") and "_model_" in f and f.endswith(".joblib")
    ]

    if not existing_files:
        raise FileNotFoundError(f"No hay modelos en {input_dir}")

    versioned = []
    
    for f in existing_files:
        try:
            # Como el archivo empieza vX_model_..., separo por "_" y tomo la primera parte
            version_str = f.split("_")[0]   # devuelve 'v3' si el archivo es 'v3_model_...'
            version = int(version_str.replace("v", ""))  # borro la 'v' y convierto a entero, me queda el
            versioned.append((f, version))
        except:
            # Ignora archivos que no siguen el formato esperado
            pass

    if not versioned:
        raise ValueError("No se encontraron modelos con formato '_vX.joblib'.")

    # Selecciono el archivo con la versión más alta
    # Explicacion del codigo max(versioned, key=lambda x: x[1]):
    # La función max() toma un iterable (en este caso, la lista 'versioned') y un argumento opcional 'key' que es una función que extrae un valor para comparar
    # En este caso, 'key=lambda x: x[1]' significa que queremos comparar los elementos de 'versioned' basándonos en el segundo elemento de cada tupla (que es la versión del modelo)
    last_model_file = max(versioned, key=lambda x: x[1])[0]

    print(f"📦 Cargando modelo: {last_model_file}")

    return jl.load(os.path.join(input_dir, last_model_file))


def load_preprocessor(version: int = None, input_dir: str = 'preprocessors/'):
    

    if version is None:
        existing_files = os.listdir(input_dir)
        version = max([int(f.split('v')[1].split('_preprocessor.joblib')[0]) for f in existing_files if f.startswith("v") and f.endswith("preprocessor.joblib")])
        
    preprocessor_path = os.path.join(input_dir, f'v{version}_preprocessor.joblib')
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"El preprocesador versión {version} no se encontró en {input_dir}")
    preprocessor = jl.load(preprocessor_path)
    return preprocessor

def save_cleaned_data(cleaned_df, end_file_name, output_dir='data/interim/'):
    os.makedirs(output_dir, exist_ok=True)

    existing_files = os.listdir(output_dir)
    # Buscar la versión correcta, empieza con v (de versión) seguido por el número y _cleaned_data.csv     save_cleaned_data(cleaned_df, "_cleaned_data.csv", output_dir="data/interim/")
    # Viene sin v y sin número, se lo tengo que agregar yo, el inicio no es end_file_name, sino el final del nombre del archivo/cadena
    version = sum(1 for f in existing_files if f.endswith(end_file_name)) + 1
    
    cleaned_df.to_csv(os.path.join(output_dir, f"v{version}_{end_file_name}"), index=False)
    print(f"Datos guardados en: {os.path.join(output_dir, f'v{version}_{end_file_name}')}")