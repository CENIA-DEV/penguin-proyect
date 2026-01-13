import joblib
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.utils.preprocessing import load_dataset, remove_nulls
from src.utils.constants import DATASET_FILE_PATH, SPECIES_MAPPING, CATEGORICAL_COLUMNS
from src.utils.normalization import normalize_dataframe, one_hot_encode, map_categorical_to_numeric
from sklearn.model_selection import train_test_split

MODEL_PATH = 'models/trained_model.pkl'
ENCODER_PATH = 'models/label_mapping.pkl'  #ruta del mapeo invers

# Cargar el modelo entrenado y el mapeode especies
def load_test_data():
    """Carga el conjunto de datos de prueba y el modelo entrenado."""
    test_df = load_dataset(DATASET_FILE_PATH) #ruta del dataset de prueba
    test_df = test_df[['species', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
    test_df = remove_nulls(test_df)
    test_df = normalize_dataframe(test_df)
    test_df = one_hot_encode(test_df, CATEGORICAL_COLUMNS)
    test_df = map_categorical_to_numeric(test_df, 'species', SPECIES_MAPPING)

    X = test_df.drop(columns=['species'])
    y = test_df['species']

    #Dividir el dataset como en train
    #solo se necesita el conjunto de prueba
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   
    return X_test, y_test

def evaluate_model():
    """Evalúa el modelo entrenado en el conjunto de datos de prueba."""
    #Cargar datos de prueba
    X_test, y_test = load_test_data()

    #Cargar modelo entrenado
    model = joblib.load(MODEL_PATH)

    #Hacer predicciones
    y_pred = model.predict(X_test)

    #Calcular métricas
    eval_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted')
    }

    print("Métricas de evaluación del modelo en el conjunto de prueba:", eval_metrics)
    return eval_metrics