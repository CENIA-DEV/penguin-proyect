import os
import joblib
from typing import Dict, Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.utils.constants import (
    DATASET_FILE_PATH, CATEGORICAL_COLUMNS, 
    SPECIES_MAPPING, SPECIES_INVERSE_MAPPING)
from src.utils.preprocessing import load_dataset, remove_nulls
from src.utils.normalization import normalize_dataframe, one_hot_encode, map_categorical_to_numeric

MODEL_SAVE_PATH = 'models/trained_model.pkl'
ENCODER_SAVE_PATH = 'models/label_mapping.pkl'  #guarda SPECIES_INVERSE_MAPPING 

def split_data(df, target_column:str="species", test_size=0.2, random_state=42):
    """Divide el DataFrame en conjuntos de entrenamiento y prueba."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_decision_tree(X_train, y_train):
    """Crea y entrena un modelo de árbol de decisión."""
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

def metrics(y_true, y_pred):
    """Calcula métricas de evaluación."""
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted'),
            "recall": recall_score(y_true, y_pred, average='weighted')}

def save_model(model, species_inverse_mapping: Dict[int, str], model_path:str, encoder_path:str):
    """Guarda el modelo entrenado y el mapeo inverso de especies."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(species_inverse_mapping, ENCODER_SAVE_PATH)

def run_training(df, species_inverse_mapping: Dict[int, str]) -> Dict[str, Any]:
    """Ejecuta el proceso completo de entrenamiento del modelo."""
    #Dividir datos
    X_train, X_test, y_train, y_test = split_data(df)
    #Entrenar modelo
    model = train_decision_tree(X_train, y_train)
    y_pred = model.predict(X_test)
    eval_metrics = metrics(y_test, y_pred)
    print("Métricas de evaluación:", eval_metrics)
    save_model(model, species_inverse_mapping, MODEL_SAVE_PATH, ENCODER_SAVE_PATH)

    return {"status": "success", "training_samples": len(X_train), "test_samples": len(X_test), "metrics": eval_metrics}


#------------------
#.     API
#------------------

def train_api():
    """Función para entrenar el modelo desde una API endpoint."""
    #Cargar y limpiar datos
    df = load_dataset(DATASET_FILE_PATH)
    df = df[df['sex'].isin(['MALE', 'FEMALE'])]
    df = df[['species', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
    df = remove_nulls(df)

    #Normalizar
    df, feature_mean, feature_std = normalize_dataframe(df)
    joblib.dump((feature_mean, feature_std), 'models/normalization_params.pkl')

    #Codificar
    df['sex'] = df['sex'].astype(str)
    df = one_hot_encode(df, CATEGORICAL_COLUMNS)
    df = map_categorical_to_numeric(df, 'species', SPECIES_MAPPING)

    #Entrenar modelo
    results = run_training(df, SPECIES_INVERSE_MAPPING)
    return results

