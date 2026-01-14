import joblib
import pandas as pd
from src.utils.normalization import one_hot_encode
from src.utils.constants import CATEGORICAL_COLUMNS
from src.utils.normalization import normalize_new_data

MODEL_PATH = 'models/trained_model.pkl'
ENCODER_PATH = 'models/label_mapping.pkl'  #ruta del mapeo invers
NORMALIZATION_PARAMS_PATH = 'models/normalization_params.pkl'

def preprocess_input(input_df: dict) -> pd.DataFrame:
    """Preprocesa el DataFrame de entrada para la predicción."""
    pred_df = pd.DataFrame([input_df])
    
    # Aplicar one-hot encoding
    pred_df = one_hot_encode(pred_df, CATEGORICAL_COLUMNS)
    
    # Cargar parámetros de normalización
    feature_mean, feature_std = joblib.load(NORMALIZATION_PARAMS_PATH)
    
    # Normalizar características numéricas
    pred_df = normalize_new_data(pred_df, feature_mean, feature_std)
    
    # Alinear columnas con el modelo
    model = joblib.load(MODEL_PATH)
    trained_columns = model.feature_names_in_
    for col in trained_columns:
        if col not in pred_df.columns:
            pred_df[col] = 0
    pred_df = pred_df[trained_columns]
    return pred_df

def predict_species(input_data: dict) -> dict:
    """Realiza la predicción de la especie dada una entrada."""
    # Preprocesar datos de entrada
    X = preprocess_input(input_data)

    # Cargar modelo entrenado y mapeo inverso
    model = joblib.load(MODEL_PATH)
    print("Columnas esperadas por el modelo:", model.feature_names_in_)
    species_inverse_mapping = joblib.load(ENCODER_PATH)

    # Hacer predicción
    pred_numeric = model.predict(X)[0]
    predicted_species = species_inverse_mapping[pred_numeric]

    return {"predicted_species": predicted_species}

if __name__ == "__main__":
    sample = {
        "culmen_length_mm": 39.1,
        "culmen_depth_mm": 18.7,
        "flipper_length_mm": 181.0,
        "body_mass_g": 3750,
        "sex": "FEMALE"
    }
    print(predict_species(sample))