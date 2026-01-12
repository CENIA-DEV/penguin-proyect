import pandas as pd

def normalize_dataframe(df):
    """Normaliza las columnas numéricas del DataFrame usando Z-Score."""
    feature_mean = df.mean(numeric_only=True)
    feature_std = df.std(numeric_only=True)
    numerical_features = df.select_dtypes('number').columns
    normalized_df = (
        df[numerical_features] - feature_mean)/feature_std
    #Reemplazar las columnas numéricas originales con las normalizadas
    df[numerical_features] = normalized_df
    return df, feature_mean, feature_std

def one_hot_encode(df, columns):
    """Aplica one-hot encoding a las columnas categóricas especificadas."""
    return pd.get_dummies(df, columns=columns)

def map_categorical_to_numeric(df, column, mapping):
    """Mapea una columna categórica a valores numéricos según el diccionario de mapeo proporcionado."""
    df[column + '_num'] = df[column].map(mapping)
    return df
