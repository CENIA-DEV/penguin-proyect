import pandas as pd

def normalize_dataframe(df):
    """Normaliza las columnas numéricas del DataFrame usando Z-Score."""
    feature_mean = df.mean(numeric_only=True)
    feature_std = df.std(numeric_only=True)
    numerical_features = df.select_dtypes('number').columns

    #Reemplazar las columnas numéricas originales con las normalizadas
    normalized_df = df.copy()
    normalized_df[numerical_features] = (df[numerical_features] - feature_mean)/feature_std
    
    return normalized_df, feature_mean, feature_std

def normalize_new_data(df, feature_mean, feature_std):
    """Normaliza un nuevo DataFrame usando medias y desviaciones estándar proporcionadas."""
    numerical_features = feature_mean.index
    df_normalized = df.copy()
    df_normalized[numerical_features] = (df[numerical_features] - feature_mean)/feature_std
    return df_normalized

def one_hot_encode(df, columns):
    """Aplica one-hot encoding a las columnas categóricas especificadas."""
    return pd.get_dummies(df, columns=columns, prefix=columns, prefix_sep='_')

def map_categorical_to_numeric(df, column, mapping):
    """Mapea una columna categórica a valores numéricos según el diccionario de mapeo proporcionado."""
    
    df = df.copy()
    df[column] = df[column].map(mapping)
    return df
