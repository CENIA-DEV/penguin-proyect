import pandas as pd

def load_dataset(file_path):
    """Carga el dataset desde un archivo CSV."""
    return pd.read_csv(file_path)

def remove_nulls(df):
    """Elimina filas con valores nulos del DataFrame."""
    return df.dropna()

def check_duplicates(df):
    """Revisa y devuelve el número de filas duplicadas en el DataFrame."""
    return df.duplicated().sum()

def get_unique_values(df, columns):
    """Devuelve los valores únicos de las columnas categóricas."""
    return {col: df[col].unique() for col in columns}