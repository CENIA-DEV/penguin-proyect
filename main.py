
import pandas as pd
import joblib
from src.utils.preprocessing import load_dataset, remove_nulls, check_duplicates
from src.utils.normalization import normalize_dataframe, one_hot_encode, map_categorical_to_numeric
from src.utils.constants import CATEGORICAL_COLUMNS, SPECIES_MAPPING, DATASET_FILE_PATH
from src.models.train import run_training
from src.utils.constants import SPECIES_INVERSE_MAPPING

pd.set_option('display.max_columns', None)
#----------------------------------
#analisis exploratorio del dataset
#----------------------------------

#Cargar dataset
penguin_dataset = load_dataset(DATASET_FILE_PATH)
penguin_dataset = penguin_dataset[
    penguin_dataset['sex'].isin(['MALE', 'FEMALE'])
]


training_df = penguin_dataset[['species', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
print('total de filas: {0}\n\n'.format(len(training_df.index)))
print(training_df.describe(include='all'))

#Revisar duplicados y eliminar valores nulos 
print("Valores únicos en 'sex':", training_df['sex'].unique())  

training_df = remove_nulls(training_df)
print('Total de filas después de eliminar nulos: {0}\n'.format(len(training_df.index)))
check_duplicates(training_df)
print('Valores duplicados: {0}\n'.format(training_df.duplicated().sum()))


#Normalización y codificación
training_df, feature_mean, feature_std = normalize_dataframe(training_df)
print('DataFrame normalizado:\n', training_df.head())

joblib.dump((feature_mean, feature_std), 'models/normalization_params.pkl')
training_df['sex'] = training_df['sex'].astype(str)
encoded_df = one_hot_encode(training_df, CATEGORICAL_COLUMNS)
print('DataFrame con codificación one-hot:\n', encoded_df.head())

final_df = map_categorical_to_numeric(encoded_df, 'species', SPECIES_MAPPING)
print('DataFrame con especies mapeadas a valores numéricos:\n', final_df.head())

#----------------------------------
#Entrenamiento del modelo
#----------------------------------
# Verificar que 'species' sea numérico
print("Tipo de 'species' en final_df:", final_df['species'].dtype)
print("Valores únicos en 'species':", final_df['species'].unique())

training_results = run_training(final_df, SPECIES_INVERSE_MAPPING)
print('Resultados del entrenamiento:', training_results)


