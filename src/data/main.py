
import pandas as pd
from src.utils.preprocessing import load_dataset, remove_nulls, check_duplicates
from src.utils.normalization import normalize_dataframe, one_hot_encode, map_categorical_to_numeric
from src.utils.constants import CATEGORICAL_COLUMNS, SPECIES_MAPPING, DATASET_FILE_PATH

pd.set_option('display.max_columns', None)
#----------------------------------
#analisis exploratorio del dataset
#----------------------------------

#Cargar dataset
penguin_dataset = load_dataset(DATASET_FILE_PATH)
training_df = penguin_dataset.loc[:, ('species', 'island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex')]
print('total de filas: {0}\n\n'.format(len(training_df.index)))
print(training_df.describe(include='all'))

#Revisar duplicados y eliminar valores nulos   
training_df = remove_nulls(training_df)
print('Total de filas después de eliminar nulos: {0}\n'.format(len(training_df.index)))
check_duplicates(training_df)
print('Valores duplicados: {0}\n'.format(training_df.duplicated().sum()))

#Normalización y codificación
training_df = normalize_dataframe(training_df)
print('DataFrame normalizado:\n', training_df.head())



encoded_df = one_hot_encode(training_df, CATEGORICAL_COLUMNS)
print('DataFrame con codificación one-hot:\n', encoded_df.head())
final_df = map_categorical_to_numeric(encoded_df, 'species', SPECIES_MAPPING)
print('DataFrame con especies mapeadas a valores numéricos:\n', final_df.head())

