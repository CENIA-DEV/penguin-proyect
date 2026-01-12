# Columnas del dataset de pingüinos
FEATURE_COLUMNS = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 
                   'body_mass_g', 'island_Biscoe', 'island_Dream', 'island_Torgersen',
                   'sex_.', 'sex_Female', 'sex_Male']
CATEGORICAL_COLUMNS = {'island': ['Biscoe', 'Dream', 'Torgersen'],
                       'sex': ['Male', 'Female', '.']
}

# Mapeo de especies de pingüinos a valores numéricos
SPECIES_MAPPING = {
    'Adelie': 0,
    'Chinstrap': 1,
    'Gentoo': 2
}

# Datasert file path
DATASET_FILE_PATH = 'cenianv/src/data/penguins_size.csv'