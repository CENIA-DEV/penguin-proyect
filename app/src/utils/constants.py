# Columnas categoricasdel dataset de pingüinos
CATEGORICAL_VALS = {'sex': ['Male', 'Female', '.']
}
CATEGORICAL_COLUMNS = list(CATEGORICAL_VALS.keys())

# Mapeo de especies de pingüinos a valores numéricos
SPECIES_MAPPING = {
    'Adelie': 0,
    'Chinstrap': 1,
    'Gentoo': 2
}
SPECIES_INVERSE_MAPPING = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}
# Dataset file path
DATASET_FILE_PATH = 'data/penguins_size.csv'