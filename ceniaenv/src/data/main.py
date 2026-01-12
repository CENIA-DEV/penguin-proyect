import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

pd.set_option('display.max_columns', None)
#----------------------------------
#analisis exploratorio del dataset 
#---------------------------------- 

#Importar dataset 
penguin_dataset = pd.read_csv("ceniaenv/src/data/penguins_size.csv")
#print(penguin_dataset.head())

#Seleccionar las columnas que se usarán
training_df = penguin_dataset.loc[:, ('species', 'island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex')]
print('total de filas: {0}\n\n'.format(len(training_df.index)))
#print(training_df.head(344))

#Revisar estadísticas descriptivas
print(training_df.describe(include='all'))

#Revisar valores nulos
print('\nValores nulos por columna:\n{0}\n'.format(training_df.isnull().sum()))
#No se presentan tantos valores nulos

#Eliminar filas con valores nulos
training_df = training_df.dropna()
print('Total de filas después de eliminar nulos: {0}\n'.format(len(training_df)))

#Revisar valores duplicados
print('Valores duplicados: {0}\n'.format(training_df.duplicated().sum()))
#No se presentan valores duplicados

#Revisar valores unicos en columnas categóricas
print(training_df['sex'].unique())
print(training_df['species'].unique())
print(training_df['island'].unique())

#Revisamos matriz de correlación
correlation_matrix = training_df.corr(numeric_only=True)
print('Matriz de correlación:\n{0}\n'.format(correlation_matrix))

#plt.figure(figsize=(10, 8))
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
#plt.title('Matriz de Correlación')
#plt.show()

#body_mass_g y flipper_length_mm tienen una correlación alta de 0.87
#flipper_length_mm y culmen_length_mm tienen una correlación moderada de 0.68

#Visualización de distribuciones
#sns.pairplot(training_df, hue='species')
#plt.suptitle('Distribuciones de características por especie', y=1.02)
#plt.show()
#Se observa que las especies tienen distribucion normal

#Normalizar los datos numéricos calculando el z-score de cada columna numérica
feature_mean = training_df.mean(numeric_only=True)
feature_std = training_df.std(numeric_only=True)
numerical_features = training_df.select_dtypes('number').columns
normalized_df = (
    training_df[numerical_features] - feature_mean)/feature_std
#Reemplazar las columnas numéricas originales con las normalizadas
training_df[numerical_features] = normalized_df
print('Datos normalizados:\n{0}\n'.format(training_df.head()))

#One-hot encoding para variables categóricas
categorical_features = {
    'island': ['Biscoe', 'Dream', 'Torgersen'],
    'sex': ['Male', 'Female', '.']}
encoded_df = pd.get_dummies(training_df, columns=['island', 'sex'])
print('Datos después de one-hot encoding:\n{0}\n'.format(encoded_df.head()))

#Transformar la columna 'species' a etiquetas numéricas
species_mapping = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
encoded_df['species_num'] = encoded_df['species'].map(species_mapping)
print('Datos con etiquetas numéricas para species_num:\n{0}\n'.format(encoded_df.sample(10)))

#Separar en características y target
feature_cols = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm',
       'body_mass_g', 'island_Biscoe', 'island_Dream', 'island_Torgersen',
       'sex_.', 'sex_FEMALE', 'sex_MALE', 'species_num']

X = encoded_df[feature_cols] # Características
y = encoded_df['species'] # Target

#Dividir el dataset en entrenamiento, validación y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

