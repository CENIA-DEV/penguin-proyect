import pandas as pd
import numpy as np
from src.utils.constants import FEATURE_COLUMNS
from src.data.main import final_df
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


#Dividir el dataset en entrenamiento y test
X = final_df[FEATURE_COLUMNS]
y = final_df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Crear el modelo de árbol de decisión
clf = DecisionTreeClassifier(random_state=42)

#Entrenar el modelo
clf = clf.fit(X_train, y_train)

#Hacer predicciones
y_pred = clf.predict(X_test)