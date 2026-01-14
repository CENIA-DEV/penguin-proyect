# Penguin Species Prediction API

## 1. Descripción del Proyecto

### Propósito y Objetivos

Este proyecto implementa una API REST para predecir la especie de pingüinos basada en características físicas como:
- culmen_length_mm
- culmen_depth_mm
- flipper_length_mm
- body_mass_g
- sex

El modelo utiliza un **Decision Tree Classifier** entrenado con el dataset de Palmer Penguins, el cual es capaz de identificar tres especies: Adelie, Chinstrap y Gentoo.

### Tecnologías Utilizadas

- **Desarrollo:** Python 3.14
- **ML:** Scikit-learn, Pandas, NumPy
- **Backend:** FastAPI + Uvicorn
- **Containerización:** Docker
- **Persistencia:** Docker Volumes

---

## 2. Instalación y Ejecución Local

### Requisitos Previos

- Python 3.14.2
- pip
- Virtual environment (pyenv)

### Pasos para Configurar el Entorno Virtual

```bash
# 1. Clonar el repositorio
git clone <URL_DEL_REPOSITORIO>
cd penguin-proyect

# 2. Crear el entorno virtual
python3 -m venv ceniaenv

# 3. Activar el entorno virtual
source ceniaenv/bin/activate  # En macOS/Linux

```

### Instalación de Dependencias

```bash
# Actualizar pip
pip install --upgrade pip

# Instalar dependencias del proyecto
pip install -r requirements.txt
```

### Comando para Ejecutar la Aplicación

```bash
# Desde el directorio app/
cd app

# Ejecutar con uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

La API se encuentra disponible en: `http://localhost:8000`

---

## 3. Uso con Docker

### Requisitos Previos

- Docker instalado y en funcionamiento

### Comando para Construir la Imagen

```bash
# Desde la raíz del proyecto
docker build -t penguin1 .
```

### Comando para Ejecutar el Contenedor

```bash
# Crear volumen para persistencia de modelos (realizar una sola vez)
docker volume create penguin_models

# Ejecutar el contenedor
# Esto debe ser con un mapeo de los puertos para que el localhost exponga al docker
docker run -p 8000:8000 -v penguin_models:/app/app/models penguin1
```

### Acceso a la API en Docker

- **URL base:** `http://localhost:8000`
- **Documentación interactiva:** `http://localhost:8000/docs` (Swagger UI)

---

## 4. Documentación de Endpoints

### 4.1 GET `/` - Información de la API

**Descripción:** Devuelve información general sobre la API y los endpoints disponibles.

**Método HTTP:** `GET`

**Parámetros:** Ninguno

**Respuesta:**
```json
{
  "message": "Bienvenido a Penguin Species Prediction API",
  "version": "1.0.0",
  "endpoints": {
    "docs": "/docs",
    "health": "/health",
    "predict": "/predict",
    "train": "/train",
    "test": "/test"
  }
}
```

---

### 4.2 GET `/health` - Verificar Estado

**Descripción:** Verifica que la API está funcionando correctamente.

**URL:** `http://127.0.0.1:8000/health`

**Método HTTP:** `GET`

**Parámetros:** Ninguno

**Respuesta:**
```json
{
  "status": "API is running"
}
```

---

### 4.3 POST `/train` - Entrenar Modelo

**Descripción:** Entrena un nuevo modelo Decision Tree con el dataset de pingüinos.

**URL:** `http://127.0.0.1:8000/train`

**Método HTTP:** `POST`

**Parámetros:** Ninguno

**Respuesta:**
```json
{
  "message": "modelo entrenado exitosamente",
  "results": {
    "status": "success",
    "training_samples": 266,
    "test_samples": 67,
    "metrics": {
      "accuracy": 1.0,
      "precision": 1.0,
      "recall": 1.0
    }
  }
}
```
### 4.4 GET `/test` - Evaluar Modelo

**Descripción:** Evalúa el modelo entrenado con el dataset de prueba.

**URL:** `http://127.0.0.1:8000/test`

**Método HTTP:** `GET`

**Parámetros:** Ninguno

**Respuesta:**
```json
{
  "message": "evaluación completada exitosamente",
  "metrics": {
    "accuracy": 1.0,
    "precision": 1.0,
    "recall": 1.0
  }
}
```
### 4.5 POST `/predict` - Predicción de Especie

**Descripción:** Predice la especie de un pingüino según sus características físicas.

**URL:** `http://127.0.0.1:8000/predict`

**Método HTTP:** `POST`

**Content-Type:** `application/json`

**Parámetros de Entrada:**
```json
{
  "culmen_length_mm": 40.0,
  "culmen_depth_mm": 18.0,
  "flipper_length_mm": 200.0,
  "body_mass_g": 4000,
  "sex": "MALE"
}
```


**Respuesta:**
```json
{
  "predicted species": "Adelie",
}
```

---


---

## 5. Ejemplos de Uso

### 5.1 Usando cURL

#### Ejemplo 1: Verificar que la API está activa
```bash
curl http://localhost:8000/health
```

**Respuesta:**
```json
{"status": "API is running"}
```

#### Ejemplo 2: Entrenar el modelo
```bash
curl -X POST http://localhost:8000/train
```

**Respuesta:**
```json
{
  "message": "modelo entrenado exitosamente",
  "results": {
    "status": "success",
    "training_samples": 266,
    "test_samples": 67,
    "metrics": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0}
  }
}
```

#### Ejemplo 3: Hacer una predicción
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "culmen_length_mm": 40.0,
    "culmen_depth_mm": 18.0,
    "flipper_length_mm": 200.0,
    "body_mass_g": 4000,
    "sex": "MALE"
  }'
```

**Respuesta:**
```json
{
  "predicted species": "Adelie",
}
```

#### Ejemplo 4: Evaluar el modelo
```bash
curl http://localhost:8000/test
```

**Respuesta:**
```json
{
  "message": "evaluación completada exitosamente",
  "metrics": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0}
}
```


### 5.3 Flujo de Trabajo Típico

```
1. Iniciar contenedor
   docker run -p 8000:8000 -v penguin_models:/app/app/models penguin1

2. Verificar que la API está activa
   curl http://localhost:8000/health

3. Entrenar el modelo (primera vez)
   curl -X POST http://localhost:8000/train

4. Hacer predicciones
   curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"culmen_length_mm": 40, "culmen_depth_mm": 18, "flipper_length_mm": 200, "body_mass_g": 4000, "sex": "MALE"}'

5. Ver métricas
   curl http://localhost:8000/test

6. Acceder a documentación interactiva
   http://localhost:8000/docs

7. Detener contenedor
   Ctrl + C
```
---