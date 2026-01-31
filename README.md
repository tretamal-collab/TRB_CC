# Modelo de Predicción de Churn (FastAPI + XGBoost)

Proyecto en **Python** para entrenar y servir un modelo de predicción de **churn (deserción)** usando un artefacto serializado (`.pkl`) y un dataset de ejemplo.

## Incluye
- Scripts de entrenamiento e inferencia
- Dataset `customer_churn_dataset.csv`
- Modelo entrenado `modelo_xgboost_churn.pkl`
- API en **FastAPI** con endpoint `POST /predict`
- Deploy en **GCP Cloud Run**

---

## Contenido
- [Descripción del problema](#descripción-del-problema)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
  - [1) Entrenar el modelo (modelo_01.py)](#1-entrenar-el-modelo-modelo_01py)
  - [2) Cargar modelo y predecir por consola (modelo_02_read_pkl.py)](#2-cargar-modelo-y-predecir-por-consola-modelo_02_read_pklpy)
  - [3) Ejecutar la API localmente (app.py)](#3-ejecutar-la-api-localmente-apppy)
- [Endpoint /predict](#endpoint-predict)
  - [Input esperado](#input-esperado)
  - [Request de ejemplo](#request-de-ejemplo)
  - [Response de ejemplo](#response-de-ejemplo)
- [Deploy en la nube](#deploy-en-la-nube)
- [Licencia](#licencia)

---

## Descripción del problema

El objetivo es predecir si un cliente hará **churn (abandono)** a partir de variables numéricas y categóricas.

### Flujo end-to-end
1. Entrenamiento de un modelo de clasificación (pipeline con preprocesamiento + **XGBoost**).
2. Evaluación con métricas (**AUC, Accuracy, Precision, Recall, F1**).
3. Serialización del pipeline completo a un archivo `.pkl`.
4. Serving del modelo vía **FastAPI** con endpoint `POST /predict`, retornando **predicción + probabilidad**.

---

## Estructura del repositorio

- `customer_churn_dataset.csv` — dataset de ejemplo.
- `modelo_01.py` — entrenamiento, evaluación y exportación del modelo (`modelo_xgboost_churn.pkl`).
- `modelo_02_read_pkl.py` — ejemplo de carga del `.pkl` y predicción por consola.
- `app.py` — API FastAPI para inferencia (`POST /predict`).
- `modelo_xgboost_churn.pkl` — modelo entrenado serializado (pipeline completo).
- `requirements.txt` — dependencias del proyecto.
- `Procfile` — comando de arranque para deploy (Uvicorn).
- `runtime.txt` — versión de Python usada para deploy.
- `LICENSE` — licencia MIT.

---

## Requisitos

- Python 3.10 (según `runtime.txt`: `python-3.10`)
- `pip`
- Recomendado: entorno virtual (`venv`)

---

## Instalación

### 1) Clonar el repositorio + entorno + dependencias
```bash
git clone https://github.com/tretamal-collab/TRB_CC
cd TRB_CC

# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual (elige según tu SO)
# Windows (PowerShell):
#   .venv\Scripts\Activate.ps1
#
# Mac/Linux:
#   source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

---

## Uso

### 1) Entrenar el modelo (modelo_01.py)

Este script:
- Carga `customer_churn_dataset.csv`
- Convierte el target `churn` de `Yes/No` a `1/0`
- Separa train/test con `stratify`
- Construye un pipeline completo con:
  - Imputación numérica (mediana)
  - Imputación categórica ("Sin Info") + `OneHotEncoder(handle_unknown="ignore")`
  - Modelo `XGBClassifier`
- Entrena el modelo y reporta métricas (**AUC, Accuracy, Precision, Recall, F1**)
- Guarda el pipeline entrenado en `modelo_xgboost_churn.pkl`

```bash
# (Requiere que tengas el venv activado)
python modelo_01.py
```

### 2) Cargar modelo y predecir por consola (modelo_02_read_pkl.py)

Este script:
- Carga `modelo_xgboost_churn.pkl`
- Pide inputs por consola
- Construye un DataFrame con el input
- Predice clase y probabilidad

```bash
# (Requiere que tengas el venv activado)
python modelo_02_read_pkl.py
```

### 3) Ejecutar la API localmente (app.py)

La API:
- Carga `modelo_xgboost_churn.pkl` al iniciar (no re-entrena en runtime)
- Expone `POST /predict`
- Calcula `predict_proba` y aplica threshold fijo 0.5
- Devuelve JSON con:
  - `Churn` (bool)
  - `ProbabilidadChurn` (float redondeado a 4 decimales)
  - `Threshold` (0.5)

```bash
# (Requiere que tengas el venv activado)
uvicorn app:app --host 0.0.0.0 --port 8080

# Swagger (local):
#   http://localhost:8080/docs
```

---

## Endpoint /predict

### Input esperado

El endpoint espera un JSON con los siguientes campos:

**Numéricos**
- `tenure` (float)
- `monthly_charges` (float)
- `total_charges` (float)
- `support_calls` (int)

**Categóricos**
- `contract` (str) — típicos: Month-to-month, One year, Two year
- `payment_method` (str) — típicos: Credit, Debit, Cash, UPI
- `internet_service` (str) — típicos: DSL, Fiber, None
- `tech_support` (str) — típicos: Yes, No
- `online_security` (str) — típicos: Yes, No

### Request de ejemplo
```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "monthly_charges": 75.5,
    "total_charges": 905.3,
    "support_calls": 2,
    "contract": "Month-to-month",
    "payment_method": "Credit",
    "internet_service": "DSL",
    "tech_support": "No",
    "online_security": "Yes"
  }'
```

### Response de ejemplo
```json
{
  "Churn": false,
  "ProbabilidadChurn": 0.2134,
  "Threshold": 0.5
}
```

---

## Deploy en la nube

Plataforma cloud usada para el deploy: **GCP Cloud Run** (FastAPI + Uvicorn).

- `runtime.txt`: `python-3.10`
- `Procfile`: `web: uvicorn app:app --host=0.0.0.0 --port=${PORT:-8080}`
- Swagger (cloud): https://trb-cc-487081242882.us-east1.run.app/docs
- Endpoint (cloud): POST https://trb-cc-487081242882.us-east1.run.app/predict

---

## Licencia

MIT. Ver `LICENSE`.