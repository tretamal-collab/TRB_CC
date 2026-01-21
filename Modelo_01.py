import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


from xgboost import XGBClassifier


# =========================
# 1. Cargar datos
# =========================
df = pd.read_csv('customer_churn_dataset.csv')


# =========================
# 2. Target (OBLIGATORIO 0/1)
# =========================
df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})


# =========================
# 3. Features (ID FUERA)
# =========================
X = df.drop(columns=['churn', 'customer_id'], errors='ignore')
y = df['churn']


# =========================
# 4. DEFINICIÓN EXPLÍCITA DE COLUMNAS
# =========================
cols_num = [
    'tenure',
    'monthly_charges',
    'total_charges',
    'support_calls'
]

cols_cat = [
    'contract',
    'payment_method',
    'internet_service',
    'tech_support',
    'online_security'
]


# =========================
# 5. Preprocesamiento
# =========================
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Sin Info')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, cols_num),
        ('cat', categorical_transformer, cols_cat)
    ],
    remainder='drop'
)


# =========================
# 6. Modelo
# =========================
xgb_model = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    scale_pos_weight=1, 
    eval_metric='logloss', 
    random_state=42
)


# =========================
# 7. Pipeline completo
# =========================
pipeline = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', xgb_model)
])


# =========================
# 8. Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# =========================
# 9. Entrenar
# =========================
pipeline.fit(X_train, y_train)


# =========================
# 10. Evaluación
# =========================
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== MÉTRICAS DEL MODELO =====")
print(f"AUC       : {auc:.4f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print("================================\n")


# =========================
# 11. Guardar PKL
# =========================
with open('modelo_xgboost_churn.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Modelo guardado correctamente: modelo_xgboost_churn.pkl")


