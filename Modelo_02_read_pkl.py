import pickle
import pandas as pd


# =========================
# 1. Cargar modelo
# =========================
with open('modelo_xgboost_churn.pkl', 'rb') as f:
    modelo = pickle.load(f)

print("\nModelo cargado correctamente\n")


# =========================
# 2. Input interactivo
# =========================
datos = {}

print("Ingrese los datos del cliente:\n")

# Numéricas
datos['tenure'] = int(input("tenure (meses): "))
datos['monthly_charges'] = float(input("monthly_charges: "))
datos['total_charges'] = float(input("total_charges: "))
datos['support_calls'] = int(input("support_calls: "))

# Categóricas
datos['contract'] = input("contract (Month-to-month / One year / Two year): ")
datos['payment_method'] = input("payment_method: ")
datos['internet_service'] = input("internet_service (DSL / Fiber optic / No): ")
datos['tech_support'] = input("tech_support (Yes / No): ")
datos['online_security'] = input("online_security (Yes / No): ")


# =========================
# 3. DataFrame
# =========================
nueva_muestra = pd.DataFrame([datos])

print("\nDatos ingresados:")
print(nueva_muestra)


# =========================
# 4. Predicción
# =========================
pred = modelo.predict(nueva_muestra)[0]
prob = modelo.predict_proba(nueva_muestra)[0, 1]


# =========================
# 5. Resultado
# =========================
print("\n============================")
if pred == 1:
    print("⚠️  Cliente con ALTO riesgo de churn")
else:
    print("✅ Cliente con BAJO riesgo de churn")

print(f"Probabilidad de churn: {prob:.4f}")
print("============================\n")
