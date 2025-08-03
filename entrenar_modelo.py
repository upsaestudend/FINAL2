import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("calificaciones_1000_estudiantes_con_id.csv")

df = df[df["Asistencia"] >= 80].copy()
df["Bono"] = np.where(df["Asistencia"] > 95, df["TP"] * 0.2, 0)
df["TP_Modificada"] = df["TP"] + df["Bono"]
df["Final_Modificada"] = df["Examen_Final"]

df["Nota_Final_Recalculada"] = (
    (df["Parcial_1"] + df["Parcial_2"] + df["Parcial_3"]) * (40 / 300) +
    df["TP_Modificada"] * 0.20 +
    df["Final_Modificada"] * 0.40
).round(1)

X = df[["Parcial_1", "Parcial_2", "Parcial_3", "Asistencia"]]
y = df["Nota_Final_Recalculada"]

modelo = LinearRegression()
modelo.fit(X, y)

joblib.dump(modelo, "modelo_entrenado.pkl")
print("✅ modelo_entrenado.pkl generado correctamente")