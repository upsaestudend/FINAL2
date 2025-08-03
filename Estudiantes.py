import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Título de la app
st.title("📚 Predicción de la Nota Final del Estudiante")

# Verificar existencia del modelo entrenado
modelo_path = "modelo_entrenado.pkl"
if not os.path.exists(modelo_path):
    st.error("❌ No se encontró el modelo entrenado. Ejecute primero 'entrenar_modelo.py'.")
    st.stop()

# Cargar el modelo
modelo = joblib.load(modelo_path)

# Verificar existencia del dataset
data_path = "calificaciones_1000_estudiantes_con_id.csv"
if not os.path.exists(data_path):
    st.error(f"❌ No se encontró el archivo '{data_path}'.")
    st.stop()

# Cargar dataset para estadísticas
df = pd.read_csv(data_path)

# Recalcular bono y nota final en el dataset
df["Bono"] = np.where(df["Asistencia"] > 95, df["TP"] * 0.20, 0)
df["TP_Modificado"] = df["TP"] + df["Bono"]
df["Final_Usado"] = np.where(df["Asistencia"] < 80, 0, df["Examen_Final"])
df["Nota_Final_Calculada"] = (
    0.1333 * df["Parcial_1"] +
    0.1333 * df["Parcial_2"] +
    0.1333 * df["Parcial_3"] +
    0.20 * df["TP_Modificado"] +
    0.40 * df["Final_Usado"]
).round(1)

# Clasificación de notas
def clasificar(nota):
    if nota >= 91:
        return "Excelente"
    elif nota >= 81:
        return "Óptimo"
    elif nota >= 71:
        return "Satisfactorio"
    elif nota >= 61:
        return "Bueno"
    elif nota >= 51:
        return "Regular"
    else:
        return "Insuficiente"

df["Clasificacion"] = df["Nota_Final_Calculada"].apply(clasificar)

# Generar matriz_confusion.png si no existe
def generar_matriz_confusion():
    if not os.path.exists("matriz_confusion.png"):
        X = df[["Parcial_1", "Parcial_2", "Parcial_3", "Asistencia"]]
        y_real = df["Nota_Final_Calculada"]
        y_pred = modelo.predict(X)

        y_real_clas = y_real.apply(clasificar)
        y_pred_clas = pd.Series(y_pred).apply(clasificar)

        labels = ["Excelente", "Óptimo", "Satisfactorio", "Bueno", "Regular", "Insuficiente"]
        cm = confusion_matrix(y_real_clas, y_pred_clas, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap="Blues")
        plt.title("Matriz de Confusión")
        plt.savefig("matriz_confusion.png")
        plt.close()
        print("✅ matriz_confusion.png generada.")

# Llamar generación
generar_matriz_confusion()

# Formulario de entrada
st.sidebar.header("✍️ Ingrese los datos del estudiante")

parcial_1 = st.sidebar.slider("Parcial 1", 0.0, 100.0, 70.0)
parcial_2 = st.sidebar.slider("Parcial 2", 0.0, 100.0, 70.0)
parcial_3 = st.sidebar.slider("Parcial 3", 0.0, 100.0, 70.0)
asistencia = st.sidebar.slider("Porcentaje de Asistencia", 0.0, 100.0, 85.0)

# Simular TP y Final con promedio general
tp_promedio = df["TP"].mean()
final_promedio = df["Examen_Final"].mean()

# Calcular bono si aplica
bono = tp_promedio * 0.20 if asistencia > 95 else 0
tp_modificado = tp_promedio + bono
final_usable = 0 if asistencia < 80 else final_promedio

# Armar vector de entrada para el modelo
X_nuevo = pd.DataFrame({
    "Parcial_1": [parcial_1],
    "Parcial_2": [parcial_2],
    "Parcial_3": [parcial_3],
    "Asistencia": [asistencia]
})

# Predicción
nota_predicha_raw = modelo.predict(X_nuevo)[0]
nota_predicha = max(0, min(100, nota_predicha_raw))
clasificacion = clasificar(nota_predicha)

# Mostrar resultado
st.subheader("📈 Resultados de la predicción")
st.write(f"📝 **Nota final estimada:** {nota_predicha:.1f}")
st.write(f"🏅 **Clasificación:** {clasificacion}")

# Mostrar gráficos
st.subheader("📈 Estadísticas del dataset")

col1, col2 = st.columns(2)

with col1:
    st.write("Distribución de clasificaciones:")
    clas_counts = df["Clasificacion"].value_counts().reindex(
        ["Excelente", "Óptimo", "Satisfactorio", "Bueno", "Regular", "Insuficiente"]
    ).fillna(0)
    fig1, ax1 = plt.subplots()
    clas_counts.plot(kind="bar", ax=ax1, color='skyblue')
    ax1.set_ylabel("Cantidad de estudiantes")
    ax1.set_title("Clasificaciones")
    st.pyplot(fig1)

with col2:
    st.write("Distribución de notas finales:")
    fig2, ax2 = plt.subplots()
    sns.histplot(df["Nota_Final_Calculada"], bins=20, kde=True, ax=ax2, color='orange')
    ax2.set_title("Histograma de Notas Finales")
    st.pyplot(fig2)

# Mostrar matriz de confusión guardada
st.subheader("📌 Matriz de Confusión del Modelo")
if os.path.exists("matriz_confusion.png"):
    st.image("matriz_confusion.png", caption="Comparación de clasificaciones reales vs. predichas")
else:
    st.info("ℹ️ No se encontró 'matriz_confusion.png'.")
