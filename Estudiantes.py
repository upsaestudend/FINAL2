import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Título de la app
st.title("Predicción de la Nota Final del Estudiante")

# Cargar el modelo
modelo = joblib.load("modelo_entrenado.pkl")

# Cargar dataset para estadísticas
df = pd.read_csv("calificaciones_1000_estudiantes_con_id.csv")

# Recalcular bono y nota final en el dataset
df["Bono"] = np.where(df["Asistencia"] > 95, df["TP"] * 0.20, 0)
df["TP_Modificado"] = df["TP"] + df["Bono"]
df["Final_Usado"] = np.where(df["Asistencia"] < 80, 0, df["Final"])
df["Nota_Final_Calculada"] = (
    0.1333 * df["Parcial_1"] +
    0.1333 * df["Parcial_2"] +
    0.1333 * df["Parcial_3"] +
    0.20 * df["TP_Modificado"] +
    0.40 * df["Final_Usado"]
).round(1)

# Clasificación
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

# Formulario de entrada
st.sidebar.header("Ingrese los datos del estudiante")

parcial_1 = st.sidebar.slider("Parcial 1", 0.0, 100.0, 70.0)
parcial_2 = st.sidebar.slider("Parcial 2", 0.0, 100.0, 70.0)
parcial_3 = st.sidebar.slider("Parcial 3", 0.0, 100.0, 70.0)
asistencia = st.sidebar.slider("Porcentaje de Asistencia", 0.0, 100.0, 85.0)

# Simular TP y Final con promedio general
tp_promedio = df["TP"].mean()
final_promedio = df["Final"].mean()

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
nota_predicha = modelo.predict(X_nuevo)[0]
clasificacion = clasificar(nota_predicha)

# Mostrar resultado
st.subheader("Resultados de la predicción")
st.write(f"📝 **Nota final estimada:** {nota_predicha:.1f}")
st.write(f"🏅 **Clasificación:** {clasificacion}")

# Mostrar gráficos
st.subheader("📊 Estadísticas del dataset")

col1, col2 = st.columns(2)

with col1:
    st.write("Distribución de clasificaciones:")
    clas_counts = df["Clasificacion"].value_counts().reindex(
        ["Excelente", "Óptimo", "Satisfactorio", "Bueno", "Regular", "Insuficiente"]
    )
    fig1, ax1 = plt.subplots()
    clas_counts.plot(kind="bar", ax=ax1)
    ax1.set_ylabel("Cantidad de estudiantes")
    ax1.set_title("Clasificaciones")
    st.pyplot(fig1)

with col2:
    st.write("Distribución de notas finales:")
    fig2, ax2 = plt.subplots()
    sns.histplot(df["Nota_Final_Calculada"], bins=20, kde=True, ax=ax2)
    ax2.set_title("Histograma de Notas Finales")
    st.pyplot(fig2)

# Mostrar matriz de confusión guardada (opcional)
st.subheader("📌 Matriz de Confusión del Modelo")
st.image("matriz_confusion.png", caption="Comparación de clasificaciones reales vs. predichas")