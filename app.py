import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Predictor F1 - Posicion Final", page_icon="🏎️", layout="centered")


@st.cache_resource
def cargar_artefactos():
    modelo = joblib.load("modelo_regresion_lineal.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
    columnas = joblib.load("columnas_modelo.pkl")
    circuito_a_pais = joblib.load("circuito_a_pais.pkl")
    return modelo, scaler, encoders, columnas, circuito_a_pais


modelo, scaler, encoders, columnas, circuito_a_pais = cargar_artefactos()

st.title("🏎️ Predictor de posición final en F1")
st.write(
    "Introduce los datos de un piloto en una carrera y el modelo de Regresión Lineal "
    "(R² en test 2024: 0.72) estimará en qué posición terminaría."
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Datos pre-carrera")
    season = st.number_input("Temporada", min_value=2022, max_value=2026, value=2024)
    ronda = st.number_input("Ronda", min_value=1, max_value=24, value=1)
    driverId = st.selectbox("Piloto", options=sorted(encoders["driverId"].classes_))
    constructorId = st.selectbox("Escudería (ID)", options=sorted(encoders["constructorId"].classes_))
    constructorName = st.selectbox("Escudería (nombre)", options=sorted(encoders["constructorName"].classes_))
    driver_nationality = st.selectbox("Nacionalidad del piloto", options=sorted(encoders["driver_nationality"].classes_))
    constructor_nationality = st.selectbox("Nacionalidad de la escudería", options=sorted(encoders["constructor_nationality"].classes_))
    circuitId = st.selectbox("Circuito (ID)", options=sorted(encoders["circuitId"].classes_))
    circuitName = st.selectbox("Circuito (nombre)", options=sorted(encoders["circuitName"].classes_))
    country = circuito_a_pais[circuitName]
    st.caption(f"País del circuito: **{country}**")
    number = st.number_input("Número del coche", min_value=1, max_value=99, value=1)
    grid = st.number_input("Posición de salida (grid)", min_value=1, max_value=20, value=1)
    quali_position = st.number_input("Posición en clasificación", min_value=1, max_value=20, value=1)
    q1 = st.number_input("Tiempo Q1 (segundos)", min_value=0.0, value=90.0)
    q2 = st.number_input("Tiempo Q2 (segundos)", min_value=0.0, value=89.0)
    q3 = st.number_input("Tiempo Q3 (segundos)", min_value=0.0, value=88.0)

with col2:
    st.subheader("Datos en directo")
    lap = st.number_input("Vuelta actual", min_value=1, max_value=80, value=1)
    position_en_vuelta = st.number_input("Posición en esa vuelta", min_value=1, max_value=20, value=1)
    paradas_hasta_ahora = st.number_input("Paradas hechas hasta ahora", min_value=0, max_value=5, value=0)
    ultima_pit_duration = st.number_input("Duración última parada (segundos)", min_value=0.0, value=0.0)

st.markdown("---")

if st.button("Predecir posición final", type="primary"):

    entrada = pd.DataFrame([{
        "season": season,
        "round": ronda,
        "driverId": encoders["driverId"].transform([driverId])[0],
        "constructorId": encoders["constructorId"].transform([constructorId])[0],
        "constructorName": encoders["constructorName"].transform([constructorName])[0],
        "driver_nationality": encoders["driver_nationality"].transform([driver_nationality])[0],
        "constructor_nationality": encoders["constructor_nationality"].transform([constructor_nationality])[0],
        "circuitId": encoders["circuitId"].transform([circuitId])[0],
        "circuitName": encoders["circuitName"].transform([circuitName])[0],
        "country": encoders["country"].transform([country])[0],
        "number": number,
        "grid": grid,
        "quali_position": quali_position,
        "Q1_seg": q1,
        "Q2_seg": q2,
        "Q3_seg": q3,
        "lap": lap,
        "position_en_vuelta": position_en_vuelta,
        "paradas_hasta_ahora": paradas_hasta_ahora,
        "ultima_pit_duration_seg": ultima_pit_duration,
    }])

    entrada = entrada[columnas]

    entrada_escalada = pd.DataFrame(
        scaler.transform(entrada),
        columns=entrada.columns
    )

    prediccion = modelo.predict(entrada_escalada)[0]
    prediccion_redondeada = max(1, min(20, round(prediccion)))

    st.success(f"### Posición final estimada: **{prediccion_redondeada}ª**")
    st.caption(f"Valor exacto del modelo (sin redondear): {prediccion:.2f}")

st.markdown("---")
st.caption(
    "Modelo de Regresión Lineal entrenado con datos F1 2022-2024 (R² en test: 0.72). "
    "Proyecto académico — las predicciones no reflejan resultados oficiales de F1."
)

