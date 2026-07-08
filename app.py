import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Predictor de Puesto en F1", page_icon="🏁", layout="centered")

# ---------------------------------------------------------------------------
# Datos auxiliares (derivados del dataset 2022-2024) para que la interfaz
# muestre nombres reales, colores de escudería y evite listas redundantes.
# ---------------------------------------------------------------------------

DRIVER_INFO = {
    "albon": {"name": "Alexander Albon", "nationality": "Thai", "number": 23, "team": "williams"},
    "alonso": {"name": "Fernando Alonso", "nationality": "Spanish", "number": 14, "team": "aston_martin"},
    "bearman": {"name": "Oliver Bearman", "nationality": "British", "number": 50, "team": "haas"},
    "bottas": {"name": "Valtteri Bottas", "nationality": "Finnish", "number": 77, "team": "sauber"},
    "colapinto": {"name": "Franco Colapinto", "nationality": "Argentine", "number": 43, "team": "williams"},
    "de_vries": {"name": "Nyck de Vries", "nationality": "Dutch", "number": 21, "team": "alphatauri"},
    "doohan": {"name": "Jack Doohan", "nationality": "Australian", "number": 61, "team": "alpine"},
    "gasly": {"name": "Pierre Gasly", "nationality": "French", "number": 10, "team": "alpine"},
    "hamilton": {"name": "Lewis Hamilton", "nationality": "British", "number": 44, "team": "mercedes"},
    "hulkenberg": {"name": "Nico Hülkenberg", "nationality": "German", "number": 27, "team": "haas"},
    "kevin_magnussen": {"name": "Kevin Magnussen", "nationality": "Danish", "number": 20, "team": "haas"},
    "latifi": {"name": "Nicholas Latifi", "nationality": "Canadian", "number": 6, "team": "williams"},
    "lawson": {"name": "Liam Lawson", "nationality": "New Zealander", "number": 30, "team": "rb"},
    "leclerc": {"name": "Charles Leclerc", "nationality": "Monegasque", "number": 16, "team": "ferrari"},
    "max_verstappen": {"name": "Max Verstappen", "nationality": "Dutch", "number": 1, "team": "red_bull"},
    "mick_schumacher": {"name": "Mick Schumacher", "nationality": "German", "number": 47, "team": "haas"},
    "norris": {"name": "Lando Norris", "nationality": "British", "number": 4, "team": "mclaren"},
    "ocon": {"name": "Esteban Ocon", "nationality": "French", "number": 31, "team": "alpine"},
    "perez": {"name": "Sergio Pérez", "nationality": "Mexican", "number": 11, "team": "red_bull"},
    "piastri": {"name": "Oscar Piastri", "nationality": "Australian", "number": 81, "team": "mclaren"},
    "ricciardo": {"name": "Daniel Ricciardo", "nationality": "Australian", "number": 3, "team": "rb"},
    "russell": {"name": "George Russell", "nationality": "British", "number": 63, "team": "mercedes"},
    "sainz": {"name": "Carlos Sainz", "nationality": "Spanish", "number": 55, "team": "ferrari"},
    "sargeant": {"name": "Logan Sargeant", "nationality": "American", "number": 2, "team": "williams"},
    "stroll": {"name": "Lance Stroll", "nationality": "Canadian", "number": 18, "team": "aston_martin"},
    "tsunoda": {"name": "Yuki Tsunoda", "nationality": "Japanese", "number": 22, "team": "rb"},
    "vettel": {"name": "Sebastian Vettel", "nationality": "German", "number": 5, "team": "aston_martin"},
    "zhou": {"name": "Guanyu Zhou", "nationality": "Chinese", "number": 24, "team": "sauber"},
}

TEAM_NAMES = {
    "alfa": "Alfa Romeo",
    "alphatauri": "AlphaTauri",
    "alpine": "Alpine F1 Team",
    "aston_martin": "Aston Martin",
    "ferrari": "Ferrari",
    "haas": "Haas F1 Team",
    "mclaren": "McLaren",
    "mercedes": "Mercedes",
    "rb": "RB F1 Team",
    "red_bull": "Red Bull",
    "sauber": "Sauber",
    "williams": "Williams",
}

TEAM_NATIONALITY = {
    "alfa": "Swiss",
    "alphatauri": "Italian",
    "alpine": "French",
    "aston_martin": "British",
    "ferrari": "Italian",
    "haas": "American",
    "mclaren": "British",
    "mercedes": "German",
    "rb": "Italian",
    "red_bull": "Austrian",
    "sauber": "Swiss",
    "williams": "British",
}

TEAM_COLORS = {
    "alfa": "#981E32",
    "alphatauri": "#2B4562",
    "alpine": "#0090FF",
    "aston_martin": "#229971",
    "ferrari": "#DC0000",
    "haas": "#E6002B",
    "mclaren": "#FF8000",
    "mercedes": "#27F4D2",
    "rb": "#1634CB",
    "red_bull": "#3671C6",
    "sauber": "#52E252",
    "williams": "#64C4FF",
}

CIRCUIT_NAMES = {
    "albert_park": "Albert Park Grand Prix Circuit",
    "americas": "Circuit of the Americas",
    "bahrain": "Bahrain International Circuit",
    "baku": "Baku City Circuit",
    "catalunya": "Circuit de Barcelona-Catalunya",
    "hungaroring": "Hungaroring",
    "imola": "Autodromo Enzo e Dino Ferrari",
    "interlagos": "Autódromo José Carlos Pace",
    "jeddah": "Jeddah Corniche Circuit",
    "losail": "Losail International Circuit",
    "marina_bay": "Marina Bay Street Circuit",
    "miami": "Miami International Autodrome",
    "monaco": "Circuit de Monaco",
    "monza": "Autodromo Nazionale di Monza",
    "red_bull_ring": "Red Bull Ring",
    "ricard": "Circuit Paul Ricard",
    "rodriguez": "Autódromo Hermanos Rodríguez",
    "shanghai": "Shanghai International Circuit",
    "silverstone": "Silverstone Circuit",
    "spa": "Circuit de Spa-Francorchamps",
    "suzuka": "Suzuka Circuit",
    "vegas": "Las Vegas Strip Street Circuit",
    "villeneuve": "Circuit Gilles Villeneuve",
    "yas_marina": "Yas Marina Circuit",
    "zandvoort": "Circuit Park Zandvoort",
}


def pill(color: str, text: str) -> str:
    return (
        f'<span style="background:{color}22;color:{color};border:1px solid {color};'
        f'padding:2px 10px;border-radius:999px;font-weight:600;font-size:0.85rem;">{text}</span>'
    )


@st.cache_resource
def cargar_artefactos():
    modelo = joblib.load("modelo_regresion_lineal.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
    columnas = joblib.load("columnas_modelo.pkl")
    circuito_a_pais = joblib.load("circuito_a_pais.pkl")
    return modelo, scaler, encoders, columnas, circuito_a_pais


modelo, scaler, encoders, columnas, circuito_a_pais = cargar_artefactos()

# ---------------------------------------------------------------------------
# Estilos
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700;900&display=swap');

    html, body, [class*="css"] { font-family: 'Titillium Web', sans-serif; }

    .f1-hero {
        background: linear-gradient(120deg, #15151e 0%, #950014 100%);
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .f1-hero h1 {
        color: #ffffff;
        font-weight: 900;
        font-size: 2.1rem;
        letter-spacing: 0.03em;
        margin: 0 0 0.35rem 0;
    }
    .f1-hero p {
        color: #e6e6e6;
        font-size: 1rem;
        margin: 0;
    }
    .checker-strip {
        height: 10px;
        border-radius: 4px;
        margin: 0.9rem 0 1.4rem 0;
        background-image: repeating-linear-gradient(90deg, #1c1c24 0 12px, #ffffff 12px 24px);
        opacity: 0.85;
    }
    .section-card {
        border-radius: 14px;
        padding: 1.1rem 1.3rem 0.4rem 1.3rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(120,120,120,0.25);
    }
    .section-title {
        font-weight: 700;
        font-size: 1.15rem;
        margin-bottom: 0.1rem;
    }
    .section-desc {
        font-size: 0.88rem;
        opacity: 0.75;
        margin-bottom: 0.8rem;
    }
    div.stButton > button {
        background: #E10600;
        color: white;
        font-weight: 700;
        letter-spacing: 0.03em;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        width: 100%;
    }
    div.stButton > button:hover {
        background: #ff1e14;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="f1-hero">
        <h1>🏁 PREDICTOR DE PUESTO EN F1</h1>
        <p>Introduce los datos de un piloto en una carrera y un modelo de <b>Regresión Lineal</b>
        (R² en test 2024: <b>0.72</b>) estimará en qué posición cruzará la meta.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="checker-strip"></div>', unsafe_allow_html=True)

st.caption(
    "🧠 El modelo combina dos tipos de información: lo que ya sabemos **antes de que se apague el "
    "semáforo** (parrilla, clasificación, circuito) y lo que va llegando **en directo durante la carrera** "
    "(vuelta actual, posición, paradas en boxes)."
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        '<div class="section-title">📋 Antes de la carrera</div>'
        '<div class="section-desc">Datos conocidos antes de la salida: quién corre, con qué coche '
        'y desde qué posición.</div>',
        unsafe_allow_html=True,
    )

    season = st.number_input("Temporada", min_value=2022, max_value=2026, value=2024)
    ronda = st.number_input("Ronda", min_value=1, max_value=24, value=1)

    driver_options = sorted(DRIVER_INFO, key=lambda d: DRIVER_INFO[d]["name"])
    driverId = st.selectbox(
        "Piloto", options=driver_options, format_func=lambda d: DRIVER_INFO[d]["name"], key="driverId"
    )

    if st.session_state.get("_prev_driver") != driverId:
        st.session_state["constructorId"] = DRIVER_INFO[driverId]["team"]
        st.session_state["_prev_driver"] = driverId

    driver_team_color = TEAM_COLORS[DRIVER_INFO[driverId]["team"]]
    st.markdown(
        f'{pill(driver_team_color, TEAM_NAMES[DRIVER_INFO[driverId]["team"]])} '
        f'&nbsp; {DRIVER_INFO[driverId]["nationality"]} · Nº {DRIVER_INFO[driverId]["number"]}',
        unsafe_allow_html=True,
    )

    team_options = sorted(TEAM_NAMES, key=lambda c: TEAM_NAMES[c])
    constructorId = st.selectbox(
        "Escudería", options=team_options, format_func=lambda c: TEAM_NAMES[c], key="constructorId"
    )
    constructorName = TEAM_NAMES[constructorId]
    constructor_nationality = TEAM_NATIONALITY[constructorId]
    driver_nationality = DRIVER_INFO[driverId]["nationality"]
    number = DRIVER_INFO[driverId]["number"]

    team_color = TEAM_COLORS[constructorId]
    st.markdown(pill(team_color, f"Escudería {constructor_nationality}"), unsafe_allow_html=True)

    circuit_options = sorted(CIRCUIT_NAMES, key=lambda c: CIRCUIT_NAMES[c])
    circuitId = st.selectbox(
        "Circuito", options=circuit_options, format_func=lambda c: CIRCUIT_NAMES[c]
    )
    circuitName = CIRCUIT_NAMES[circuitId]
    country = circuito_a_pais[circuitName]
    st.caption(f"📍 País: **{country}**")

    grid = st.number_input("Posición de salida (grid)", min_value=1, max_value=20, value=1)
    quali_position = st.number_input("Posición en clasificación", min_value=1, max_value=20, value=1)
    q1 = st.number_input("Tiempo Q1 (segundos)", min_value=0.0, value=90.0)
    q2 = st.number_input("Tiempo Q2 (segundos)", min_value=0.0, value=89.0)
    q3 = st.number_input("Tiempo Q3 (segundos)", min_value=0.0, value=88.0)

with col2:
    st.markdown(
        '<div class="section-title">🔴 En directo</div>'
        '<div class="section-desc">Datos que cambian vuelta a vuelta mientras la carrera '
        'está en marcha.</div>',
        unsafe_allow_html=True,
    )
    lap = st.number_input("Vuelta actual", min_value=1, max_value=80, value=1)
    position_en_vuelta = st.number_input("Posición en esa vuelta", min_value=1, max_value=20, value=1)
    paradas_hasta_ahora = st.number_input("Paradas hechas hasta ahora", min_value=0, max_value=5, value=0)
    ultima_pit_duration = st.number_input("Duración última parada (segundos)", min_value=0.0, value=0.0)

st.markdown('<div class="checker-strip"></div>', unsafe_allow_html=True)

if st.button("🏎️ Predecir posición final", type="primary"):

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

    medalla = {1: "🥇", 2: "🥈", 3: "🥉"}.get(prediccion_redondeada, "🏁")

    st.markdown(
        f"""
        <div class="section-card" style="border-color:{team_color};background:{team_color}14;">
            <div style="display:flex;align-items:center;gap:0.9rem;">
                <div style="font-size:2.6rem;">{medalla}</div>
                <div>
                    <div style="font-size:0.9rem;opacity:0.8;">
                        {DRIVER_INFO[driverId]["name"]} · {TEAM_NAMES[constructorId]}
                    </div>
                    <div style="font-size:1.8rem;font-weight:800;">
                        Posición final estimada: {prediccion_redondeada}ª
                    </div>
                </div>
            </div>
            <p style="margin-top:0.6rem;opacity:0.7;font-size:0.85rem;">
                Valor exacto del modelo (sin redondear): {prediccion:.2f}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if prediccion_redondeada <= 3:
        st.balloons()

st.markdown('<div class="checker-strip"></div>', unsafe_allow_html=True)
st.caption(
    "Modelo de Regresión Lineal entrenado con datos F1 2022-2024 (R² en test: 0.72). "
    "Proyecto académico — las predicciones no reflejan resultados oficiales de F1."
)
