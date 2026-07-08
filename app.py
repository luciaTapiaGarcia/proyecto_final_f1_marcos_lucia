import json

import streamlit as st
import pandas as pd
import joblib
import streamlit.components.v1 as components

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

# nombre visible (piloto o escudería) -> color de su equipo, para pintar
# cada opción de los desplegables con el color real de su escudería.
NAME_COLOR_MAP = {info["name"]: TEAM_COLORS[info["team"]] for info in DRIVER_INFO.values()}
NAME_COLOR_MAP.update({TEAM_NAMES[tid]: TEAM_COLORS[tid] for tid in TEAM_NAMES})


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
# Estilos — tema F1: fondo rojo/negro, tarjetas de cristal, inputs en blanco.
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700;900&display=swap');

    html, body, [class*="css"] { font-family: 'Titillium Web', sans-serif; }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(160deg, #c40000 0%, #6b0000 45%, #150000 100%) fixed;
    }
    [data-testid="stHeader"] {
        background: #150000;
    }
    [data-testid="stAppViewContainer"] * {
        color: #f5f5f5;
    }

    .f1-hero {
        background: rgba(0,0,0,0.4);
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255,255,255,0.15);
        backdrop-filter: blur(6px);
    }
    .f1-hero h1 {
        color: #ffffff;
        font-weight: 900;
        font-size: 2.2rem;
        letter-spacing: 0.03em;
        margin: 0 0 0.35rem 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }
    .f1-hero p {
        color: #f0f0f0;
        font-size: 1rem;
        margin: 0;
    }
    .checker-strip {
        height: 10px;
        border-radius: 4px;
        margin: 0.9rem 0 1.4rem 0;
        background-image: repeating-linear-gradient(90deg, #0c0c0c 0 12px, #ffffff 12px 24px);
        opacity: 0.9;
    }
    .section-title {
        font-weight: 700;
        font-size: 1.15rem;
        margin-bottom: 0.1rem;
        color: #ffffff;
    }
    .section-desc {
        font-size: 0.88rem;
        opacity: 0.85;
        margin-bottom: 0.8rem;
    }

    /* Tarjetas de cristal alrededor de cada columna */
    .st-key-col1_card, .st-key-col2_card {
        background: rgba(0,0,0,0.38) !important;
        border-radius: 18px !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        padding: 0.6rem 1rem 1.2rem 1rem !important;
        backdrop-filter: blur(6px);
    }

    /* Inputs y selects en blanco, redondeados e interactivos */
    div[data-testid="stNumberInput"] input,
    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border-radius: 10px !important;
        border: 2px solid rgba(255,255,255,0.6) !important;
        transition: box-shadow 0.15s ease, transform 0.15s ease, border-color 0.15s ease;
    }
    div[data-testid="stNumberInput"] input {
        color: #111111 !important;
    }
    div[data-baseweb="select"] * {
        color: #111111 !important;
    }
    div[data-testid="stNumberInput"] input:focus,
    div[data-baseweb="select"]:focus-within > div {
        box-shadow: 0 0 0 3px rgba(255,255,255,0.5);
        transform: translateY(-1px);
    }
    div[data-testid="stNumberInput"] button {
        border-radius: 8px !important;
    }
    [data-baseweb="popover"] li[role="option"] {
        transition: background 0.15s ease;
        border-radius: 6px;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #E10600, #ff4136);
        color: white;
        font-weight: 700;
        letter-spacing: 0.03em;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.2rem;
        width: 100%;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.45);
        color: white;
    }

    .result-card {
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        background: rgba(0,0,0,0.55);
        border-left: 8px solid var(--team-color, #E10600);
        backdrop-filter: blur(6px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Script "puente": pinta cada opción de los desplegables (piloto/escudería)
# con el color real de su equipo, y añade un puntito de color delante.
components.html(
    f"""
    <script>
    const COLOR_MAP = {json.dumps(NAME_COLOR_MAP)};
    function colorizeOptions() {{
        try {{
            const doc = window.parent.document;
            const items = doc.querySelectorAll('li[role="option"]');
            items.forEach((li) => {{
                const label = li.textContent.trim();
                const color = COLOR_MAP[label];
                if (color && !li.dataset.f1Colored) {{
                    li.dataset.f1Colored = "1";
                    li.style.borderLeft = `5px solid ${{color}}`;
                    li.style.background = `linear-gradient(90deg, ${{color}}26, transparent 65%)`;
                    const dot = doc.createElement('span');
                    dot.style.display = 'inline-block';
                    dot.style.width = '9px';
                    dot.style.height = '9px';
                    dot.style.borderRadius = '50%';
                    dot.style.background = color;
                    dot.style.marginRight = '8px';
                    li.prepend(dot);
                    li.addEventListener('mouseenter', () => {{
                        li.style.background = `linear-gradient(90deg, ${{color}}55, transparent 85%)`;
                    }});
                    li.addEventListener('mouseleave', () => {{
                        li.style.background = `linear-gradient(90deg, ${{color}}26, transparent 65%)`;
                    }});
                }}
            }});
        }} catch (e) {{ /* entorno restringido: se ignora */ }}
    }}
    try {{
        const observer = new MutationObserver(colorizeOptions);
        observer.observe(window.parent.document.body, {{ childList: true, subtree: true }});
        colorizeOptions();
        setInterval(colorizeOptions, 400);
    }} catch (e) {{ /* entorno restringido: se ignora */ }}
    </script>
    """,
    height=0,
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
    with st.container(border=True, key="col1_card"):
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
    with st.container(border=True, key="col2_card"):
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
        <div class="result-card" style="--team-color:{team_color};">
            <div style="display:flex;align-items:center;gap:0.9rem;">
                <div style="font-size:2.6rem;">{medalla}</div>
                <div>
                    <div style="font-size:0.9rem;opacity:0.85;">
                        {DRIVER_INFO[driverId]["name"]} · {TEAM_NAMES[constructorId]}
                    </div>
                    <div style="font-size:1.8rem;font-weight:800;color:#ffffff;">
                        Posición final estimada: {prediccion_redondeada}ª
                    </div>
                </div>
            </div>
            <p style="margin-top:0.6rem;opacity:0.75;font-size:0.85rem;">
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
