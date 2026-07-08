import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Predictor de Puesto en F1", page_icon="🏁", layout="wide")

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

SHORT_NAME = {did: info["name"].split()[-1] for did, info in DRIVER_INFO.items()}

# Pilotos agrupados y ordenados por escudería, para que el "muro" de botones
# quede agrupado por color en vez de disperso alfabéticamente.
DRIVER_ORDER = sorted(DRIVER_INFO, key=lambda d: (TEAM_NAMES[DRIVER_INFO[d]["team"]], DRIVER_INFO[d]["name"]))
TEAM_ORDER = sorted(TEAM_NAMES, key=lambda t: TEAM_NAMES[t])


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
# Estilos — tema F1: fondo rojo/negro con relieve, tarjetas de cristal,
# inputs en blanco y botones de piloto/escudería coloreados por equipo.
# ---------------------------------------------------------------------------

driver_btn_css = "\n".join(
    f'.st-key-driver_{did} button {{ background:{TEAM_COLORS[info["team"]]} !important; '
    f'border:2px solid rgba(255,255,255,0.25) !important; color:#fff !important; }}'
    for did, info in DRIVER_INFO.items()
)
team_btn_css = "\n".join(
    f'.st-key-team_{tid} button {{ background:{color} !important; '
    f'border:2px solid rgba(255,255,255,0.25) !important; color:#fff !important; }}'
    for tid, color in TEAM_COLORS.items()
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700;900&display=swap');

    html, body, [class*="css"] {{ font-family: 'Titillium Web', sans-serif; }}

    [data-testid="stAppViewContainer"] {{
        background:
            radial-gradient(circle at 12% -10%, rgba(225,6,0,0.38), transparent 40%),
            radial-gradient(circle at 105% 10%, rgba(225,6,0,0.22), transparent 42%),
            radial-gradient(circle at 50% 40%, rgba(255,255,255,0.05), transparent 55%),
            linear-gradient(180deg, #17070a 0%, #0c0508 45%, #050304 100%);
        background-attachment: scroll;
    }}
    [data-testid="stMain"] {{
        background: transparent;
    }}
    [data-testid="stHeader"] {{ background: rgba(10,4,5,0.95); }}
    [data-testid="stAppViewContainer"] * {{ color: #f5f5f5; }}

    .f1-hero {{
        background: rgba(0,0,0,0.45);
        border-radius: 18px;
        padding: 1.7rem 2rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        backdrop-filter: blur(8px);
    }}
    .f1-hero h1 {{
        color: #ffffff;
        font-weight: 900;
        font-size: 2.3rem;
        letter-spacing: 0.03em;
        margin: 0 0 0.4rem 0;
        text-shadow: 0 2px 12px rgba(0,0,0,0.55);
    }}
    .f1-hero p {{ color: #f0f0f0; font-size: 1rem; margin: 0; }}

    .checker-strip {{
        height: 10px;
        border-radius: 4px;
        margin: 0.8rem 0 1.2rem 0;
        background-image: repeating-linear-gradient(90deg, #0c0c0c 0 12px, #ffffff 12px 24px);
        box-shadow: 0 2px 6px rgba(0,0,0,0.4);
        opacity: 0.95;
    }}

    .section-title {{ font-weight: 700; font-size: 1.2rem; margin: 0.2rem 0 0.1rem 0; color: #ffffff; }}
    .section-desc {{ font-size: 0.86rem; opacity: 0.85; margin-bottom: 0.7rem; }}
    .subgroup-title {{ font-weight: 700; font-size: 0.95rem; margin: 0.6rem 0 0.4rem 0; opacity: 0.9; }}

    .glass-card {{
        background: rgba(0,0,0,0.4) !important;
        border-radius: 18px !important;
        border: 1px solid rgba(255,255,255,0.16) !important;
        padding: 1rem 1.2rem 1.3rem 1.2rem !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        backdrop-filter: blur(6px);
        margin-bottom: 1rem;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div .glass-anchor) {{
        background: rgba(0,0,0,0.4) !important;
        border-radius: 18px !important;
        border: 1px solid rgba(255,255,255,0.16) !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        backdrop-filter: blur(6px);
    }}

    /* Inputs y selects: cajas blancas, redondeadas e interactivas, con un
       borde marcado para que se note claramente dónde se puede escribir. */
    div[data-testid="stNumberInput"] input,
    div[data-baseweb="select"] > div {{
        background: #ffffff !important;
        border-radius: 10px !important;
        border: 2px solid #ffd7d2 !important;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.15);
        transition: box-shadow 0.15s ease, transform 0.15s ease, border-color 0.15s ease;
    }}
    div[data-testid="stNumberInput"] input {{ color: #111111 !important; }}
    div[data-baseweb="select"] * {{ color: #111111 !important; }}
    div[data-testid="stNumberInput"] input:focus,
    div[data-baseweb="select"]:focus-within > div {{
        border-color: #E10600 !important;
        box-shadow: 0 0 0 3px rgba(225,6,0,0.35);
        transform: translateY(-1px);
    }}
    div[data-testid="stNumberInput"] button {{
        border-radius: 8px !important;
        background: #ffe8e5 !important;
        border: 1px solid #ffd7d2 !important;
    }}
    /* los iconos +/- y la flecha del select heredaban el blanco global y
       quedaban invisibles sobre fondo blanco: se fuerzan a oscuro */
    div[data-testid="stNumberInput"] svg,
    div[data-baseweb="select"] svg {{
        fill: #111111 !important;
        color: #111111 !important;
    }}
    label[data-testid="stWidgetLabel"] p {{
        font-weight: 600 !important;
    }}

    /* Botones de piloto / escudería: coloreados por equipo, mismo tamaño
       para todos independientemente de lo largo del nombre. */
    div.stButton button {{
        width: 100%;
        min-height: 3rem;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.5rem 0.4rem !important;
        white-space: normal !important;
        line-height: 1.15 !important;
        transition: transform 0.12s ease, box-shadow 0.12s ease, filter 0.12s ease !important;
        box-shadow: 0 3px 10px rgba(0,0,0,0.35);
    }}
    div.stButton button:hover {{
        transform: translateY(-3px) scale(1.03);
        filter: brightness(1.2);
        box-shadow: 0 8px 18px rgba(0,0,0,0.5);
    }}
    {driver_btn_css}
    {team_btn_css}

    .st-key-predict_action {{
        margin: 0.6rem 0 1.2rem 0;
    }}
    .st-key-predict_action button {{
        background: linear-gradient(90deg, #ff4136, #E10600, #ff4136) !important;
        background-size: 200% auto !important;
        color: white !important;
        font-weight: 800 !important;
        letter-spacing: 0.04em;
        border: none !important;
        border-radius: 14px !important;
        padding: 1.1rem 1.2rem !important;
        font-size: 1.3rem !important;
        min-height: 4rem;
        box-shadow: 0 8px 24px rgba(225,6,0,0.55);
        animation: f1-pulse 2.2s ease-in-out infinite;
    }}
    .st-key-predict_action button:hover {{
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 12px 30px rgba(225,6,0,0.75);
        animation-play-state: paused;
    }}
    @keyframes f1-pulse {{
        0% {{ box-shadow: 0 8px 24px rgba(225,6,0,0.45); background-position: 0% center; }}
        50% {{ box-shadow: 0 8px 30px rgba(225,6,0,0.85); background-position: 100% center; }}
        100% {{ box-shadow: 0 8px 24px rgba(225,6,0,0.45); background-position: 0% center; }}
    }}

    .result-card {{
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        background: rgba(0,0,0,0.55);
        border-left: 8px solid var(--team-color, #E10600);
        box-shadow: 0 10px 26px rgba(0,0,0,0.4);
        backdrop-filter: blur(6px);
    }}
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

# ---------------------------------------------------------------------------
# Selección de piloto y escudería mediante "muros" de botones de color.
# ---------------------------------------------------------------------------

if "selected_driver" not in st.session_state:
    st.session_state["selected_driver"] = DRIVER_ORDER[0]
if "selected_team" not in st.session_state:
    st.session_state["selected_team"] = DRIVER_INFO[DRIVER_ORDER[0]]["team"]
if "_prev_driver" not in st.session_state:
    st.session_state["_prev_driver"] = DRIVER_ORDER[0]

st.markdown('<div class="section-title">🏎️ Elige piloto y escudería</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">Cada botón lleva el color real de su equipo — pulsa un piloto y su '
    'escudería se selecciona sola (puedes cambiarla a mano para escenarios hipotéticos).</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="subgroup-title">Piloto</div>', unsafe_allow_html=True)
clicked_driver = None
driver_cols = st.columns(7)
for i, did in enumerate(DRIVER_ORDER):
    with driver_cols[i % 7]:
        with st.container(key=f"driver_{did}"):
            if st.button(SHORT_NAME[did], key=f"btn_driver_{did}", use_container_width=True):
                clicked_driver = did

if clicked_driver:
    st.session_state["selected_driver"] = clicked_driver

if st.session_state["_prev_driver"] != st.session_state["selected_driver"]:
    st.session_state["selected_team"] = DRIVER_INFO[st.session_state["selected_driver"]]["team"]
    st.session_state["_prev_driver"] = st.session_state["selected_driver"]

st.markdown('<div class="subgroup-title">Escudería</div>', unsafe_allow_html=True)
clicked_team = None
team_cols = st.columns(6)
for i, tid in enumerate(TEAM_ORDER):
    with team_cols[i % 6]:
        with st.container(key=f"team_{tid}"):
            if st.button(TEAM_NAMES[tid], key=f"btn_team_{tid}", use_container_width=True):
                clicked_team = tid

if clicked_team:
    st.session_state["selected_team"] = clicked_team

driverId = st.session_state["selected_driver"]
constructorId = st.session_state["selected_team"]
constructorName = TEAM_NAMES[constructorId]
constructor_nationality = TEAM_NATIONALITY[constructorId]
driver_nationality = DRIVER_INFO[driverId]["nationality"]
number = DRIVER_INFO[driverId]["number"]
team_color = TEAM_COLORS[constructorId]

st.markdown(
    f'{pill(TEAM_COLORS[DRIVER_INFO[driverId]["team"]], "Piloto: " + DRIVER_INFO[driverId]["name"])} '
    f'&nbsp; {pill(team_color, "Escudería: " + constructorName)} '
    f'&nbsp; {DRIVER_INFO[driverId]["nationality"]} · Nº {DRIVER_INFO[driverId]["number"]} · '
    f'{constructor_nationality}',
    unsafe_allow_html=True,
)

# resalte de la selección actual (anillo blanco + brillo del color de equipo)
st.markdown(
    f"""
    <style>
    .st-key-driver_{driverId} button {{
        box-shadow: 0 0 0 3px #ffffff, 0 0 16px {TEAM_COLORS[DRIVER_INFO[driverId]["team"]]} !important;
        transform: scale(1.05);
    }}
    .st-key-team_{constructorId} button {{
        box-shadow: 0 0 0 3px #ffffff, 0 0 16px {team_color} !important;
        transform: scale(1.05);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="checker-strip"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Resto de variables, en cuadrícula compacta (2 filas) en vez de una lista larga.
# ---------------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown('<span class="glass-anchor"></span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">📋 Antes de la carrera</div>'
            '<div class="section-desc">Datos conocidos antes de la salida: circuito, parrilla '
            'y clasificación.</div>',
            unsafe_allow_html=True,
        )

        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            season = st.number_input("Temporada", min_value=2022, max_value=2026, value=2024)
        with r1c2:
            ronda = st.number_input("Ronda", min_value=1, max_value=24, value=1)
        with r1c3:
            grid = st.number_input("Grid", min_value=1, max_value=20, value=1)
        with r1c4:
            quali_position = st.number_input("Quali", min_value=1, max_value=20, value=1)

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            q1 = st.number_input("Q1 (seg)", min_value=0.0, value=90.0)
        with r2c2:
            q2 = st.number_input("Q2 (seg)", min_value=0.0, value=89.0)
        with r2c3:
            q3 = st.number_input("Q3 (seg)", min_value=0.0, value=88.0)

        circuit_options = sorted(CIRCUIT_NAMES, key=lambda c: CIRCUIT_NAMES[c])
        circuitId = st.selectbox(
            "Circuito", options=circuit_options, format_func=lambda c: CIRCUIT_NAMES[c]
        )
        circuitName = CIRCUIT_NAMES[circuitId]
        country = circuito_a_pais[circuitName]
        st.caption(f"📍 País: **{country}**")

with col2:
    with st.container(border=True):
        st.markdown('<span class="glass-anchor"></span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">🔴 En directo</div>'
            '<div class="section-desc">Datos que cambian vuelta a vuelta mientras la carrera '
            'está en marcha.</div>',
            unsafe_allow_html=True,
        )
        r3c1, r3c2 = st.columns(2)
        with r3c1:
            lap = st.number_input("Vuelta actual", min_value=1, max_value=80, value=1)
        with r3c2:
            position_en_vuelta = st.number_input("Posición en vuelta", min_value=1, max_value=20, value=1)

        r4c1, r4c2 = st.columns(2)
        with r4c1:
            paradas_hasta_ahora = st.number_input("Paradas hechas", min_value=0, max_value=5, value=0)
        with r4c2:
            ultima_pit_duration = st.number_input("Última parada (seg)", min_value=0.0, value=0.0)

st.markdown('<div class="checker-strip"></div>', unsafe_allow_html=True)

with st.container(key="predict_action"):
    predict_clicked = st.button("🏎️ Predecir posición final", type="primary")

if predict_clicked:

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
