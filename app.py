import os
import re
import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH = "motoriq.db"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

st.set_page_config(page_title="MotorIQ", page_icon="🏎️", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .sql-box {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
    }
    .stat-card {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Schema context for GPT ────────────────────────────────────────────────────
SCHEMA = """
You are a SQL expert for a SQLite automotive database called motoriq.db.
It has two tables:

TABLE: vehicles_1945
  make             TEXT    -- manufacturer (e.g. 'Ferrari', 'Bmw', 'Ford')
  model            TEXT    -- model name
  year             INTEGER -- production year (1904-2020)
  body_style       TEXT    -- e.g. 'Sedan', 'Coupe', 'SUV', 'Convertible'
  seats            REAL
  doors            REAL
  engine_hp        REAL    -- horsepower
  engine_cylinders REAL
  displacement_cc  REAL    -- engine displacement in cc
  torque_nm        REAL    -- torque in newton-meters
  fuel_type        TEXT    -- e.g. 'Gasoline', 'Diesel', 'Electric'
  boost_type       TEXT    -- e.g. 'Turbo', 'Supercharger', NULL if naturally aspirated
  drive_type       TEXT    -- e.g. 'Rear', 'Front', 'All'
  transmission     TEXT    -- e.g. 'Manual', 'Automatic'
  gears            REAL
  accel_0_100_s    REAL    -- 0-100 km/h in seconds (lower = faster)
  top_speed_kph    REAL    -- top speed in km/h
  mixed_fuel_l100km REAL   -- fuel consumption L/100km
  co2_g_per_km     REAL
  electric_range_km REAL   -- for EVs
  battery_kwh      REAL    -- for EVs
  weight_kg        REAL
  country          TEXT    -- country of origin
  car_class        TEXT
  cylinder_layout  TEXT    -- e.g. 'V', 'Inline', 'Boxer'
  engine_placement TEXT    -- e.g. 'Front', 'Mid', 'Rear'

TABLE: vehicles_msrp
  make             TEXT
  model            TEXT
  year             INTEGER -- 1990-2017
  fuel_type        TEXT
  engine_hp        REAL
  engine_cylinders REAL
  transmission     TEXT
  drive_type       TEXT
  doors            REAL
  market_category  TEXT    -- e.g. 'Luxury', 'Performance', 'High-Performance'
  vehicle_size     TEXT    -- 'Compact', 'Midsize', 'Large'
  body_style       TEXT
  highway_mpg      REAL
  city_mpg         REAL
  popularity       REAL
  msrp             REAL    -- manufacturer suggested retail price in USD

RULES:
- Always use SELECT statements only, never INSERT/UPDATE/DELETE
- Limit results to 50 rows unless the user asks for more
- For decade analysis use: CAST(year/10 AS INT)*10 AS decade
- For averages always ROUND to 2 decimal places
- Use LOWER(make) for case-insensitive make comparisons
- vehicles_1945 has depth (1904-2020) and performance specs
- vehicles_msrp has MSRP and market_category but only 1990-2017
- When the question involves MSRP or market category, use vehicles_msrp
- When the question involves torque, displacement, acceleration, top speed, use vehicles_1945
- Return only the raw SQL query with no explanation, no markdown, no backticks
"""

# ── Helpers ───────────────────────────────────────────────────────────────────
def generate_sql(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SCHEMA},
            {"role": "user", "content": question}
        ],
        temperature=0
    )
    sql = response.choices[0].message.content.strip()
    sql = re.sub(r"```sql|```", "", sql).strip()
    return sql

def run_query(sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df

def auto_chart(df: pd.DataFrame, question: str):
    if df.empty or len(df.columns) < 2:
        return None

    cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(include="object").columns.tolist()

    if not numeric_cols:
        return None

    q = question.lower()
    y_col = numeric_cols[0]
    x_col = text_cols[0] if text_cols else cols[0]

    # Time series
    if "year" in cols or "decade" in cols:
        x_col = "decade" if "decade" in cols else "year"
        fig = px.line(
            df, x=x_col, y=y_col,
            color=text_cols[0] if len(text_cols) > 1 else None,
            markers=True,
            template="plotly_dark",
            title=question.capitalize()
        )
        return fig

    # Scatter with two numeric axes
    if len(numeric_cols) >= 2 and ("vs" in q or "compare" in q or "correlation" in q):
        fig = px.scatter(
            df, x=numeric_cols[0], y=numeric_cols[1],
            color=text_cols[0] if text_cols else None,
            hover_data=cols,
            template="plotly_dark",
            title=question.capitalize()
        )
        return fig

    # Bar for most other cases
    fig = px.bar(
        df.head(30), x=x_col, y=y_col,
        color=text_cols[1] if len(text_cols) > 1 else None,
        template="plotly_dark",
        title=question.capitalize()
    )
    fig.update_layout(xaxis_tickangle=-35)
    return fig

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏎️ MotorIQ")
st.markdown("Natural language questions over **82,500 vehicle records** spanning 1904–2020. Powered by GPT-4o-mini → SQL → Plotly.")

st.divider()

# Sample questions
st.markdown("**Try asking:**")
examples = [
    "How has average horsepower changed by decade for Ferrari?",
    "Which turbocharged cars have the fastest 0-100 times?",
    "Compare average horsepower of German vs Italian vs American makes since 1980",
    "What is the average MSRP of rear-wheel drive performance cars by make?",
    "Show the top 20 highest horsepower cars of all time",
    "How does average torque compare between V8 and V12 engines by decade?",
]

cols = st.columns(3)
for i, ex in enumerate(examples):
    with cols[i % 3]:
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["query"] = ex

st.divider()

query = st.text_input(
    "Ask anything about cars:",
    value=st.session_state.get("query", ""),
    placeholder="e.g. Which naturally aspirated V12 cars have the highest top speed?",
    key="query_input"
)

if query:
    with st.spinner("Generating SQL and querying database..."):
        try:
            sql = generate_sql(query)
            df = run_query(sql)
        except Exception as e:
            st.error(f"Query failed: {e}")
            st.stop()

    if df.empty:
        st.warning("Query returned no results. Try rephrasing.")
    else:
        # Results row
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{len(df):,} rows returned**")
        with col2:
            st.download_button(
                "⬇️ Download CSV",
                df.to_csv(index=False),
                file_name="motoriq_results.csv",
                mime="text/csv"
            )

        # Chart
        fig = auto_chart(df, query)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Table
        st.dataframe(df, use_container_width=True)

        # SQL transparency
        with st.expander("🔍 View Generated SQL"):
            st.markdown(f'<div class="sql-box">{sql}</div>', unsafe_allow_html=True)
