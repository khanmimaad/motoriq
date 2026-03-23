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
CRITICAL INSTRUCTION: You must use ONLY the exact column names defined below. Do NOT invent, rename, or substitute column names. The following column names are FORBIDDEN and do not exist in this database: engine_hp, body_style, car_class, displacement_cc, torque_nm, electric_range_km, battery_kwh, weight_kg, engine_cylinders, fuel_economy, mpg. Using any forbidden column name will cause the query to fail.

You are a SQL expert for a SQLite automotive database called motoriq.db.
It has two tables:

TABLE: vehicles_1945
  make             TEXT    -- manufacturer (e.g. 'Ferrari', 'BMW', 'Ford')
  model            TEXT    -- model name
  year             INTEGER -- production year (1904-2020)
  body             TEXT    -- body style e.g. 'Sedan', 'Coupe', 'SUV', 'Convertible'
  seats            INTEGER
  doors            INTEGER
  hp               INTEGER -- horsepower
  engine           TEXT    -- number of cylinders e.g. '4', '6', '8', '12'
  displacement     REAL    -- engine displacement in cc
  torque           INTEGER -- torque in lb-ft
  powertrain       TEXT    -- e.g. 'Gasoline', 'Diesel', 'Electric', 'Hybrid'
  boost            TEXT    -- e.g. 'Turbo', 'Supercharger', NULL if naturally aspirated
  drivetrain       TEXT    -- values vary: 'Rear wheel drive', 'RWD', 'Front wheel drive', 'FWD', 'All wheel drive (AWD)', 'AWD', 'Four wheel drive (4WD)' -- always use LIKE for filtering e.g. drivetrain LIKE '%Rear%' OR drivetrain LIKE '%RWD%'
  transmission     TEXT    -- e.g. 'Manual', 'Automatic'
  gears            INTEGER
  zerotosixty      REAL    -- 0-60 mph in seconds (lower = faster)
  top_speed        INTEGER -- top speed in kph
  mixed_fuel_l100km REAL   -- fuel consumption L/100km
  co2_g_per_km     REAL
  electric_range   INTEGER -- EV range in km
  battery_capacity INTEGER -- battery size in kWh for EVs
  weight           INTEGER -- curb weight in kg
  country          TEXT    -- country of origin e.g. 'Germany', 'Italy', 'Japan', 'USA'
  class            TEXT    -- vehicle class
  engine2          TEXT    -- cylinder layout e.g. 'V', 'Inline', 'Boxer'
  engine_placement TEXT    -- e.g. 'Front', 'Mid', 'Rear'
  source           TEXT    -- always '1945_2020' for this table
  codename         TEXT    -- internal model codename e.g. 'E46', 'MX-5', 'G80', NULL if unknown

TABLE: vehicles_msrp
  make             TEXT
  model            TEXT
  year             INTEGER -- 1990-2017
  fuel_type        TEXT    -- e.g. 'premium unleaded (required)', 'regular unleaded'
  hp               INTEGER -- horsepower
  engine           TEXT    -- number of cylinders
  transmission     TEXT    -- e.g. 'MANUAL', 'AUTOMATIC'
  drivetrain       TEXT    -- e.g. 'rear wheel drive', 'all wheel drive', 'front wheel drive'
  doors            INTEGER
  market           TEXT    -- comma-delimited tags e.g. 'Luxury,Performance', 'Exotic,High-Performance' -- always use LIKE for filtering e.g. market LIKE '%Performance%'
  size             TEXT    -- 'Compact', 'Midsize', 'Large'
  body             TEXT    -- body style
  highway_mpg      REAL
  city_mpg         REAL
  popularity       REAL    -- search popularity score from Edmunds
  msrp             REAL    -- manufacturer suggested retail price in USD
  source           TEXT    -- always 'msrp' for this table

RULES:
- Always use SELECT statements only, never INSERT, UPDATE, or DELETE
- Limit results to 50 rows unless the user asks for more
- For decade analysis use: CAST(year/10 AS INT)*10 AS decade
- For averages always ROUND to 2 decimal places
- makes in vehicles_1945 are properly title-cased (e.g. 'BMW', 'Ferrari', 'Mercedes-Benz') -- use exact equality: make = 'BMW', NOT LOWER(make) = 'bmw'
- makes in vehicles_msrp may vary in casing -- use LOWER(make) only when querying vehicles_msrp
- NEVER use fuel_type when querying vehicles_1945 -- that column does not exist in vehicles_1945. Use powertrain instead (e.g. powertrain = 'Electric', powertrain = 'Hybrid', powertrain = 'Gasoline')
- NEVER use powertrain when querying vehicles_msrp -- that column does not exist in vehicles_msrp. Use fuel_type instead
- vehicles_1945 has depth (1904-2020) and performance specs: hp, torque, displacement, zerotosixty, top_speed, boost, weight, codename
- vehicles_msrp has MSRP and market category but only covers 1990-2017
- When the question involves MSRP, market category, or popularity use vehicles_msrp
- When the question involves torque, displacement, zerotosixty, top_speed, boost, weight, electric_range, codename use vehicles_1945
- For general hp or year range questions prefer vehicles_1945 as it has more records
- Return only the raw SQL query with no explanation, no markdown, no backticks
"""

# ── Helpers ───────────────────────────────────────────────────────────────────
def generate_sql(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
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
        color_col = next((c for c in text_cols if c != x_col), None)
        df[x_col] = df[x_col].astype(str)
        fig = px.line(
            df, x=x_col, y=y_col,
            color=color_col,
            markers=True,
            template="plotly_dark",
            title=question.capitalize()
        )
        return fig

    # Scatter
    if len(numeric_cols) >= 2 and ("vs" in q or "compare" in q or "correlation" in q):
        fig = px.scatter(
            df, x=numeric_cols[0], y=numeric_cols[1],
            color=text_cols[0] if text_cols else None,
            hover_data=cols,
            template="plotly_dark",
            title=question.capitalize()
        )
        return fig

    # Bar
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
st.markdown("Natural language questions over **70,617 vehicle records** spanning 1904–2029. Powered by GPT-4o → SQL → Plotly.")

st.divider()

# Sample questions
st.markdown("**Try asking:**")
examples = [
    "How has average horsepower changed by decade for Ferrari?",
    "Which turbocharged cars have the fastest 0-60 times?",
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
        if "decade" in df.columns or "year" in df.columns:
            x_col = "decade" if "decade" in df.columns else "year"
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            text_cols = df.select_dtypes(include="object").columns.tolist()
            y_col = next((c for c in numeric_cols if c != x_col), None)
            color_col = next((c for c in df.columns if c not in numeric_cols and c != x_col), None)
            fig = px.line(
                df, x=x_col, y=y_col,
                color=color_col,
                markers=True,
                template="plotly_dark",
                title=query.capitalize()
            )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Table
        st.dataframe(df, use_container_width=True)

        # SQL transparency
        with st.expander("🔍 View Generated SQL"):
            st.markdown(f'<div class="sql-box">{sql}</div>', unsafe_allow_html=True)
