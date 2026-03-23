import pandas as pd
import sqlite3
import os

DB_PATH = "motoriq.db"

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)

# ── Dataset 1: Car Dataset 1945-2020 ──────────────────────────────────────────
print("Loading Car Dataset 1945-2020...")
df1 = pd.read_csv("Car Dataset 1945-2020.csv", low_memory=False)
print(f"  Raw rows: {len(df1)}")

df1 = df1.rename(columns={
    "Make":                         "make",
    "Modle":                        "model",
    "Year_from":                    "year",
    "Body_type":                    "body",
    "number_of_seats":              "seats",
    "number_of_doors":              "doors",
    "engine_hp":                    "hp",
    "number_of_cylinders":          "engine",
    "capacity_cm3":                 "displacement",
    "maximum_torque_n_m":           "torque",
    "engine_type":                  "powertrain",
    "boost_type":                   "boost",
    "drive_wheels":                 "drivetrain",
    "transmission":                 "transmission",
    "number_of_gears":              "gears",
    "acceleration_0_100_km/h_s":    "zerotosixty",
    "max_speed_km_per_h":           "top_speed",
    "mixed_fuel_consumption_per_100_km_l": "mixed_fuel_l100km",
    "CO2_emissions_g/km":           "co2_g_per_km",
    "electric_range_km":            "electric_range",
    "battery_capacity_KW_per_h":    "battery_capacity",
    "curb_weight_kg":               "weight",
    "country_of_origin":            "country",
    "car_class":                    "class",
    "cylinder_layout":              "engine2",
    "engine_placement":             "engine_placement",
})

keep1 = [c for c in [
    "make", "model", "year", "body", "seats", "doors",
    "hp", "engine", "displacement", "torque",
    "powertrain", "boost", "drivetrain", "transmission", "gears",
    "zerotosixty", "top_speed", "mixed_fuel_l100km",
    "co2_g_per_km", "electric_range", "battery_capacity",
    "weight", "country", "class", "engine2", "engine_placement"
] if c in df1.columns]

df1 = df1[keep1].copy()
df1["hp"] = pd.to_numeric(df1["hp"], errors="coerce")
df1["year"] = pd.to_numeric(df1["year"], errors="coerce")
df1["year"] = df1["year"].astype("Int64")
df1["torque"] = pd.to_numeric(df1["torque"], errors="coerce")
df1["displacement"] = pd.to_numeric(df1["displacement"], errors="coerce")
df1["zerotosixty"] = pd.to_numeric(df1["zerotosixty"], errors="coerce")
df1["top_speed"] = pd.to_numeric(df1["top_speed"], errors="coerce")
df1 = df1.dropna(subset=["make", "model", "year"])
df1["make"] = df1["make"].str.strip().str.title()
df1["model"] = df1["model"].str.strip()
df1["source"] = "1945_2020"

print(f"  Clean rows: {len(df1)}")
df1.to_sql("vehicles_1945", conn, if_exists="replace", index=False, dtype={
    "make":             "TEXT",
    "model":            "TEXT",
    "year":             "INTEGER",
    "body":             "TEXT",
    "seats":            "INTEGER",
    "doors":            "INTEGER",
    "hp":               "INTEGER",
    "engine":           "TEXT",
    "displacement":     "REAL",
    "torque":           "INTEGER",
    "powertrain":       "TEXT",
    "fuel_type":        "TEXT",
    "boost":            "TEXT",
    "drivetrain":       "TEXT",
    "transmission":     "TEXT",
    "gears":            "INTEGER",
    "zerotosixty":      "REAL",
    "top_speed":        "INTEGER",
    "mixed_fuel_l100km":"REAL",
    "co2_g_per_km":     "REAL",
    "electric_range":   "INTEGER",
    "battery_capacity": "INTEGER",
    "weight":           "INTEGER",
    "country":          "TEXT",
    "class":            "TEXT",
    "engine2":          "TEXT",
    "engine_placement": "TEXT",
    "source":           "TEXT"
})
print("  Saved to vehicles_1945")

# ── Dataset 2: Car Features and MSRP ─────────────────────────────────────────
print("\nLoading Car Features and MSRP (data.csv)...")
df2 = pd.read_csv("data.csv", low_memory=False)
print(f"  Raw rows: {len(df2)}")

df2.columns = [c.strip() for c in df2.columns]
df2 = df2.rename(columns={
    "Make":               "make",
    "Model":              "model",
    "Year":               "year",
    "Engine Fuel Type":   "fuel_type",
    "Engine HP":          "hp",
    "Engine Cylinders":   "engine",
    "Transmission Type":  "transmission",
    "Driven_Wheels":      "drivetrain",
    "Number of Doors":    "doors",
    "Market Category":    "market",
    "Vehicle Size":       "size",
    "Vehicle Style":      "body",
    "highway MPG":        "highway_mpg",
    "city mpg":           "city_mpg",
    "Popularity":         "popularity",
    "MSRP":               "msrp",
})

keep2 = [c for c in [
    "make", "model", "year", "fuel_type", "hp",
    "engine", "transmission", "drivetrain",
    "doors", "market", "size", "body",
    "highway_mpg", "city_mpg", "popularity", "msrp"
] if c in df2.columns]

df2 = df2[keep2].copy()
df2["hp"] = pd.to_numeric(df2["hp"], errors="coerce")
df2["year"] = pd.to_numeric(df2["year"], errors="coerce")
df2["year"] = df2["year"].astype("Int64")
if "msrp" in df2.columns:
    df2["msrp"] = pd.to_numeric(df2["msrp"], errors="coerce")
df2 = df2.dropna(subset=["make", "model", "year"])
df2["make"] = df2["make"].str.strip().str.title()
df2["model"] = df2["model"].str.strip()
df2["source"] = "msrp"

print(f"  Clean rows: {len(df2)}")
df2.to_sql("vehicles_msrp", conn, if_exists="replace", index=False, dtype={
    "make":             "TEXT",
    "model":            "TEXT",
    "year":             "INTEGER",
    "fuel_type":        "TEXT",
    "hp":               "INTEGER",
    "engine":           "TEXT",
    "transmission":     "TEXT",
    "drivetrain":       "TEXT",
    "doors":            "INTEGER",
    "market":           "TEXT",
    "size":             "TEXT",
    "body":             "TEXT",
    "highway_mpg":      "REAL",
    "city_mpg":         "REAL",
    "popularity":       "REAL",
    "msrp":             "REAL",
    "source":           "TEXT"
})
print("  Saved to vehicles_msrp")

# ── Summary ───────────────────────────────────────────────────────────────────
r1 = conn.execute("SELECT COUNT(*) FROM vehicles_1945").fetchone()[0]
r2 = conn.execute("SELECT COUNT(*) FROM vehicles_msrp").fetchone()[0]
yr1 = conn.execute("SELECT MIN(year), MAX(year) FROM vehicles_1945").fetchone()
yr2 = conn.execute("SELECT MIN(year), MAX(year) FROM vehicles_msrp").fetchone()
m1 = conn.execute("SELECT COUNT(DISTINCT make) FROM vehicles_1945").fetchone()[0]
m2 = conn.execute("SELECT COUNT(DISTINCT make) FROM vehicles_msrp").fetchone()[0]

print(f"\nDatabase ready:")
print(f"  vehicles_1945 : {r1:,} records | {yr1[0]:.0f}-{yr1[1]:.0f} | {m1} makes")
print(f"  vehicles_msrp : {r2:,} records | {yr2[0]:.0f}-{yr2[1]:.0f} | {m2} makes")
print(f"\nRich columns available in vehicles_1945:")
print(f"  hp, torque, displacement, zerotosixty,")
print(f"  top_speed, boost, electric_range, weight")
print(f"\nRich columns available in vehicles_msrp:")
print(f"  hp, msrp, market, highway_mpg, city_mpg")

conn.close()
print("\nBuild complete - motoriq.db is ready")
