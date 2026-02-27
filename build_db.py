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
    "Body_type":                    "body_style",
    "number_of_seats":              "seats",
    "number_of_doors":              "doors",
    "engine_hp":                    "engine_hp",
    "number_of_cylinders":          "engine_cylinders",
    "capacity_cm3":                 "displacement_cc",
    "maximum_torque_n_m":           "torque_nm",
    "engine_type":                  "fuel_type",
    "boost_type":                   "boost_type",
    "drive_wheels":                 "drive_type",
    "transmission":                 "transmission",
    "number_of_gears":              "gears",
    "acceleration_0_100_km/h_s":    "accel_0_100_s",
    "max_speed_km_per_h":           "top_speed_kph",
    "mixed_fuel_consumption_per_100_km_l": "mixed_fuel_l100km",
    "CO2_emissions_g/km":           "co2_g_per_km",
    "electric_range_km":            "electric_range_km",
    "battery_capacity_KW_per_h":    "battery_kwh",
    "curb_weight_kg":               "weight_kg",
    "country_of_origin":            "country",
    "car_class":                    "car_class",
    "cylinder_layout":              "cylinder_layout",
    "engine_placement":             "engine_placement",
})

keep1 = [c for c in [
    "make", "model", "year", "body_style", "seats", "doors",
    "engine_hp", "engine_cylinders", "displacement_cc", "torque_nm",
    "fuel_type", "boost_type", "drive_type", "transmission", "gears",
    "accel_0_100_s", "top_speed_kph", "mixed_fuel_l100km",
    "co2_g_per_km", "electric_range_km", "battery_kwh",
    "weight_kg", "country", "car_class", "cylinder_layout", "engine_placement"
] if c in df1.columns]

df1 = df1[keep1].copy()
df1["engine_hp"] = pd.to_numeric(df1["engine_hp"], errors="coerce")
df1["year"] = pd.to_numeric(df1["year"], errors="coerce")
df1["torque_nm"] = pd.to_numeric(df1["torque_nm"], errors="coerce")
df1["displacement_cc"] = pd.to_numeric(df1["displacement_cc"], errors="coerce")
df1["accel_0_100_s"] = pd.to_numeric(df1["accel_0_100_s"], errors="coerce")
df1["top_speed_kph"] = pd.to_numeric(df1["top_speed_kph"], errors="coerce")
df1 = df1.dropna(subset=["make", "model", "year"])
df1["make"] = df1["make"].str.strip().str.title()
df1["model"] = df1["model"].str.strip()
df1["source"] = "1945_2020"

print(f"  Clean rows: {len(df1)}")
df1.to_sql("vehicles_1945", conn, if_exists="replace", index=False)
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
    "Engine HP":          "engine_hp",
    "Engine Cylinders":   "engine_cylinders",
    "Transmission Type":  "transmission",
    "Driven_Wheels":      "drive_type",
    "Number of Doors":    "doors",
    "Market Category":    "market_category",
    "Vehicle Size":       "vehicle_size",
    "Vehicle Style":      "body_style",
    "highway MPG":        "highway_mpg",
    "city mpg":           "city_mpg",
    "Popularity":         "popularity",
    "MSRP":               "msrp",
})

keep2 = [c for c in [
    "make", "model", "year", "fuel_type", "engine_hp",
    "engine_cylinders", "transmission", "drive_type",
    "doors", "market_category", "vehicle_size", "body_style",
    "highway_mpg", "city_mpg", "popularity", "msrp"
] if c in df2.columns]

df2 = df2[keep2].copy()
df2["engine_hp"] = pd.to_numeric(df2["engine_hp"], errors="coerce")
df2["year"] = pd.to_numeric(df2["year"], errors="coerce")
if "msrp" in df2.columns:
    df2["msrp"] = pd.to_numeric(df2["msrp"], errors="coerce")
df2 = df2.dropna(subset=["make", "model", "year"])
df2["make"] = df2["make"].str.strip().str.title()
df2["model"] = df2["model"].str.strip()
df2["source"] = "msrp"

print(f"  Clean rows: {len(df2)}")
df2.to_sql("vehicles_msrp", conn, if_exists="replace", index=False)
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
print(f"  engine_hp, torque_nm, displacement_cc, accel_0_100_s,")
print(f"  top_speed_kph, boost_type, electric_range_km, weight_kg")
print(f"\nRich columns available in vehicles_msrp:")
print(f"  engine_hp, msrp, market_category, highway_mpg, city_mpg")

conn.close()
print("\nBuild complete - motoriq.db is ready")
