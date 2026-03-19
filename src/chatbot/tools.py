import psycopg2
import pandas as pd
import json


# ----------------------------------------
# DATABASE FUNCTION
# ----------------------------------------

def get_property_data():

    conn = psycopg2.connect(
        host="localhost",
        database="real_estate_db",
        user="postgres",
        password="!Texas123"
    )

    query = """
    SELECT
        p.address,
        p.metro_area,
        p.property_type,
        f.revenue,
        f.net_income,
        f.expenses
    FROM properties p
    JOIN financials f
    ON p.property_id = f.property_id
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df


# ----------------------------------------
# SEC REPORTS
# ----------------------------------------

def load_sec_reports():
    with open("data/raw/sec_reports.json") as f:
        data = json.load(f)
    return pd.DataFrame(data)


# ----------------------------------------
# PRESS RELEASES
# ----------------------------------------

def load_press_releases():
    with open("data/raw/press_releases.json") as f:
        data = json.load(f)
    return pd.DataFrame(data)


# ----------------------------------------
# UNIFIED TOOL: PROPERTY + FINANCIALS
# ----------------------------------------

def query_properties_with_financials(metro_area=None, property_type=None):

    df = get_property_data()

    if metro_area:
        df = df[df["metro_area"].str.contains(metro_area, case=False)]

    if property_type:
        df = df[df["property_type"].str.contains(property_type, case=False)]

    # ALWAYS return text for LLM
    return {
        "content": {
            "parts": [
                {"text": df.to_json(orient="records")}
            ]
        }
    }


# ----------------------------------------
# SEC TOOL
# ----------------------------------------

def get_sec_financials():

    df = load_sec_reports()

    return {
        "content": {
            "parts": [
                {"text": df.to_json(orient="records")}
            ]
        }
    }


# ----------------------------------------
# PRESS RELEASE TOOL
# ----------------------------------------

def search_press_releases(keyword):

    df = load_press_releases()

    result = df[df["title"].str.contains(keyword, case=False)]

    return {
        "content": {
            "parts": [
                {"text": result.to_json(orient="records")}
            ]
        }
    }
