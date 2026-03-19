import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import json
import os
import joblib
import vertexai
# Modular structure (prepared for scaling)
#import src.tools
#import src.prompts

from vertexai.generative_models import (
    GenerativeModel,
    Tool,
    FunctionDeclaration,
    ToolConfig
)

# ----------------------------------------
# VERTEX AI INIT
# ----------------------------------------

vertexai.init(
    project="real-estate-490304",
    location="global"
)

# ----------------------------------------
# PATH SETUP (FINAL FIX)
# ----------------------------------------

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# 👉 LOCAL MODELS PATH (FIXED)
MODEL_PATH = os.path.join(ROOT_DIR, "models", "local")

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
# DATA LOADERS
# ----------------------------------------
# ----------------------------------------
# DATA LOADERS (FINAL FIX)
# ----------------------------------------

def load_sec_reports():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ROOT_DIR = os.path.dirname(BASE_DIR)
        DATA_PATH = os.path.join(ROOT_DIR, "data", "raw")

        with open(os.path.join(DATA_PATH, "sec_reports.json")) as f:
            data = json.load(f)

        return pd.DataFrame(data)

    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


def load_press_releases():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ROOT_DIR = os.path.dirname(BASE_DIR)
        DATA_PATH = os.path.join(ROOT_DIR, "data", "raw")

        with open(os.path.join(DATA_PATH, "press_releases.json")) as f:
            data = json.load(f)

        return pd.DataFrame(data)

    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

# ----------------------------------------
# TOOL FUNCTIONS (LLM TOOLS)
# ----------------------------------------

def chicago_industrial_revenue_tool():
    try:
        df = get_property_data()

        df = df[
            (df["metro_area"] == "Chicago") &
            (df["property_type"] == "Industrial")
        ]

        return df[["address", "revenue", "net_income"]].to_dict(orient="records")

    except Exception as e:
        return [{"error": f"Database error: {str(e)}"}]


def press_releases_tool():
    try:
        df = load_press_releases()

        if df is None or df.empty:
            return [{"message": "No press release data available"}]

        return df.to_dict(orient="records")

    except Exception as e:
        return [{"error": f"Press release error: {str(e)}"}]


def sec_reports_tool():
    try:
        df = load_sec_reports()

        if df is None or df.empty:
            return [{"message": "No SEC report data available"}]

        return df.to_dict(orient="records")

    except Exception as e:
        return [{"error": f"SEC report error: {str(e)}"}]

# ----------------------------------------
# TOOL DECLARATION (TEK TOOL İÇİNDE 3 FONKSİYON)
# ----------------------------------------

combined_tool = Tool(
    function_declarations=[
        FunctionDeclaration(
            name="chicago_industrial_revenue_tool",
            description="Return industrial properties in Chicago with revenue and net income",
            parameters={"type": "object", "properties": {}}
        ),
        FunctionDeclaration(
            name="press_releases_tool",
            description="Return company press releases",
            parameters={"type": "object", "properties": {}}
        ),
        FunctionDeclaration(
            name="sec_reports_tool",
            description="Return SEC financial reports",
            parameters={"type": "object", "properties": {}}
        )
    ]
)

# ----------------------------------------
# GEMINI AGENT MODEL
# ----------------------------------------

agent_model = GenerativeModel(
    "gemini-2.5-flash",
    tools=[combined_tool],
    generation_config={
        "max_output_tokens": 2048,
        "temperature": 0.2
    }
)

# ----------------------------------------
# LOAD ML MODELS (REGRESSION & CLASSIFICATION)
# ----------------------------------------

regression_model = joblib.load(
    os.path.join(MODEL_PATH, "regression", "random_forest_model.pkl")
)

classification_model = joblib.load(
    os.path.join(MODEL_PATH, "classification", "logistic_model.pkl")
)

selected_features = joblib.load(
    os.path.join(MODEL_PATH, "classification", "features.pkl")
)

# 🔥 CLEAN (0 gibi garbage varsa kaldır)
selected_features = [f for f in selected_features if isinstance(f, str)]

# ----------------------------------------
# STREAMLIT UI
# ----------------------------------------

st.title("🏡 Real Estate Financial Assistant")

tab1, tab2, tab3 = st.tabs([
    "House Price Prediction",
    "Customer Subscription",
    "AI Financial Assistant"
])

with tab1:

    st.header("House Price Prediction")

    col_btn1, col_btn2 = st.columns([1, 1])

    with col_btn1:
        if st.button("Load Example Data"):
            st.session_state["medinc"] = 8.32
            st.session_state["house_age"] = 41.0
            st.session_state["avg_rooms"] = 6.98
            st.session_state["avg_bedrooms"] = 1.02
            st.session_state["population"] = 322.0
            st.session_state["avg_occupancy"] = 2.55
            st.session_state["latitude"] = 37.88
            st.session_state["longitude"] = -122.23
            st.rerun()

    with col_btn2:
        if st.button("Clear Inputs"):
            for key in ["medinc", "house_age", "avg_rooms", "avg_bedrooms",
                        "population", "avg_occupancy", "latitude", "longitude"]:
                st.session_state[key] = 0.0
            st.rerun()

    col1, col2 = st.columns(2)

    with col1:
        medinc = st.number_input("Median Income", key="medinc")
        house_age = st.number_input("House Age", key="house_age")
        avg_rooms = st.number_input("Average Rooms", key="avg_rooms")
        avg_bedrooms = st.number_input("Average Bedrooms", key="avg_bedrooms")

    with col2:
        population = st.number_input("Population", key="population")
        avg_occupancy = st.number_input("Average Occupancy", key="avg_occupancy")
        latitude = st.number_input("Latitude", key="latitude")
        longitude = st.number_input("Longitude", key="longitude")

    if st.button("Predict Price"):
        features = np.array([
            medinc, house_age, avg_rooms, avg_bedrooms,
            population, avg_occupancy, latitude, longitude
        ]).reshape(1, -1)

        prediction = regression_model.predict(features)[0]
        price = prediction * 100000
        st.success(f"Predicted House Price: ${price:,.0f}")

# ----------------------------------------
# CLASSIFICATION TAB
# ----------------------------------------

with tab2:

    st.header("Customer Subscription")

    col_btn1, col_btn2 = st.columns([1, 1])

    with col_btn1:
        if st.button("Load Example Data", key="load_example_tab2"):
            example_values = {
                "campaign": 1.0,
                "pdays": 999.0,
                "previous": 0.0,
                "job_blue-collar": 0.0,
                "job_retired": 0.0,
                "job_student": 1.0,
                "marital_single": 1.0,
                "education_tertiary": 1.0,
                "housing_yes": 0.0,
                "loan_yes": 0.0,
                "contact_unknown": 0.0,
                "month_dec": 0.0,
                "month_mar": 1.0,
                "month_may": 0.0,
                "month_oct": 0.0,
                "month_sep": 0.0,
                "poutcome_success": 1.0,
                "poutcome_unknown": 0.0,
            }
            for key, val in example_values.items():
                st.session_state[f"feat_{key}"] = val
            st.rerun()

    with col_btn2:
        if st.button("Clear Inputs", key="clear_tab2"):
            for feature in selected_features:
                st.session_state[f"feat_{feature}"] = 0.0
            st.rerun()

    cols = st.columns(3)
    input_features = []

    for i, feature in enumerate(selected_features):
        with cols[i % 3]:
            val = st.number_input(feature, key=f"feat_{feature}")
        input_features.append(val)


    if st.button("Predict Subscription", key="predict_tab2"):
        features = np.array(input_features).reshape(1, -1)
        prediction = int(classification_model.predict(features)[0])
        probability = float(classification_model.predict_proba(features)[0][1])

        st.markdown("### Result")

        if prediction == 1:
            st.success(f"✅ Customer likely to subscribe ({probability:.2%})")
        else:
            st.error(f"❌ Customer unlikely to subscribe ({probability:.2%})")

        st.markdown("**Subscription Probability**")
        st.progress(probability)


# ----------------------------------------
# HELPER (FORMAT OUTPUT)
# ----------------------------------------

def format_property_response(data):
    response = "Here are the industrial properties in Chicago:\n\n"

    for item in data:
        address = item.get("address", "N/A")
        revenue = f"${item.get('revenue', 0):,}"
        net_income = f"${item.get('net_income', 0):,}"

        response += f"""• Address: {address}
  - Revenue: {revenue}
  - Net Income: {net_income}

"""

    return response


# ----------------------------------------
# AI CHATBOT TAB (FINAL VERSION)
# ----------------------------------------

with tab3:

    st.header("AI Financial Assistant")

    if st.button("Clear Chat"):
        keys_to_keep = ["chat_input"]
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        st.session_state["chat_input"] = ""
        st.rerun()

    question = st.text_input(
        "Ask about properties, financials, SEC reports or press releases",
        key="chat_input"
    )

    if st.button("Ask AI"):

        # 🔥 SPINNER EKLENDİ
        with st.spinner("Analyzing data..."):

            prompt = f"""
You are a financial real estate assistant.

You have tools to retrieve:
- Chicago industrial property revenue
- SEC financial reports
- Company press releases

IMPORTANT RULES:
- Always use tools when relevant
- If the question is about acquisitions or announcements → use press_releases_tool
- If the question is about revenue or net income → use sec_reports_tool
- If the question is about Chicago industrial properties → use chicago_industrial_revenue_tool
- Assume all data refers to a single company
- DO NOT ask for clarification
- ALWAYS return an answer

User question:
{question}
"""

            # 1️⃣ First LLM call
            response = agent_model.generate_content(prompt)

            part = response.candidates[0].content.parts[0]
            fn_call = getattr(part, "function_call", None)

            if fn_call is not None:

                fn = fn_call.name

                # 2️⃣ Run tool
                if fn == "chicago_industrial_revenue_tool":
                    tool_result = chicago_industrial_revenue_tool()

                elif fn == "press_releases_tool":
                    tool_result = press_releases_tool()

                elif fn == "sec_reports_tool":
                    tool_result = sec_reports_tool()

                else:
                    tool_result = {"error": "Unknown tool called"}

                wrapped_result = {"data": tool_result}

                # 🔥 FORMAT PROPERTY OUTPUT
                if fn == "chicago_industrial_revenue_tool":
                    st.subheader("AI Response")
                    st.write(format_property_response(tool_result))
                    

                # 3️⃣ Send tool result back to Gemini
                followup = agent_model.generate_content(
                    contents=[
                        {"role": "user", "parts": [{"text": prompt}]},
                        {
                            "role": "tool",
                            "parts": [
                                {
                                    "function_response": {
                                        "name": fn,
                                        "response": wrapped_result
                                    }
                                }
                            ]
                        }
                    ]
                )

                st.subheader("AI Response")

                final_part = followup.candidates[0].content.parts[0]

                if hasattr(final_part, "text"):
                    st.write(final_part.text)

                else:
                    summary_prompt = f"""
Explain the following financial real estate data clearly for a business user.

Data:
{wrapped_result}
"""

                    final_response = agent_model.generate_content(summary_prompt)

                    st.write(final_response.text)

            else:

                st.subheader("AI Response")

                part = response.candidates[0].content.parts[0]

                if hasattr(part, "text"):
                    st.write(part.text)

                else:
                    st.write("No response generated.")