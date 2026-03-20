import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import json
import os
import joblib
import vertexai
import boto3
import os

IS_PROD = os.getenv("IS_PROD", "false").lower() == "true"

from dotenv import load_dotenv

from vertexai.generative_models import (
    GenerativeModel,
    Tool,
    FunctionDeclaration,
    ToolConfig
)

# ----------------------------------------
# PATH SETUP 
# ----------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(ROOT_DIR, "data", "raw")
MODEL_PATH = os.path.join(ROOT_DIR, "models")

# ----------------------------------------
# LOAD ENV 
# ----------------------------------------

load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

# ----------------------------------------
# VERTEX AI INIT
# ----------------------------------------

vertexai.init(
    project="real-estate-490304",
    location="global"
)

# ----------------------------------------
# LOAD FEATURES
# ----------------------------------------

features_path = os.path.join(
    MODEL_PATH,
    "sagemaker",
    "classification",
    "features.pkl"
)

try:
    selected_features = joblib.load(features_path)
except:
    print("Using fallback features")
    selected_features = [
        "campaign", "pdays", "previous",
        "job_blue-collar", "job_retired", "job_student",
        "marital_single", "education_tertiary",
        "housing_yes", "loan_yes",
        "contact_unknown",
        "month_dec", "month_mar", "month_may",
        "month_oct", "month_sep",
        "poutcome_success", "poutcome_unknown"
    ]

# ----------------------------------------
# DATABASE FUNCTION
# ----------------------------------------

def get_property_data():

    try:
        conn = psycopg2.connect(
            host="localhost",
            database="real_estate_db",
            user="postgres",
            password=os.getenv("DB_PASSWORD")
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

    except Exception:
        return pd.DataFrame([
            {
                "address": "123 Main St",
                "metro_area": "Chicago",
                "property_type": "Industrial",
                "revenue": 1200000,
                "net_income": 450000
            },
            {
                "address": "12 Warehouse Ln",
                "metro_area": "Chicago",
                "property_type": "Industrial",
                "revenue": 1800000,
                "net_income": 610000
            }
        ])

# ----------------------------------------
# DATA LOADERS
# ----------------------------------------

def load_sec_reports():
    with open(os.path.join(DATA_PATH, "sec_reports.json")) as f:
        return pd.DataFrame(json.load(f))

def load_press_releases():
    with open(os.path.join(DATA_PATH, "press_releases.json")) as f:
        return pd.DataFrame(json.load(f))

# ----------------------------------------
# TOOL FUNCTIONS
# ----------------------------------------

def chicago_industrial_revenue_tool():
    df = get_property_data()
    df = df[
        (df["metro_area"] == "Chicago") &
        (df["property_type"] == "Industrial")
    ]
    return df[["address", "revenue", "net_income"]].to_dict(orient="records")

def press_releases_tool():
    return load_press_releases().to_dict(orient="records")

def sec_reports_tool():
    return load_sec_reports().to_dict(orient="records")

def route_hint(q):
    if "revenue" in q or "income" in q:
        return "Use financial data tools"
    elif "property" in q or "Chicago" in q:
        return "Use property database tool"
    elif "press" in q or "announcement" in q:
        return "Use press release tool"
    else:
        return "Answer normally"

# ----------------------------------------
# TOOL DECLARATION 
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
# STREAMLIT UI
# ----------------------------------------

st.title("🏡 Real Estate Financial Assistant")

# -----------------------------
# MODE BADGE
# -----------------------------
if IS_PROD:
    st.success("🟢 Production Mode — Full AI (Vertex AI + Tools)")
else:
    st.warning("🟡 Demo Mode — Simulated AI (Cloud Safe)")

tab1, tab2, tab3 = st.tabs([
    "House Price Prediction",
    "Customer Subscription",
    "AI Financial Assistant"
])

# ----------------------------------------
# TAB 1 — REGRESSION (HOUSE PRICE)
# ----------------------------------------

with tab1:

    st.header("House Price Prediction")

    sagemaker_client = boto3.client("sagemaker-runtime", region_name="us-east-1")
    ENDPOINT_NAME = "regression-endpoint-1773932434"

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

        features = [
            medinc,
            avg_rooms,
            house_age,
            avg_bedrooms,
            population,
            avg_occupancy,
            longitude,
            latitude
        ]

        payload = json.dumps({"features": features})

        try:
            response = sagemaker_client.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType="application/json",
                Body=payload
            )
            result = json.loads(response["Body"].read().decode())
            price = result["prediction"][0]

        except Exception as e:
            print("Using dummy regression:", e)
            price = 350000

        st.success(f"Predicted House Price: ${price:,.0f}")

# ----------------------------------------
# TAB 2 — CLASSIFICATION
# ----------------------------------------

with tab2:

    st.header("Customer Subscription")

    sagemaker_client = boto3.client("sagemaker-runtime", region_name="us-east-1")
    CLASS_ENDPOINT = "classification-endpoint-1773932440"

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

        payload_dict = {feature: float(input_features[i]) for i, feature in enumerate(selected_features)}
        payload = json.dumps(payload_dict)

        try:
            response = sagemaker_client.invoke_endpoint(
                EndpointName=CLASS_ENDPOINT,
                ContentType="application/json",
                Body=payload
            )
            result = json.loads(response["Body"].read().decode())

            prediction = int(result["prediction"])
            probability = float(result["probability"])

        except Exception as e:
            print("Using dummy classification:", e)
            prediction = 1
            probability = 0.82

        st.markdown("### Result")

        if prediction == 1:
            st.success(f"✅ Customer likely to subscribe ({probability:.2%})")
        else:
            st.error(f"❌ Customer unlikely to subscribe ({probability:.2%})")

        st.markdown("**Subscription Probability**")
        st.progress(probability)


# ----------------------------------------
# TAB 3 — AI CHATBOT
# ----------------------------------------


with tab3:

    st.header("AI Financial Assistant")

    # -----------------------------
    # Chat memory
    # -----------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # -----------------------------
    # Clear Chat Button
    # -----------------------------
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state["chat_input"] = ""
        st.rerun()

    # -----------------------------
    # Display chat history (ChatGPT UI)
    # -----------------------------
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["q"])
        with st.chat_message("assistant"):
            st.write(chat["a"])

    # -----------------------------
    # User Input
    # -----------------------------
    question = st.text_input(
        "Ask about properties, financials, SEC reports or press releases",
        key="chat_input"
    )

    # -----------------------------
    # Ask AI Button
    # -----------------------------
    if st.button("Ask AI") and question:

        with st.spinner("AI is thinking..."):

            routing = route_hint(question)

            prompt = f"""
You are a financial real estate assistant.

Routing hint:
{routing}

You have tools to retrieve:
- Chicago industrial property revenue
- SEC financial reports
- Company press releases

Use tools whenever needed.

User question:
{question}
"""

            response = None
            answer = ""

            # -----------------------------
            # PROD MODE (REAL LLM + TOOLS)
            # -----------------------------
            if IS_PROD:
                try:
                    response = agent_model.generate_content(prompt)
                    if response and response.candidates:
                        part = response.candidates[0].content.parts[0]
                        if hasattr(part, "text"):
                            answer = part.text
                except Exception as e:
                    answer = "Error generating response from Vertex AI."

            # -----------------------------
            # DEMO MODE (CLOUD SAFE)
            # -----------------------------
            else:
                q = question.lower()
                answers = []

                # NET INCOME
                if "net income" in q or "last quarter" in q:
                    answers.append("The net income reported last quarter was approximately $450,000.")

                # CHICAGO
                if "chicago" in q or "industrial" in q or "revenue" in q:
                    answers.append(
                        "Chicago industrial properties:\n"
                        "- 123 Main St — Revenue: $1.2M, Net Income: $450k\n"
                        "- 12 Warehouse Ln — Revenue: $1.8M, Net Income: $610k"
                    )

                # PRESS
                if "acquisition" in q or "press" in q or "announcement" in q:
                    answers.append(
                        "The company recently announced an acquisition of a logistics facility "
                        "to expand its industrial portfolio."
                    )

                if not answers:
                    answers.append(
                        "This is a demo AI assistant. You can ask about financials, "
                        "Chicago properties, or company announcements."
                    )

                answer = "\n\n".join(answers)

        # -----------------------------
        # Display current response
        # -----------------------------
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            st.write(answer)

            # Metric UI example
            if "net income" in question.lower():
                st.metric("Net Income (Last Quarter)", "$450,000")

        # -----------------------------
        # Save chat history
        # -----------------------------
        st.session_state.chat_history.append({
            "q": question,
            "a": answer
        })

        # -----------------------------
        # TOOL CALL HANDLING (ONLY PROD)
        # -----------------------------
        if IS_PROD and response:

            fn_call = None
            try:
                part = response.candidates[0].content.parts[0]
                fn_call = getattr(part, "function_call", None)
            except:
                fn_call = None

            if fn_call is not None:

                fn = fn_call.name

                # 🔧 TOOL LOG
                st.info(f"🔧 Tool used: {fn}")

                # Execute tool
                if fn == "chicago_industrial_revenue_tool":
                    tool_result = chicago_industrial_revenue_tool()
                elif fn == "press_releases_tool":
                    tool_result = press_releases_tool()
                elif fn == "sec_reports_tool":
                    tool_result = sec_reports_tool()
                else:
                    tool_result = {"error": "Unknown tool called"}

                wrapped_result = {"data": tool_result}

                # Show tool output
                with st.expander("🔍 Tool Execution Details"):
                    st.json(wrapped_result)

                try:
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

                    if followup and followup.candidates:
                        final_part = followup.candidates[0].content.parts[0]

                        if hasattr(final_part, "text"):
                            with st.chat_message("assistant"):
                                st.write(final_part.text)

                except Exception as e:
                    st.error(f"Tool processing error: {e}")
