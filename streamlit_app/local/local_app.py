import streamlit as st
import joblib
import numpy as np
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
# PRESS RELEASE FUNCTION
# ----------------------------------------

def load_press_releases():

    with open("../data/raw/press_releases.json") as f:
        data = json.load(f)

    return pd.DataFrame(data)


# ----------------------------------------
# LOAD ML MODELS
# ----------------------------------------

regression_model = joblib.load("../models/regression/random_forest_model.pkl")

classification_model = joblib.load("../models/classification/logistic_model.pkl")

selected_features = joblib.load("../models/classification/features.pkl")

#scaler = joblib.load("../models/classification/scaler.pkl")


# ----------------------------------------
# APP TITLE
# ----------------------------------------

st.title("🏡 Real Estate Financial Assistant")

st.write("Machine Learning powered predictions for real estate and customer behavior.")


# ----------------------------------------
# TABS
# ----------------------------------------

tab1, tab2, tab3 = st.tabs([
    "🏠 House Price Prediction",
    "👤 Customer Subscription",
    "🤖 AI Financial Assistant"
])


# ----------------------------------------
# REGRESSION TAB
# ----------------------------------------

with tab1:

    st.header("House Price Prediction")

    col1, col2, col3 = st.columns(3)

    load_example = col1.button("Load Example", key="example_reg")
    predict = col2.button("Predict House Price", key="predict_reg")
    clear_inputs = col3.button("Clear Inputs", key="clear_reg")

    if "example_loaded" not in st.session_state:
        st.session_state.example_loaded = False

    example_values = [5, 20, 6, 2, 1000, 3, 34, -118]

    if load_example:
        st.session_state.example_loaded = True
        st.rerun()

    if clear_inputs:
        st.session_state.example_loaded = False
        st.rerun()

    if st.session_state.example_loaded:
        defaults = example_values
    else:
        defaults = [0] * 8

    st.subheader("Property Features")

    col1, col2 = st.columns(2)

    with col1:
        medinc = st.number_input("Median Income", value=defaults[0])
        avg_rooms = st.number_input("Average Rooms", value=defaults[2])
        population = st.number_input("Population", value=defaults[4])
        latitude = st.number_input("Latitude", value=defaults[6])

    with col2:
        house_age = st.number_input("House Age", value=defaults[1])
        avg_bedrooms = st.number_input("Average Bedrooms", value=defaults[3])
        avg_occupancy = st.number_input("Average Occupancy", value=defaults[5])
        longitude = st.number_input("Longitude", value=defaults[7])

    st.divider()

    if predict:

        features = np.array([
            medinc,
            house_age,
            avg_rooms,
            avg_bedrooms,
            population,
            avg_occupancy,
            latitude,
            longitude
        ]).reshape(1, -1)

        prediction = regression_model.predict(features)[0]

        price = prediction * 100000

        st.success(f"Predicted House Price: ${price:,.0f}")

        st.metric(
            label="Estimated Price",
            value=f"${price:,.0f}"
        )


# ----------------------------------------
# CLASSIFICATION TAB
# ----------------------------------------

with tab2:

    st.header("Customer Subscription Prediction")

    col1, col2, col3 = st.columns(3)

    load_example = col1.button("Load Example")
    predict_sub = col2.button("Predict Subscription")
    clear_class = col3.button("Clear Inputs")

    example_values = {
        "campaign": 0,
        "pdays": 999,
        "previous": 0,
        "job_blue-collar": "No",
        "job_retired": "No",
        "job_student": "No",
        "marital_single": "No",
        "education_tertiary": "No",
        "housing_yes": "Yes",
        "loan_yes": "No",
        "contact_unknown": "No",
        "month_dec": "No",
        "month_mar": "No",
        "month_may": "Yes",
        "month_oct": "No",
        "month_sep": "No",
        "poutcome_success": "No",
        "poutcome_unknown": "Yes"
    }

    categorical_features = [
        "job_blue-collar","job_retired","job_student",
        "marital_single","education_tertiary",
        "housing_yes","loan_yes",
        "contact_unknown",
        "month_dec","month_mar","month_may","month_oct","month_sep",
        "poutcome_success","poutcome_unknown"
    ]

    if load_example:
        for k, v in example_values.items():
            st.session_state[k] = v
        st.rerun()

    st.subheader("Customer Features")

    col1, col2 = st.columns(2)

    input_features = []

    for i, feature in enumerate(selected_features):

        label = feature.replace("_", " ").title()

        column = col1 if i % 2 == 0 else col2

        with column:

            if feature in categorical_features:

                val = st.selectbox(
                    label,
                    ["No", "Yes"],
                    key=feature
                )

                val = 1 if val == "Yes" else 0

            else:

                val = st.number_input(
                    label,
                    key=feature
                )

        input_features.append(val)

    st.divider()

    if predict_sub:

        features = np.array(input_features).reshape(1, -1)

        #features = scaler.transform(features)

        prediction = int(classification_model.predict(features)[0])
        probability = float(classification_model.predict_proba(features)[0][1])

        prob_percent = probability * 100

        if prob_percent < 0.01:
            prob_display = "<0.01%"
        else:
            prob_display = f"{prob_percent:.2f}%"

        if prediction == 1:
            st.success(f"Customer is likely to subscribe ({prob_display})")
        else:
            st.error(f"Customer is unlikely to subscribe ({prob_display})")

        st.metric("Subscription Probability", prob_display)

        st.progress(min(probability, 1.0))

    if clear_class:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ----------------------------------------
# AI FINANCIAL ASSISTANT TAB
# ----------------------------------------

with tab3:

    st.header("📊 Property Financial Data")

    if st.button("Load Property Data"):

        df = get_property_data()
        st.dataframe(df)


    st.header("🤖 AI Financial Assistant")

    user_question = st.text_input(
        "Ask a question about properties or financial data (Example: Show Chicago properties)"
    )

    col1, col2 = st.columns(2)

    ask = col1.button("Ask", key="ask_chat")
    clear_chat = col2.button("Clear Chat", key="clear_chat")


    if ask:

        question = user_question.lower()

        df = get_property_data()

        if "chicago" in question:

            st.write("Showing properties in Chicago")
            st.dataframe(df[df["metro_area"] == "Chicago"])

        elif "new york" in question:

            st.write("Showing properties in New York")
            st.dataframe(df[df["metro_area"] == "New York"])

        elif "industrial" in question:

            st.write("Showing industrial properties")
            st.dataframe(df[df["property_type"] == "Industrial"])

        elif "revenue" in question:

            st.write("Showing revenue and income information")
            st.dataframe(df[["address", "revenue", "net_income"]])

        elif "press" in question:

            st.write("Showing press releases")
            press_df = load_press_releases()
            st.dataframe(press_df)

        else:

            st.warning(
                "I couldn't understand the question. Try asking about Chicago properties, revenue, or press releases."
            )


    if clear_chat:

        st.rerun()