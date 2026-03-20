🏠 AI Financial Assistant Web Application

An end-to-end AI-powered real estate analytics platform that integrates machine learning, cloud services, and multi-source data into a single intelligent web application.

🚀 Overview

This application provides a unified interface to:

📈 Predict house prices (Regression – AWS SageMaker)

⚠️ Predict customer subscription (Classification – AWS SageMaker)

🏢 Query real estate financial data (PostgreSQL)

📰 Analyze company press releases (JSON dataset)

🤖 Interact with an AI-powered financial assistant (Vertex AI)

☁️ Multi-Cloud Architecture

This project demonstrates real-world multi-cloud system design:

AWS SageMaker → Deployment of regression and classification models

Google Vertex AI → Conversational AI agent (LLM-powered chatbot)

🤖 AI Agent System

The chatbot is implemented using a tool-based agent architecture.

Capabilities:

Interprets natural language queries

Dynamically selects appropriate data sources:

PostgreSQL (structured financial & property data)

SEC financial reports (JSON)

Press releases (JSON)

Executes tool calls

Generates natural language responses via Vertex AI

Intelligence Layer:

Query routing logic

Multi-source response generation

Context-aware answers

## ☁️ Demo vs Production Mode

This application is designed with a dual-mode architecture to support both public deployment and full AI functionality.

### 🔹 Demo Mode (Streamlit Cloud)

* Runs without external AI credentials
* Uses predefined financial data for responses
* Ensures stable and secure cloud deployment

### 🔹 Production Mode (Local / GCP)

* Enables full **Vertex AI agent capabilities**
* Supports **tool-based reasoning and data retrieval**
* Connects to:

  * PostgreSQL database
  * SEC financial datasets
  * Press release data

### ⚙️ Configuration

```bash
IS_PROD=true   # Enable full AI (local)
IS_PROD=false  # Demo mode (cloud)
```

> ⚠️ Note: Vertex AI is not enabled on Streamlit Cloud by default. Demo Mode is intentionally used for safe deployment.


🧮 Machine Learning Models
🔹 Regression Model

Model: Random Forest Regressor

Dataset: California Housing (scikit-learn)

Metrics:

RMSE

MAE

R² Score

🔹 Classification Model

Model: Logistic Regression

Dataset: Bank Marketing (UCI)

Metrics:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

🚀 Deployment

Both models are deployed as AWS SageMaker endpoints and integrated into the Streamlit application for real-time predictions.

🗄️ Database

PostgreSQL database with two tables:

Properties

property_id (PK)

address

metro_area

sq_footage

property_type

Financials

property_id (FK)

revenue

net_income

expenses

⚠️ Database Handling

The application connects to a local PostgreSQL database.

If unavailable, the system automatically falls back to sample/mock data, ensuring uninterrupted execution.

📂 Data Sources

The system integrates three primary data sources:

SEC Financial Reports (JSON)

PostgreSQL Database (structured data)

Press Releases (JSON)

⚙️ Tech Stack
Core

Python

Streamlit

Pandas, NumPy, Scikit-learn

Database

PostgreSQL (psycopg2)

Cloud

AWS SageMaker (ML deployment)

Google Vertex AI (LLM / AI Agent)

▶️ Run Locally
pip install -r requirements.txt
streamlit run streamlit_app/aws_vertex_app.py
🔐 Environment Variables

Create a .env file in the root directory:

DB_PASSWORD=your_password_here
📂 Project Structure
real-estate-financial-assistant/
├── data/                      # Raw and processed datasets
├── inference/                 # SageMaker inference scripts
│   ├── regression_inference.py
│   └── classification_inference.py
├── models/                    # Placeholder (models deployed in SageMaker)
│   └── README.md
├── notebooks/                 # EDA and model training
├── sql/                       # Database schema and queries
├── src/                       # Core backend logic
├── streamlit_app/             # Streamlit UI
│   ├── aws_vertex_app.py
│   └── components/
├── requirements.txt
└── README.md
🏗️ System Architecture
User
 │
 ▼
Streamlit Web App
 │
 ├── AWS SageMaker (ML Models)
 ├── PostgreSQL (Structured Data)
 ├── JSON (Press / SEC Data)
 │
 ▼
Vertex AI Agent (LLM)
 │
 ▼
Final Response Layer
🌟 Key Highlights

End-to-end AI system architecture

Multi-cloud integration (AWS + GCP)

Real-time predictions via deployed ML models

Tool-based AI agent with intelligent routing

Robust fallback handling for database failures
