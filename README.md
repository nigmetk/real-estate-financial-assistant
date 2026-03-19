# 🏠 AI Financial Assistant Web Application

An end-to-end **AI-powered real estate analytics platform** that integrates machine learning, cloud services, and a multi-source data system into a single intelligent web application.

---

## 🚀 Overview

This application provides a unified interface to:

* 📈 Predict house prices (Regression – AWS SageMaker)
* ⚠️ Predict customer subscription (Classification – AWS SageMaker)
* 🏢 Query real estate financial data (PostgreSQL)
* 📰 Analyze company press releases (JSON dataset)
* 🤖 Interact with an AI-powered financial assistant (Vertex AI)

---

## ☁️ Multi-Cloud Architecture

* **AWS SageMaker** → Model deployment (Regression & Classification)
* **Google Vertex AI** → Conversational AI agent

This project demonstrates real-world **multi-cloud integration**.

---

## 🤖 AI Agent System

The chatbot is implemented using a **tool-based agent architecture**:

* Interprets user queries
* Dynamically selects data sources:

  * PostgreSQL (property & financial data)
  * SEC reports (JSON)
  * Press releases (JSON)
* Executes tool calls
* Returns natural language responses

Includes **routing logic** to guide tool selection.

---

## 🧮 Machine Learning Models

### Regression

* Random Forest (California Housing dataset)
* Metrics: RMSE, MAE, R²

### Classification

* Logistic Regression (Bank Marketing dataset)
* Metrics: Accuracy, Precision, Recall, F1 Score

Both models are deployed on **AWS SageMaker endpoints**.

---

## 🗄️ Database

PostgreSQL with two tables:

* **Properties** → location, type, size
* **Financials** → revenue, net income, expenses

---

## ⚠️ Database Handling

The application connects to a local PostgreSQL database.

If the database is unavailable, the system **automatically falls back to sample data**, ensuring the application always runs successfully.

---

## ⚙️ Tech Stack

* Python, Streamlit
* Pandas, NumPy, Scikit-learn
* PostgreSQL (psycopg2)

Cloud:

* AWS SageMaker (model hosting)
* Google Vertex AI (LLM agent)

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app/aws_vertex_app.py
```

### 🔐 Environment Variables

Create a `.env` file in the root directory:

```bash
DB_PASSWORD=your_password_here
```

---

## 📂 Project Structure

```
real-estate-financial-assistant
├── data/
├── models/
├── notebooks/
├── sql/
├── src/
├── streamlit_app/
├── requirements.txt
└── README.md
```

---

## 🏗️ System Architecture

```
User
 │
 ▼
Streamlit Web App
 │
 ├── SageMaker (ML Models)
 ├── PostgreSQL (Structured Data)
 ├── JSON (Press / SEC Data)
 │
 ▼
AI Agent (Vertex AI)
 │
 ▼
Results Layer
```

---

## 🌟 Key Highlights

* End-to-end AI system
* Multi-cloud integration (AWS + GCP)
* Real-time predictions via deployed models
* Tool-based AI agent with data routing
* Robust fallback handling for database failures
