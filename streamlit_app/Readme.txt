🤖 AI Financial Assistant (Streamlit App)

This Streamlit application includes an AI-powered financial assistant designed to operate in both Demo (Cloud) and Production (Local/GCP) modes.

☁️ Demo Mode (Streamlit Cloud)

When deployed on Streamlit Cloud, the app runs in Demo Mode.

No external AI services (Vertex AI) are required

Responses are generated using predefined financial data

Ensures stable performance without credentials

Example supported questions:

What was the net income reported last quarter?

Show industrial properties in the Chicago region with revenue details

Did the company announce any acquisitions recently?

💻 Production Mode (Local / GCP)

When running locally with IS_PROD=true, the app enables full AI capabilities:

Uses Vertex AI (Gemini)

Supports tool-based reasoning

Dynamically retrieves:

Property financial data

SEC reports

Press releases

⚙️ Mode Configuration

The application mode is controlled via an environment variable:

IS_PROD=true   # Production mode (Vertex AI + tools)
IS_PROD=false  # Demo mode (default for Streamlit Cloud)
🧠 Key Capabilities

Natural language query understanding

Intelligent routing to relevant data sources

Tool-based data retrieval

Context-aware response generation

Multi-source financial insights

⚠️ Notes

Vertex AI is not configured on Streamlit Cloud by default

Demo Mode simulates AI behavior for safe public deployment

Full AI functionality is available in local or GCP environments

🚀 Summary
Mode	Environment	Features
Demo	Streamlit Cloud	Simulated AI responses
Prod	Local / GCP	Full AI agent with tool calling