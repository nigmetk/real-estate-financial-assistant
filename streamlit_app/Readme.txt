# 🤖 AI Financial Assistant (Streamlit App)

This Streamlit application includes an AI-powered financial assistant designed to operate across multiple deployment environments, supporting both lightweight cloud execution and full AI capabilities.

---

## ☁️ Cloud Deployment

When deployed on Streamlit Cloud, the application runs using a **lightweight simulation layer**.

* No external AI services (e.g., Vertex AI) are required
* Responses are generated using predefined financial data
* Ensures stable, secure, and credential-free deployment

### Example Supported Queries

* What was the net income reported last quarter?
* Show industrial properties in the Chicago region with revenue details
* Did the company announce any acquisitions recently?

---

## 💻 Full AI Deployment (Local / GCP)

When running with `IS_PROD=true`, the application enables full AI capabilities:

* Uses **Vertex AI (Gemini)**
* Supports **tool-based reasoning**
* Dynamically retrieves data from:

  * Property financial datasets
  * SEC reports
  * Press releases

---

## ⚙️ Mode Configuration

The application mode is controlled via an environment variable:

```bash
IS_PROD=true   # Full AI mode (Vertex AI + tools)
IS_PROD=false  # Cloud-compatible execution mode
```

---

## 🧠 Key Capabilities

* Natural language query understanding
* Intelligent routing to relevant data sources
* Tool-based data retrieval
* Context-aware response generation
* Multi-source financial insights

---

## 🚀 Summary

| Mode               | Environment     | Capabilities                                              |
| ------------------ | --------------- | --------------------------------------------------------- |
| Cloud Deployment   | Streamlit Cloud | Lightweight simulation layer for stable execution         |
| Full AI Deployment | Local / GCP     | Full AI agent with tool calling and real-time data access |

---
