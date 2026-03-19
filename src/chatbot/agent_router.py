from chatbot.vertex_client import ask_llm
from chatbot.prompts import SYSTEM_PROMPT


# ----------------------------------------
# FINAL ROUTER (ADK-COMPATIBLE)
# ----------------------------------------

def route_question(question):

    # System prompt + user question
    prompt = SYSTEM_PROMPT + "\n\nUser question: " + question

    # Send everything to Vertex AI Agent
    response = ask_llm(prompt)

    return response
