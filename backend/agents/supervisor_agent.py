from services.intent_classifier import classify_intent
from graph.state import FleetState

# My LangGraph Node Implementation
def supervisor_node(state: FleetState):
    # Returning from Information Agent
    if state.get("response"):
        return {}

    query = state.get("query", "")

    if not query.strip():
        return {
            "intent": "invalid",
            "error": "Empty query"
        }

    intent = classify_intent(query)

    if intent == "invalid":
        return {
            "intent": "invalid",
            "error": "Not a fleet management query"
        }

    return {
        "intent": intent
    }

# Teammate's Supervisor Agent Wrapper
from agents.information_agent import information_agent

def supervisor_agent(question):
    print("\nSupervisor Agent Received Query")

    if not question.strip():
        return "Error: Empty query"

    intent = classify_intent(question)
    
    if intent == "invalid":
        return "Error: Not a fleet management query"
        
    print(f"Intent classified as: {intent}")

    answer = information_agent(question)

    print("Information Agent Completed")

    return answer
