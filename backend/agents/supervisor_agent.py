from graph.state import FleetState
from services.intent_classifier import classify_intent


def supervisor_node(state: FleetState):

    # Returning from Information Agent
    if state.get("response"):
        return {}

    query = state["query"]

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