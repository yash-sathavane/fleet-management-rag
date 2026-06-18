from graph.state import FleetState


def information_agent(state: FleetState):

    return {
        "response":
            f"Information Agent received {state['intent']} query"
    }