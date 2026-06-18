from langgraph.graph import StateGraph, END

from graph.state import FleetState
from agents.supervisor_agent import supervisor_node
from agents.information_agent import information_agent

builder = StateGraph(FleetState)

builder.add_node(
    "supervisor",
    supervisor_node
)

builder.add_node(
    "information_agent",
    information_agent
)

builder.set_entry_point("supervisor")


def route_after_supervisor(state: FleetState):

    # invalid query
    if state.get("intent") == "invalid":
        return END

    # first pass through supervisor
    if not state.get("response"):
        return "information_agent"

    # second pass through supervisor
    return END


builder.add_conditional_edges(
    "supervisor",
    route_after_supervisor
)

builder.add_edge(
    "information_agent",
    "supervisor"
)

graph = builder.compile()