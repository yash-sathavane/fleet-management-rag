from typing import TypedDict

from langgraph.graph import (
    StateGraph,
    END
)

from information_agent import (
    information_agent
)

class AgentState(TypedDict):
    question: str
    answer: str

def supervisor_node(state):

    print("\nSupervisor Node")

    return {
        "question": state["question"]
    }

def information_node(state):

    answer = information_agent(
        state["question"]
    )

    return {
        "question": state["question"],
        "answer": answer
    }

graph_builder = StateGraph(
    AgentState
)

graph_builder.add_node(
    "supervisor",
    supervisor_node
)

graph_builder.add_node(
    "information",
    information_node
)

graph_builder.set_entry_point(
    "supervisor"
)

graph_builder.add_edge(
    "supervisor",
    "information"
)

graph_builder.add_edge(
    "information",
    END
)

graph = graph_builder.compile()

response = graph.invoke(
    {
        "question":
        "Is TruckA temperature normal?"
    }
)

print("\nFinal Answer:\n")
print(response["answer"])