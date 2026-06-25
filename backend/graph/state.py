from typing import Any, TypedDict


class FleetState(TypedDict):
    query: str
    intent: str
    response: str
    error: str

    manual_context: str
    retrieved_chunks: list[str]
    truck_data: Any
    prompt: str