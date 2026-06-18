from typing import TypedDict


class FleetState(TypedDict):
    query: str
    intent: str
    response: str
    error: str