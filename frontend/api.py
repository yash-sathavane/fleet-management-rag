import requests

API_URL = "http://127.0.0.1:8000/ask"


def ask_question(query: str):
    """
    Sends the user's question to the FastAPI backend.
    Returns the JSON response.
    """

    try:
        response = requests.post(
            API_URL,
            json={
                "query": query
            },
            timeout=60
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.ConnectionError:

        return {
            "error": "Unable to connect to backend.\n\nIs FastAPI running?"
        }

    except requests.exceptions.Timeout:

        return {
            "error": "Backend request timed out."
        }

    except Exception as exc:

        return {
            "error": str(exc)
        }