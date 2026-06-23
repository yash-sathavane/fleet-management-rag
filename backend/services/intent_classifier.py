from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

def _load_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    except Exception as exc:
        print(f"Falling back to keyword intent detection: {exc}")
        return None


# Lightweight local embedding model, with offline fallback if the model is unavailable.
model = _load_model()

INTENTS = {
    "manual": """
    How do I reset the GPS tracker?
    How do I repair a truck component?
    What is the maintenance procedure?
    How do I troubleshoot engine issues?
    How do I troubleshoot overheating if TruckA is running hot?
    Is TruckA running hot?
    What are the operating instructions?
    How do I calibrate sensors?
    How do I perform maintenance?
    Vehicle manuals and documentation.
    """,

    "iot": """
    What is the truck fuel level?
    What is the engine temperature?
    Where is the truck located?
    What is the truck status?
    Is the truck overheating?
    What are the sensor readings?
    Live truck telemetry data.
    """
}


MIN_INTENT_SCORE = 0.30 
HYBRID_DELTA = 0.15


def get_embedding(text: str):
    if model is None:
        keywords = [
            "manual",
            "maintenance",
            "repair",
            "truck",
            "sensor",
            "gps",
            "engine",
            "fuel",
            "temperature",
            "location",
            "status",
            "telemetry",
            "iot",
        ]
        text_l = text.lower()
        return np.array([[1.0 if keyword in text_l else 0.0 for keyword in keywords]])

    embedding = model.encode(text)
    return np.array(embedding).reshape(1, -1)


# Generate once during startup
manual_embedding = get_embedding(INTENTS["manual"])
iot_embedding = get_embedding(INTENTS["iot"])


def classify_intent(query: str):
    intent = None

    if model is None:
        query_l = query.lower()
        manual_hits = [
            "manual",
            "maintenance",
            "repair",
            "gps",
            "engine",
            "sensor",
        ]
        iot_hits = [
            "fuel",
            "temperature",
            "location",
            "status",
            "telemetry",
            "iot",
        ]
        manual_score = sum(1 for term in manual_hits if term in query_l)
        iot_score = sum(1 for term in iot_hits if term in query_l)
        if manual_score == 0 and iot_score == 0:
            intent = "invalid"
            print(f"Intent: {intent}")
            return intent
        if manual_score > 0 and iot_score > 0:
            intent = "hybrid"
            print(f"Intent: {intent}")
            return intent
        intent = "manual" if manual_score > iot_score else "iot"
        print(f"Intent: {intent}")
        return intent

    query_embedding = get_embedding(query)

    manual_score = cosine_similarity(
        query_embedding,
        manual_embedding
    )[0][0]

    iot_score = cosine_similarity(
        query_embedding,
        iot_embedding
    )[0][0]

    print(f"Manual Score: {manual_score:.4f}")
    print(f"IoT Score: {iot_score:.4f}")

    # Invalid
    if (
        manual_score < MIN_INTENT_SCORE
        and
        iot_score < MIN_INTENT_SCORE
    ):
        intent = "invalid"
        print(f"Intent: {intent}")
        return intent

    # Hybrid
    if (
        manual_score > MIN_INTENT_SCORE
        and
        iot_score > MIN_INTENT_SCORE
        and
        abs(manual_score - iot_score) < HYBRID_DELTA
    ):
        intent = "hybrid"
        print(f"Intent: {intent}")
        return intent

    # Manual vs IoT
    intent = "manual" if manual_score > iot_score else "iot"
    print(f"Intent: {intent}")
    return intent
