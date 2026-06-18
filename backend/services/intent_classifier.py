from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Lightweight local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

INTENTS = {
    "manual": """
    How do I reset the GPS tracker?
    How do I repair a truck component?
    What is the maintenance procedure?
    How do I troubleshoot engine issues?
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
    embedding = model.encode(text)
    return np.array(embedding).reshape(1, -1)


# Generate once during startup
manual_embedding = get_embedding(INTENTS["manual"])
iot_embedding = get_embedding(INTENTS["iot"])


def classify_intent(query: str):

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
        return "invalid"

    # Hybrid
    if (
        manual_score > MIN_INTENT_SCORE
        and
        iot_score > MIN_INTENT_SCORE
        and
        abs(manual_score - iot_score) < HYBRID_DELTA
    ):
        return "hybrid"

    # Manual vs IoT
    return "manual" if manual_score > iot_score else "iot"