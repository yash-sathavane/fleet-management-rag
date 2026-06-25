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
    How do I reconnect the tracker to the fleet dashboard?
    What is the maintenance schedule for the TitanX 4000?
    When should the engine oil be changed?
    When should the air filter be replaced?
    When should engine diagnostics be performed?
    What should I check before starting the engine?
    What is the recommended tyre pressure?
    What is the normal engine temperature range?
    What should I do if the fuel level drops below 15%?
    How can I prevent fuel pump damage?
    What safety precautions should drivers follow?
    What is the maximum safe driving speed?
    What should I do if the IoT dashboard reports a temperature anomaly?
    What is the fuel capacity of the TitanX 4000?
    What engine does the TitanX 4000 use?

    What is the GVW of the Blazo X 28 Cargo?
    What is the GVW of the Blazo X 42?
    What engine powers the Blazo X?
    What is the maximum engine power?
    What is the maximum torque?
    What is the fuel tank capacity?
    What is the AdBlue tank capacity?
    What gearbox is used?
    What suspension system does the Blazo X use?
    What steering system is provided?
    What braking system is used?
    What are the cabin features?
    What comfort features are available?
    What safety features are available?
    What is FuelSmart technology?
    Explain mPOWER FuelSmart.
    What is iMAXX telematics?
    What information is available in the Driver Information System?
    What warranty does Mahindra provide?
    What is the Double Service Guarantee?
    What is MTrust?
    What is MCover?
    What is MAashray?
    What applications is the Blazo X suitable for?
    Compare the different Blazo X variants.
    Explain the Blazo truck specifications.
    """,

    "iot": """
    What is TruckA's current fuel level?
    What is TruckB's engine temperature?
    Where is TruckA currently located?
    What is TruckB's current speed?
    Show the live telemetry for all trucks.
    Show the current fleet status.
    Is TruckA overheating?
    Is TruckB overheating?
    Which truck has the lowest fuel?
    Which truck has the highest engine temperature?
    Which truck is moving the fastest?
    Which truck needs immediate attention?
    Which truck is offline?
    Show all live sensor readings.
    Show the current IoT dashboard data.
    """
}
MIN_INTENT_SCORE = 0.24
#HYBRID_DELTA = 0.15


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
       
    ):
        intent = "hybrid"
        print(f"Intent: {intent}")
        return intent

    # Manual vs IoT
    intent = "manual" if manual_score > iot_score else "iot"
    print(f"Intent: {intent}")
    return intent
