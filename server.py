from fastapi import FastAPI
from pydantic import BaseModel
import uuid
from datetime import datetime

app = FastAPI()

# -----------------------------
# In‑memory storage
# -----------------------------

registered_clients = {}
model_updates = []

global_model = None
model_version = 0
round_number = 0

training_metrics = {
    "total_updates": 0,
    "round": 0,
    "registered_clients": 0,
    "model_version": 0
}

# -----------------------------
# Data model for client update
# -----------------------------

class ModelUpdate(BaseModel):
    client_id: str
    api_key: str
    weights: dict


# -----------------------------
# Server status endpoint
# -----------------------------

@app.get("/status")
def status():
    return {
        "message": "Federated Server Running",
        "round": round_number,
        "total_updates": training_metrics["total_updates"],
        "registered_clients": len(registered_clients),
        "model_version": model_version
    }


# -----------------------------
# Client registration
# -----------------------------

@app.post("/register_client")
def register_client():

    client_id = str(uuid.uuid4())
    api_key = str(uuid.uuid4())

    registered_clients[client_id] = {
        "api_key": api_key,
        "registered_at": str(datetime.now())
    }

    training_metrics["registered_clients"] = len(registered_clients)

    return {
        "client_id": client_id,
        "api_key": api_key
    }


# -----------------------------
# Receive model update
# -----------------------------

@app.post("/send_update")
def receive_update(update: ModelUpdate):

    # Verify client
    if update.client_id not in registered_clients:
        return {"error": "client not registered"}

    if registered_clients[update.client_id]["api_key"] != update.api_key:
        return {"error": "invalid api key"}

    # Store update
    model_updates.append(update.weights)

    training_metrics["total_updates"] += 1

    # Aggregation trigger
    if len(model_updates) >= 3:
        aggregate_models()

    return {
        "message": "update received",
        "total_updates": training_metrics["total_updates"]
    }


# -----------------------------
# Model aggregation
# -----------------------------

def aggregate_models():

    global global_model
    global model_updates
    global model_version
    global round_number

    # Simple averaging placeholder
    global_model = {
        "aggregated_weights": "placeholder"
    }

    model_version += 1
    round_number += 1

    training_metrics["round"] = round_number
    training_metrics["model_version"] = model_version

    model_updates = []

    print(f"Model aggregated → version {model_version}")


# -----------------------------
# Get latest global model
# -----------------------------

@app.get("/get_global_model")
def get_global_model():

    if global_model is None:
        return {"message": "No model available yet"}

    return {
        "model_version": model_version,
        "model": global_model
    }


# -----------------------------
# Training metrics endpoint
# -----------------------------

@app.get("/metrics")
def get_metrics():

    return {
        "round": round_number,
        "total_updates": training_metrics["total_updates"],
        "registered_clients": len(registered_clients),
        "model_version": model_version
    }