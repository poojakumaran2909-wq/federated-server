from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uuid
import torch
import json
from datetime import datetime
import os
from database import SessionLocal, engine
from models import Base, Client, Update, ModelVersion


Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/global_model.pt"

if os.path.exists(MODEL_PATH):
    model = torch.load(MODEL_PATH)
else:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    torch.save(model, MODEL_PATH)

# -------------------------
# Register Client
# -------------------------

@app.post("/register_client")
def register():

    db = SessionLocal()

    client_id = str(uuid.uuid4())
    api_key = str(uuid.uuid4())

    client = Client(
        client_id=client_id,
        api_key=api_key
    )

    db.add(client)
    db.commit()

    return {
        "client_id": client_id,
        "api_key": api_key
    }


# -------------------------
# Get Global Model
# -------------------------

@app.get("/get_model")
def get_model():

    weights = model.state_dict()

    return {
        "weights": {k: v.tolist() for k, v in weights.items()}
    }


# -------------------------
# Receive Client Update
# -------------------------

@app.post("/send_update")
def receive(update: dict):

    db = SessionLocal()

    weights = update["weights"]
    client_id = update["client_id"]

    upd = Update(
        client_id=client_id,
        weights=json.dumps(weights)
    )

    db.add(upd)
    db.commit()

    updates = db.query(Update).all()

    if len(updates) >= 3:
        aggregate(db, updates)

    return {"message": "update stored"}


# -------------------------
# Federated Averaging
# -------------------------

def aggregate(db, updates):

    global model

    weight_list = []

    for u in updates:
        weight_list.append(json.loads(u.weights))

    avg = {}

    for key in weight_list[0]:

        avg[key] = sum(
            torch.tensor(w[key]) for w in weight_list
        ) / len(weight_list)

    model.load_state_dict(avg)

    torch.save(model, MODEL_PATH)

    version = db.query(ModelVersion).count() + 1

    model_record = ModelVersion(
        version=version,
        path=MODEL_PATH
    )

    db.add(model_record)

    db.query(Update).delete()

    db.commit()

    print("Model aggregated. Version:", version)


# -------------------------
# Server Status
# -------------------------
@app.get("/status")
def status():
    db = SessionLocal()
    try:
        return {
            "clients": db.query(Client).count(),
            "updates": db.query(Update).count(),
            "models": db.query(ModelVersion).count()
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        db.close()
