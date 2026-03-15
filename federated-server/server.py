from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import base64
import io
import copy
import uuid

registered_clients = {}
app = FastAPI()

# =========================
# CORS (Allow Frontend Access)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Global Variables
# =========================
client_updates = []
total_updates = 0
round_number = 0


# =========================
# Model Definition
# =========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 6 * 6)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize Global Model
global_model = SimpleCNN()


# =========================
# Root Route
# =========================
@app.get("/")
def home():
    return {"message": "Federated Server Running"}


# =========================
# Status Route (Frontend Uses This)
# =========================
@app.get("/status")
def status():
    return {
        "message": "Federated Server Running",
        "total_updates": total_updates,
        "round_number": round_number
    }


# =========================
# Send Global Model To Client
# =========================
@app.get("/get_model")
def get_model():
    buffer = io.BytesIO()
    torch.save(global_model.state_dict(), buffer)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return {"weights": encoded}


# =========================
# Receive Client Update
# =========================
@app.post("/send_update")
def receive_update(data: dict):

    client_id = data.get("client_id")
    api_key = data.get("api_key")

    if client_id not in registered_clients:
        return {"error": "client not registered"}

    if registered_clients[client_id]["api_key"] != api_key:
        return {"error": "invalid api key"}

    # continue with update logic

@app.post("/register_client")
def register_client():
    client_id = str(uuid.uuid4())
    api_key = str(uuid.uuid4())

    registered_clients[client_id] = {
        "api_key": api_key
    }

    return {
        "client_id": client_id,
        "api_key": api_key
    }
    
# =========================
# Federated Averaging
# =========================
def average_weights(models):
    avg_model = copy.deepcopy(global_model)

    for key in avg_model.state_dict().keys():
        avg_model.state_dict()[key].data.copy_(
            sum(model[key] for model in models) / len(models)
        )

    return avg_model
