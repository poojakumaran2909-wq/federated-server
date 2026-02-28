from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import copy
import base64
import io

app = FastAPI()

# Simple CNN model (same as client)
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
        return x  # not used on server

# Global model stored on server
global_model = SimpleCNN()

# Store client updates temporarily
client_updates = []

# Data format for receiving model
class ModelUpdate(BaseModel):
    weights: str  # base64 encoded model weights


@app.get("/")
def home():
    return {"message": "Federated Server Running"}


# Client downloads global model
@app.get("/get_model")
def get_model():
    buffer = io.BytesIO()
    torch.save(global_model.state_dict(), buffer)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return {"weights": encoded}


# Client sends update
@app.post("/send_update")
def receive_update(update: ModelUpdate):
    global client_updates

    decoded = base64.b64decode(update.weights)
    buffer = io.BytesIO(decoded)
    state_dict = torch.load(buffer)

    client_updates.append(state_dict)

    print(f"Received update. Total received: {len(client_updates)}")

    # If 3 clients sent updates → aggregate
    if len(client_updates) >= 3:
        aggregate_models()

    return {"status": "update received"}


def aggregate_models():
    global global_model, client_updates

    print("Aggregating models...")

    new_state = copy.deepcopy(client_updates[0])

    for key in new_state.keys():
        new_state[key] = sum(
            [client[key] for client in client_updates]
        ) / len(client_updates)

    global_model.load_state_dict(new_state)
    client_updates = []

    print("Aggregation complete.")