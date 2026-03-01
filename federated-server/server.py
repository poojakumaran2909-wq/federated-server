from fastapi import FastAPI
import torch
import torch.nn as nn
import base64
import io
import copy

app = FastAPI()

# =========================
# Global Variables
# =========================
global_model = None
client_updates = []
total_updates = 0   # 👈 NEW (for frontend)


# =========================
# Define Model
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


# Initialize global model
global_model = SimpleCNN()


# =========================
# Root Route
# =========================
@app.get("/")
def home():
    return {"message": "Federated Server Running"}


# =========================
# NEW STATUS ROUTE (For Frontend)
# =========================
@app.get("/status")
def status():
    return {
        "message": "Federated Server Running",
        "total_updates": total_updates
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
    global client_updates
    global global_model
    global total_updates

    encoded_weights = data["weights"]
    decoded = base64.b64decode(encoded_weights)
    buffer = io.BytesIO(decoded)
    state_dict = torch.load(buffer)

    client_updates.append(state_dict)
    total_updates += 1   # 👈 increment for frontend

    print(f"Received update. Total received: {len(client_updates)}")

    # Aggregate after 3 clients
    if len(client_updates) >= 3:
        print("Aggregating models...")
        global_model = average_weights(client_updates)
        client_updates = []
        print("Aggregation complete.")

    return {"status": "update received"}


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
