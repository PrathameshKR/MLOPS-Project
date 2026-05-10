import io
import os
import subprocess

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms


# =====================================================
# FastAPI App
# =====================================================

app = FastAPI()


# =====================================================
# Config
# =====================================================

MODEL_PATH = "models/cnn_model.pth"

LOG_FILE = "monitoring/prediction_logs.csv"

DRIFT_TRIGGER_COUNT = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================================
# Create Monitoring Directory
# =====================================================

os.makedirs(
    "monitoring",
    exist_ok=True
)


# =====================================================
# Class Names
# =====================================================

CLASS_NAMES = [

    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",

    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# =====================================================
# CNN Model Definition
# =====================================================

class SimpleCNN(nn.Module):

    def __init__(self):

        super(SimpleCNN, self).__init__()

        self.conv_layers = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 8 * 8,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                10
            )
        )

    def forward(self, x):

        x = self.conv_layers(x)

        x = self.fc_layers(x)

        return x


# =====================================================
# Load Model
# =====================================================

model = SimpleCNN().to(DEVICE)

model.load_state_dict(

    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()


# =====================================================
# Image Transform
# =====================================================

transform = transforms.Compose([

    transforms.Resize((32, 32)),

    transforms.ToTensor()
])


# =====================================================
# Drift Monitoring Trigger
# =====================================================

def trigger_monitoring():

    try:

        subprocess.run(

            [
                "python",
                "monitoring/evidently_monitor.py"
            ],

            check=True
        )

        print(
            "Drift monitoring executed."
        )

    except Exception as e:

        print(
            f"Monitoring failed: {e}"
        )


# =====================================================
# Prediction Logging
# =====================================================

def log_prediction(
    prediction,
    confidence
):

    row = pd.DataFrame([

        {
            "prediction": prediction,
            "confidence": confidence
        }
    ])

    # -----------------------------------------
    # Append Prediction
    # -----------------------------------------

    if os.path.exists(LOG_FILE):

        row.to_csv(

            LOG_FILE,

            mode="a",

            header=False,

            index=False
        )

    else:

        row.to_csv(

            LOG_FILE,

            index=False
        )

    # -----------------------------------------
    # Trigger Monitoring Every N Predictions
    # -----------------------------------------

    df = pd.read_csv(LOG_FILE)

    if len(df) % DRIFT_TRIGGER_COUNT == 0:

        trigger_monitoring()


# =====================================================
# Home Route
# =====================================================

@app.get("/")

def home():

    return {

        "message":
        "Data-Centric CV MLOps API Running"
    }


# =====================================================
# Prediction Endpoint
# =====================================================

@app.post("/predict")

async def predict(
    file: UploadFile = File(...)
):

    try:

        # -----------------------------------------
        # Read Uploaded Image
        # -----------------------------------------

        image_bytes = await file.read()

        image = Image.open(

            io.BytesIO(image_bytes)

        ).convert("RGB")


        # -----------------------------------------
        # Transform Image
        # -----------------------------------------

        image = transform(image)

        image = image.unsqueeze(0)

        image = image.to(DEVICE)


        # -----------------------------------------
        # Model Inference
        # -----------------------------------------

        with torch.no_grad():

            outputs = model(image)

            probabilities = F.softmax(
                outputs,
                dim=1
            )

            confidence, predicted = torch.max(

                probabilities,
                1
            )


        # -----------------------------------------
        # Extract Prediction
        # -----------------------------------------

        predicted_class = CLASS_NAMES[

            predicted.item()
        ]

        confidence_score = float(

            confidence.item()
        )


        # -----------------------------------------
        # Log Prediction
        # -----------------------------------------

        log_prediction(

            predicted_class,

            confidence_score
        )


        # -----------------------------------------
        # Return Response
        # -----------------------------------------

        return {

            "prediction":
            predicted_class,

            "confidence":
            confidence_score
        }

    except Exception as e:

        return {

            "error": str(e)
        }