import os
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#Configuration

DATA_DIR = "data/processed"
MODEL_DIR = "models"

BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(MODEL_DIR, exist_ok=True)


#Loading Dataset

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform
)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


#CNN model

class SimpleCNN(nn.Module):

    def __init__(self):

        super(SimpleCNN, self).__init__()

        self.conv_layers = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),

            nn.Linear(128, 10)
        )

    def forward(self, x):

        x = self.conv_layers(x)
        x = self.fc_layers(x)

        return x


model = SimpleCNN().to(DEVICE)

#optimization with Adam and CrossEntropyLoss

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

#Mlflow tracking

mlflow.set_experiment(
    "Data-Centric-CV"
)

with mlflow.start_run():

    mlflow.log_param(
        "batch_size",
        BATCH_SIZE
    )

    mlflow.log_param(
        "epochs",
        EPOCHS
    )

    mlflow.log_param(
        "learning_rate",
        LEARNING_RATE
    )

    #training loop

    for epoch in range(EPOCHS):

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {epoch_loss:.4f}"
        )

        mlflow.log_metric(
            "loss",
            epoch_loss,
            step=epoch
        )

    #saving model

    model_path = os.path.join(
        MODEL_DIR,
        "cnn_model.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    mlflow.log_artifact(model_path)

print("Training completed.")