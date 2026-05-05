import os
from torchvision.datasets import CIFAR10

OUTPUT_DIR = "data/processed"

def save_images():

    dataset = CIFAR10(
        root="data/raw",
        train=True,
        download=False
    )

    for idx, (image, label) in enumerate(dataset):

        class_dir = os.path.join(OUTPUT_DIR, str(label))
        os.makedirs(class_dir, exist_ok=True)

        image.save(
            os.path.join(class_dir, f"{idx}.png")
        )

    print("Images processed successfully")

if __name__ == "__main__":
    save_images()