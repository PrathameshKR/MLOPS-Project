import os
import torchvision
import torchvision.transforms as transforms

def download_data(data_dir="data/raw"):
    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                  download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, 
                                                download=True, transform=transform)

    print("Data downloaded and transformed successfully.")

if __name__ == "__main__":
    download_data()