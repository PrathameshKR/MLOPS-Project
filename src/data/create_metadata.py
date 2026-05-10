import os
import pandas as pd
from PIL import Image

DATA_DIR = "data/processed"

metadata = []

for class_name in os.listdir(DATA_DIR):

    class_path = os.path.join(DATA_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    for file in os.listdir(class_path):

        file_path = os.path.join(class_path, file)

        try:

            with Image.open(file_path) as img:

                width, height = img.size

                metadata.append({

                    "file_path": file_path,
                    "label": class_name,
                    "width": width,
                    "height": height

                })

        except Exception as e:

            print(f"Error reading {file_path}: {e}")

df = pd.DataFrame(metadata)

os.makedirs("data/metadata", exist_ok=True)

output_path = "data/metadata/image_metadata.csv"

df.to_csv(output_path, index=False)

print(f"Metadata saved at: {output_path}")