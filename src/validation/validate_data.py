import os
from PIL import Image

DATA_DIR = "data/processed"

def validate_images():

    issues = []
    image_count = 0

    for root, dirs, files in os.walk(DATA_DIR):

        for file in files:

            file_path = os.path.join(root, file)

            try:

                with Image.open(file_path) as img:

                    width, height = img.size

                    if width <= 0 or height <= 0:
                        issues.append(f"Invalid dimensions: {file_path}")

                    image_count += 1

            except Exception as e:

                issues.append(
                    f"Corrupted file: {file_path} | Error: {e}"
                )

    print(f"\nTotal Images Checked: {image_count}")

    if len(issues) == 0:
        print("No issues found ✅")

    else:
        print("\nIssues Found:")

        for issue in issues:
            print(issue)

if __name__ == "__main__":
    validate_images()