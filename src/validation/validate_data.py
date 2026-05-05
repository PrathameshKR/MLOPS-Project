import os
from PIL import Image
from collections import defaultdict

DATA_DIR = "data/processed"
REPORT_FILE = "reports/validation_report.txt"


def validate_images():

    issues = []
    image_count = 0
    class_counts = defaultdict(int)

    # Create reports directory if not exists
    os.makedirs("reports", exist_ok=True)

    # Traverse dataset
    for class_name in os.listdir(DATA_DIR):

        class_path = os.path.join(DATA_DIR, class_name)

        # Skip non-folder files
        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):

            file_path = os.path.join(class_path, file)

            try:

                with Image.open(file_path) as img:

                    width, height = img.size

                    # Validate image dimensions
                    if width <= 0 or height <= 0:

                        issues.append(
                            f"Invalid dimensions: {file_path}"
                        )

                    # Count valid image
                    image_count += 1
                    class_counts[class_name] += 1

            except Exception as e:

                issues.append(
                    f"Corrupted file: {file_path} | Error: {e}"
                )

    # Generate report
    report_lines = []

    report_lines.append("===== VALIDATION REPORT =====\n\n")

    report_lines.append(
        f"Total Images Checked: {image_count}\n\n"
    )

    report_lines.append("Class Distribution:\n")

    for class_name, count in sorted(class_counts.items()):

        report_lines.append(
            f"Class {class_name}: {count}\n"
        )

    # Validation status
    if len(issues) == 0:

        report_lines.append(
            "\nNo issues found\n"
        )

    else:

        report_lines.append("\nIssues Found:\n")

        for issue in issues:

            report_lines.append(f"{issue}\n")

    # Save report
    with open(REPORT_FILE, "w") as f:

        f.writelines(report_lines)

    print("Validation report generated successfully!")
    print(f"Report saved at: {REPORT_FILE}")


if __name__ == "__main__":

    validate_images()