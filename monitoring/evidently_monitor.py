import os
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


# ---------------------------------
# Load Data
# ---------------------------------

REFERENCE_FILE = "monitoring/reference_data.csv"

CURRENT_FILE = "monitoring/prediction_logs.csv"


reference_data = pd.read_csv(
    REFERENCE_FILE
)

current_data = pd.read_csv(
    CURRENT_FILE
)


# ---------------------------------
# Create Report
# ---------------------------------

report = Report(

    metrics=[
        DataDriftPreset()
    ]
)


report.run(

    reference_data=reference_data,

    current_data=current_data
)


# ---------------------------------
# Save Report
# ---------------------------------

os.makedirs(
    "monitoring/evidently_reports",
    exist_ok=True
)

output_path = (
    "monitoring/evidently_reports/"
    "drift_report.html"
)

report.save_html(output_path)

print(
    f"Drift report saved at: "
    f"{output_path}"
)