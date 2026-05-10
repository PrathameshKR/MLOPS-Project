# Data-Centric CV MLOps Pipeline

An end-to-end Computer Vision MLOps project focused on data-centric machine learning practices using CIFAR-10 image classification.

This project demonstrates a complete ML pipeline including:

- Data ingestion
- Data validation
- Data versioning with DVC
- CNN model training using PyTorch
- Experiment tracking using MLflow
- Evaluation pipeline
- FastAPI inference API
- Streamlit frontend application
- Prediction logging
- Drift monitoring using Evidently AI
- Docker containerization

---

# Project Architecture

```text
User Upload
      ↓
Streamlit Frontend
      ↓
FastAPI Backend
      ↓
PyTorch CNN Model
      ↓
Prediction Logging
      ↓
Evidently Drift Monitoring
```

---

# Features

## Data-Centric Validation
- Image integrity validation
- Corruption detection
- Class distribution analysis
- Validation report generation

## Data Versioning
- Dataset tracking using DVC
- Reproducible dataset pipeline

## Model Training
- CNN-based image classification
- Training pipeline using PyTorch
- Loss tracking

## Experiment Tracking
- MLflow integration
- Parameter logging
- Metric logging
- Artifact tracking

## Evaluation Pipeline
- Accuracy calculation
- Classification report

## Inference API
- FastAPI backend
- Image upload endpoint
- Confidence score prediction

## Monitoring
- Prediction logging
- Evidently AI drift reports
- Confidence monitoring

## Dockerization
- Dockerfile for reproducible environment
- Container-ready backend

---

# Tech Stack

- Python
- PyTorch
- FastAPI
- Streamlit
- MLflow
- DVC
- Evidently AI
- Docker
- Pandas
- NumPy
- Scikit-learn

---

# Project Structure

```text
MLOPS-Project/
│
├── app/
│   ├── main.py
│   ├── frontend.py
│   └── __init__.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── models/
│   └── cnn_model.pth
│
├── monitoring/
│   ├── prediction_logs.csv
│   ├── reference_data.csv
│   ├── evidently_monitor.py
│   └── evidently_reports/
│
├── reports/
│
├── src/
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── process_data.py
│   │   └── create_metadata.py
│   │
│   ├── validation/
│   │   └── validate_data.py
│   │
│   ├── training/
│   │   └── train.py
│   │
│   └── evaluation/
│       └── evaluate.py
│
├── Dockerfile
├── requirements.txt
├── dvc.yaml
├── .gitignore
└── README.md
```

---

# Setup Instructions

## Clone Repository

```bash
git clone <repository-url>
cd MLOPS-Project
```

---

## Create Conda Environment

```bash
conda create -n mlops-cv python=3.10 -y
conda activate mlops-cv
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Pipeline

## Data Ingestion

```bash
python src/data/ingestion.py
```

---

## Process Dataset

```bash
python src/data/process_data.py
```

---

## Validate Dataset

```bash
python src/validation/validate_data.py
```

---

# DVC Data Versioning

## Initialize DVC

```bash
dvc init
```

---

## Track Dataset

```bash
dvc add data/processed
```

---

# Model Training

```bash
python src/training/train.py
```

---

# MLflow Experiment Tracking

Start MLflow UI:

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

---

# Model Evaluation

```bash
python src/evaluation/evaluate.py
```

Generated outputs:
- Classification report
- Confusion matrix
- Evaluation metrics

---

# Run FastAPI Backend

```bash
uvicorn app.main:app --reload
```

Open Swagger Docs:

```text
http://127.0.0.1:8000/docs
```

---

# Run Streamlit Frontend

```bash
streamlit run app/frontend.py
```

Open:

```text
http://localhost:8501
```

---

# Evidently Monitoring

## Generate Drift Report

```bash
python monitoring/evidently_monitor.py
```

Generated report:

```text
monitoring/evidently_reports/drift_report.html
```

---

# Dockerization

## Build Docker Image

```bash
docker build -t mlops-cv-app .
```

---

## Run Docker Container

```bash
docker run -p 8000:8000 mlops-cv-app
```

---

# Future Improvements

- Kubernetes orchestration
- CI/CD pipelines
- Automated retraining
- Cloud deployment
- Real-time monitoring dashboards
- GPU inference optimization
- Advanced drift detection

---

# Learning Outcomes

This project demonstrates practical understanding of:

- Data-centric machine learning
- Computer Vision pipelines
- MLOps lifecycle
- Model monitoring
- Experiment tracking
- Data versioning
- API development
- Drift detection
- Containerization

---

# License

This project is intended for educational and research purposes.
