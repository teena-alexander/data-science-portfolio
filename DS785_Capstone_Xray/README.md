# Agentic AI System for Chest X-Ray Analysis and Automated Radiology Report Generation

**DS785 Capstone Project | Teena Alexander | April 2026**

---

## Project Overview

This project builds a proof-of-concept agentic AI system that analyzes chest X-rays and automatically generates structured radiology reports. It combines a fine-tuned ResNet50 CNN classifier with a Retrieval-Augmented Generation (RAG) pipeline and an agentic confidence gate that dynamically decides whether to retrieve similar clinical reports before generating output.

---

## Dataset

**Indiana University Chest X-Ray Collection (IU X-Ray / Open-I)** — 6,461 paired image-report records split into 4,615 train / 534 validation / 1,312 test.

> Dataset not included. Download from [Open-I platform](https://openi.nlm.nih.gov/).

---

## Code Structure

Scripts must be run in order:

| Script | Description | Run On |
|--------|-------------|--------|
| `01_etl_eda.py` | XML parsing, text cleaning, data splitting, EDA visualizations | Local CPU |
| `02_model_training_COLAB.py` | ResNet50 fine-tuning, Grad-CAM, test evaluation | Google Colab T4 GPU |
| `03_rag_pipeline.py` | SentenceTransformer embeddings, FAISS index, agentic pipeline, report generation | Local CPU |

```
01_etl_eda.py → nlmcxr_cleaned_for_eda.csv
                      ↓
02_model_training_COLAB.py → resnet50_chestxray.pth
                      ↓
03_rag_pipeline.py → rag_generated_reports.txt
```

---

## How to Run

**Script 01 — Local:**
```bash
pip install pandas scikit-learn matplotlib seaborn
python 01_etl_eda.py
```

**Script 02 — Google Colab:**
1. Open [colab.research.google.com](https://colab.research.google.com)
2. Set Runtime → T4 GPU
3. Upload script and dataset, run cells top to bottom
4. Download `resnet50_chestxray.pth` when complete

> Model weights (~92MB) are not included in this repo. Run Script 02 to generate them.

**Script 03 — Local:**
```bash
pip install sentence-transformers faiss-cpu torch torchvision
python 03_rag_pipeline.py
```

---

## Results

| Metric | Value |
|--------|-------|
| Test AUC-ROC | 0.62 |
| Test Accuracy | 59% |
| Normal Recall | 0.67 |
| Abnormal Recall | 0.50 |

---

## Dependencies

PyTorch · torchvision · SentenceTransformers · FAISS · scikit-learn · pandas · matplotlib · seaborn · SQLite

---

*DS785 Capstone | University of Wisconsin*

