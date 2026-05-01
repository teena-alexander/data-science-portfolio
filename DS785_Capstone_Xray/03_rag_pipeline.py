# -*- coding: utf-8 -*-
"""
DS785 Capstone Project: Chest X-Ray Analysis & Report Generation
Script 03: Retrieval-Augmented Generation (RAG) Pipeline
Author: Teena Alexander

Pipeline Flow:
  1. Embed the IU X-Ray report corpus using SentenceTransformers
  2. Index embeddings with FAISS for fast similarity search
  3. Given a new X-ray prediction → retrieve top-k similar reports
  4. Agentic confidence gating: only retrieve if model confidence < threshold
  5. Construct a structured prompt → generate a draft radiology report
"""

# =============================================================================
# SECTION 1: IMPORT LIBRARIES
# =============================================================================
import os
import json
import sqlite3
import numpy as np
import pandas as pd
import faiss
import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image

# =============================================================================
# SECTION 2: CONFIGURATION
# =============================================================================
# Paths
CLEAN_CSV       = "nlmcxr_cleaned_for_eda.csv"
IMAGE_DIR       = "data/NLMCXR_png"
MODEL_PATH      = "resnet50_chestxray.pth"
FAISS_INDEX     = "rag_faiss.index"
CORPUS_DB       = "rag_corpus.db"
EMBEDDINGS_FILE = "rag_embeddings.npy"
CORPUS_META     = "rag_corpus_meta.json"

# RAG settings
EMBED_MODEL_NAME  = "all-MiniLM-L6-v2"   # Lightweight, fast, good for clinical text
TOP_K             = 5                      # Number of similar reports to retrieve
CONFIDENCE_THRESH = 0.70                   # Agentic gate: retrieve only if prob < threshold
IMAGE_SIZE        = 224

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =============================================================================
# SECTION 3: LOAD CORPUS
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 3: Loading report corpus...")
print("=" * 60)

df = pd.read_csv(CLEAN_CSV)

# Use training set only as the RAG corpus
# (prevents any test/val leakage into retrieval)
corpus_df = df[df['split'] == 'train'].reset_index(drop=True)

# Build combined text field for embedding
# Format: "FINDINGS: <text> IMPRESSION: <text>"
corpus_df['combined_text'] = (
    "FINDINGS: " + corpus_df['findings_clean'] +
    " IMPRESSION: " + corpus_df['impression_clean']
)

print(f"  Corpus size (train split): {len(corpus_df)} records")
print(f"  Sample corpus entry:\n  {corpus_df['combined_text'].iloc[0][:200]}...")

# =============================================================================
# SECTION 4: STORE CORPUS IN SQLITE
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 4: Storing corpus in SQLite database...")
print("=" * 60)

conn = sqlite3.connect(CORPUS_DB)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS reports (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        report_id     TEXT,
        image_file    TEXT,
        findings      TEXT,
        impression    TEXT,
        combined_text TEXT,
        label         TEXT
    )
""")
conn.commit()

# Clear and reload (ensures fresh data on each run)
cursor.execute("DELETE FROM reports")
for _, row in corpus_df.iterrows():
    cursor.execute("""
        INSERT INTO reports (report_id, image_file, findings, impression, combined_text, label)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        row['report_id'], row['image_file'],
        row['findings_clean'], row['impression_clean'],
        row['combined_text'], str(row['label'])
    ))
conn.commit()
conn.close()

print(f"  SQLite database saved → {CORPUS_DB}")
print(f"  Total records stored: {len(corpus_df)}")

# =============================================================================
# SECTION 5: EMBED CORPUS WITH SENTENCETRANSFORMERS
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 5: Embedding corpus with SentenceTransformers...")
print("=" * 60)

embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# Encode all corpus texts — batched for efficiency
print(f"  Using embedding model: {EMBED_MODEL_NAME}")
print("  Encoding corpus (this may take a few minutes)...")

corpus_texts = corpus_df['combined_text'].tolist()
embeddings   = embed_model.encode(
    corpus_texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

# Save embeddings to disk (avoid re-computing on future runs)
np.save(EMBEDDINGS_FILE, embeddings)
print(f"  Embeddings saved → {EMBEDDINGS_FILE}")
print(f"  Embedding shape: {embeddings.shape}")

# Save corpus metadata (report_id + label) for lookup after retrieval
corpus_meta = corpus_df[['report_id', 'image_file', 'label',
                          'findings_clean', 'impression_clean']].to_dict(orient='records')
with open(CORPUS_META, 'w') as f:
    json.dump(corpus_meta, f)
print(f"  Corpus metadata saved → {CORPUS_META}")

# =============================================================================
# SECTION 6: BUILD FAISS INDEX
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 6: Building FAISS index...")
print("=" * 60)

# Normalize embeddings for cosine similarity (L2 norm → inner product = cosine)
faiss.normalize_L2(embeddings)

embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)   # Inner Product (cosine after normalization)
index.add(embeddings)

faiss.write_index(index, FAISS_INDEX)
print(f"  FAISS index built → {FAISS_INDEX}")
print(f"  Indexed vectors: {index.ntotal} | Dimension: {embedding_dim}")

# =============================================================================
# SECTION 7: RETRIEVAL FUNCTION
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 7: Defining retrieval function...")
print("=" * 60)

def retrieve_similar_reports(query_text: str, top_k: int = TOP_K) -> list[dict]:
    """
    Given a query string (predicted findings description),
    retrieve the top-k most similar reports from the FAISS index.

    Returns a list of dicts with keys:
        report_id, label, findings_clean, impression_clean, similarity_score
    """
    # Load index and metadata
    index = faiss.read_index(FAISS_INDEX)
    with open(CORPUS_META, 'r') as f:
        meta = json.load(f)

    # Embed query
    query_embedding = embed_model.encode([query_text], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    # Search
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        record = meta[idx].copy()
        record['similarity_score'] = float(score)
        results.append(record)

    return results

print("  retrieve_similar_reports() defined.")

# =============================================================================
# SECTION 8: LOAD CLASSIFIER MODEL
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 8: Loading trained ResNet50 classifier...")
print("=" * 60)

def build_model() -> nn.Module:
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 1)
    )
    return model

classifier = build_model()
classifier.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
classifier = classifier.to(DEVICE)
classifier.eval()
print(f"  Model loaded from → {MODEL_PATH}")

# Image preprocessing (same as val/test transform in Script 02)
image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict_image(image_path: str) -> tuple[str, float]:
    """
    Run classifier on a single X-ray image.
    Returns predicted label and confidence probability.
    """
    img    = Image.open(image_path).convert("RGB")
    tensor = image_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = classifier(tensor)
        prob   = torch.sigmoid(output).item()

    label = "Abnormal" if prob >= 0.5 else "Normal"
    return label, prob

print("  predict_image() defined.")

# =============================================================================
# SECTION 9: AGENTIC CONFIDENCE GATING
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 9: Defining agentic decision logic...")
print("=" * 60)

def agentic_decision(prob: float, threshold: float = CONFIDENCE_THRESH) -> dict:
    """
    Agentic gate: determines whether retrieval is needed based on
    model confidence. High confidence → skip retrieval (trust model).
    Low confidence → retrieve similar reports for grounding.

    Args:
        prob      : Model output probability (0–1, where 1 = Abnormal)
        threshold : Confidence threshold below which retrieval is triggered

    Returns:
        dict with keys: label, confidence, retrieve, reason
    """
    label      = "Abnormal" if prob >= 0.5 else "Normal"
    confidence = prob if prob >= 0.5 else 1 - prob   # Distance from decision boundary

    if confidence >= threshold:
        retrieve = False
        reason   = (f"Model is confident ({confidence:.1%} confidence). "
                    f"Generating report from model predictions directly.")
    else:
        retrieve = True
        reason   = (f"Model confidence is low ({confidence:.1%}). "
                    f"Retrieving similar reports to ground the generated report.")

    return {
        "label"     : label,
        "probability": prob,
        "confidence": confidence,
        "retrieve"  : retrieve,
        "reason"    : reason,
    }

print("  agentic_decision() defined.")
print(f"  Confidence threshold: {CONFIDENCE_THRESH:.0%}")

# =============================================================================
# SECTION 10: REPORT GENERATION (TEMPLATE-BASED)
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 10: Defining report generation function...")
print("=" * 60)

def build_report_prompt(decision: dict, retrieved: list[dict]) -> str:
    """
    Construct a structured radiology report prompt from:
    - Model prediction (label + confidence)
    - Retrieved similar reports (if retrieval was triggered)

    This prompt can be passed to an LLM (e.g., via Anthropic API or HuggingFace)
    or used as a structured template report for the capstone demo.
    """
    label       = decision['label']
    confidence  = decision['confidence']
    probability = decision['probability']

    prompt = f"""
RADIOLOGY REPORT DRAFT
========================
AI Classification Result : {label}
Confidence               : {confidence:.1%}
Raw Probability (Abnormal): {probability:.4f}
Retrieval Triggered       : {'Yes' if decision['retrieve'] else 'No'}
Reason                    : {decision['reason']}
"""

    if retrieved:
        prompt += "\n--- RETRIEVED REFERENCE REPORTS ---\n"
        for i, ref in enumerate(retrieved, 1):
            prompt += f"""
Reference {i} (Similarity: {ref['similarity_score']:.4f} | Label: {ref['label']}):
  FINDINGS  : {ref['findings_clean'][:300]}...
  IMPRESSION: {ref['impression_clean'][:200]}...
"""

    prompt += """
--- GENERATED DRAFT REPORT ---
TECHNIQUE: PA and lateral chest radiograph.

FINDINGS:
"""
    if label == "Normal":
        prompt += (
            "The lungs are clear bilaterally. No focal consolidation, pleural effusion, "
            "or pneumothorax is identified. The cardiomediastinal silhouette is within "
            "normal limits. Osseous structures are intact.\n"
        )
    else:
        if retrieved:
            # Ground the report in retrieved reference findings
            ref_findings = retrieved[0]['findings_clean']
            prompt += f"[AI-assisted, grounded in similar cases]: {ref_findings[:400]}\n"
        else:
            prompt += (
                "Radiographic findings suggest an abnormality. "
                "Clinical correlation and radiologist review are recommended.\n"
            )

    prompt += "\nIMPRESSION:\n"
    if label == "Normal":
        prompt += "No acute cardiopulmonary abnormality.\n"
    else:
        if retrieved:
            ref_impression = retrieved[0]['impression_clean']
            prompt += f"[AI-assisted, grounded in similar cases]: {ref_impression[:200]}\n"
        else:
            prompt += (
                "Findings are suspicious for an acute cardiopulmonary process. "
                "Radiologist review is strongly recommended.\n"
            )

    prompt += "\n⚠️  DISCLAIMER: This is an AI-generated draft report for review purposes only.\n"
    prompt += "    Final diagnosis must be made by a qualified radiologist.\n"

    return prompt

print("  build_report_prompt() defined.")

# =============================================================================
# SECTION 11: END-TO-END PIPELINE
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 11: Defining end-to-end pipeline...")
print("=" * 60)

def run_pipeline(image_path: str, verbose: bool = True) -> dict:
    """
    Full agentic RAG pipeline for a single chest X-ray image.

    Steps:
        1. Run classifier → get prediction + confidence
        2. Apply agentic gate → decide whether to retrieve
        3. If retrieve → query FAISS for similar reports
        4. Build structured draft report
        5. Return all outputs as a dict

    Args:
        image_path : Path to the chest X-ray .png file
        verbose    : Print step-by-step output if True

    Returns:
        dict with keys: image_path, label, probability, confidence,
                        retrieve, retrieved_reports, report
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Processing: {os.path.basename(image_path)}")
        print(f"{'='*60}")

    # Step 1: Classify image
    label, prob = predict_image(image_path)
    if verbose:
        print(f"  [Step 1] Classifier → Label: {label} | Prob: {prob:.4f}")

    # Step 2: Agentic decision
    decision = agentic_decision(prob)
    if verbose:
        print(f"  [Step 2] Agentic Gate → Retrieve: {decision['retrieve']}")
        print(f"           Reason: {decision['reason']}")

    # Step 3: Retrieval (conditional)
    retrieved = []
    if decision['retrieve']:
        # Build query from predicted label for retrieval
        query_text = f"chest xray {label.lower()} findings cardiopulmonary"
        retrieved  = retrieve_similar_reports(query_text, top_k=TOP_K)
        if verbose:
            print(f"  [Step 3] Retrieved {len(retrieved)} similar reports.")
            for i, r in enumerate(retrieved, 1):
                print(f"           [{i}] {r['report_id']} | "
                      f"Label: {r['label']} | Similarity: {r['similarity_score']:.4f}")
    else:
        if verbose:
            print("  [Step 3] Retrieval skipped (high confidence).")

    # Step 4: Generate report
    report = build_report_prompt(decision, retrieved)
    if verbose:
        print("\n  [Step 4] Generated Draft Report:")
        print("  " + "-" * 50)
        print(report)

    return {
        "image_path"      : image_path,
        "label"           : label,
        "probability"     : prob,
        "confidence"      : decision['confidence'],
        "retrieve"        : decision['retrieve'],
        "retrieved_reports": retrieved,
        "report"          : report,
    }

print("  run_pipeline() defined.")

# =============================================================================
# SECTION 12: DEMO — RUN PIPELINE ON SAMPLE IMAGES
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 12: Running demo on sample test images...")
print("=" * 60)

# Load test set from cleaned CSV
df_test   = pd.read_csv(CLEAN_CSV)
df_test   = df_test[df_test['split'] == 'test'].reset_index(drop=True)
sample_df = df_test.sample(5, random_state=42).reset_index(drop=True)

results = []
for _, row in sample_df.iterrows():
    image_path = os.path.join(IMAGE_DIR, row['image_file'])
    if not os.path.exists(image_path):
        print(f"  ⚠️  Image not found: {image_path} — skipping.")
        continue

    result = run_pipeline(image_path, verbose=True)
    result['true_label'] = row['label']
    results.append(result)

# =============================================================================
# SECTION 13: SUMMARY OF PIPELINE RESULTS
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 13: Pipeline Summary")
print("=" * 60)

summary_rows = []
for r in results:
    summary_rows.append({
        "Image"          : os.path.basename(r['image_path']),
        "True Label"     : r.get('true_label', 'N/A'),
        "Predicted"      : r['label'],
        "Probability"    : f"{r['probability']:.4f}",
        "Confidence"     : f"{r['confidence']:.1%}",
        "Retrieval Used" : "Yes" if r['retrieve'] else "No",
        "Reports Retrieved": len(r['retrieved_reports']),
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

# Save results summary
summary_df.to_csv("rag_pipeline_results.csv", index=False)
print(f"\n  Results saved → rag_pipeline_results.csv")

# Save all generated reports to text file
with open("rag_generated_reports.txt", "w") as f:
    for r in results:
        f.write(f"Image: {os.path.basename(r['image_path'])}\n")
        f.write(f"True Label: {r.get('true_label', 'N/A')} | "
                f"Predicted: {r['label']} | Confidence: {r['confidence']:.1%}\n")
        f.write(r['report'])
        f.write("\n" + "=" * 60 + "\n\n")

print("  Generated reports saved → rag_generated_reports.txt")
print("\n✅ RAG Pipeline Complete!")
print(f"   FAISS index  → {FAISS_INDEX}")
print(f"   Corpus DB    → {CORPUS_DB}")
print(f"   Results CSV  → rag_pipeline_results.csv")
print(f"   Reports TXT  → rag_generated_reports.txt")
