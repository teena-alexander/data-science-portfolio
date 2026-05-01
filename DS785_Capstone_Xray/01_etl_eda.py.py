
# -*- coding: utf-8 -*-
"""
DS785 Capstone Project: Chest X-Ray Analysis & Report Generation
Script 01: ETL — Data Extraction, Cleaning, and Exploratory Data Analysis
Author: Teena Alexander
"""

# =============================================================================
# SECTION 1: IMPORT LIBRARIES
# =============================================================================
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

sns.set(style="whitegrid")

# =============================================================================
# SECTION 2: DEFINE PATHS
# =============================================================================
REPORT_DIR = "data/NLMCXR_reports/ecgen-radiology"
IMAGE_DIR  = "data/NLMCXR_png"
RAW_CSV    = "nlmcxr_preprocessed.csv"
CLEAN_CSV  = "nlmcxr_cleaned_for_eda.csv"

# =============================================================================
# SECTION 3: EXTRACT XML REPORTS AND LINK IMAGES
# =============================================================================
print("=" * 60)
print("SECTION 3: Extracting XML reports and linking images...")
print("=" * 60)

rows = []
all_images_on_disk = set(os.listdir(IMAGE_DIR))

for file in os.listdir(REPORT_DIR):
    if not file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(REPORT_DIR, file))
    root = tree.getroot()

    findings   = ""
    impression = ""

    # Extract FINDINGS and IMPRESSION from AbstractText tags
    for abstract in root.findall(".//AbstractText"):
        label = abstract.attrib.get("Label", "")
        if label == "FINDINGS":
            findings = abstract.text.strip() if abstract.text else ""
        elif label == "IMPRESSION":
            impression = abstract.text.strip() if abstract.text else ""

    # Link parentImage IDs to .png files on disk
    for img in root.findall(".//parentImage"):
        img_id = img.attrib.get("id", "")
        if img_id:
            filename = f"{img_id}.png"
            if filename in all_images_on_disk:
                rows.append({
                    "report_id" : file.replace(".xml", ""),
                    "image_file": filename,
                    "findings"  : findings,
                    "impression": impression,
                })

df_raw = pd.DataFrame(rows)
print(f"  Total records extracted (before cleaning): {len(df_raw)}")
print(f"  Duplicate image files: {df_raw['image_file'].duplicated().sum()}")

# =============================================================================
# SECTION 4: TEXT CLEANING
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 4: Cleaning text fields...")
print("=" * 60)

def clean_text(text: str) -> str:
    """
    Clean radiology report text:
      - Lowercase
      - Remove de-identification tokens (xxxx)
      - Remove numbered list markers (e.g. '1.' '2.' at start of items)
      - Remove special characters except periods
      - Collapse extra whitespace
    """
    text = str(text).lower()
    text = re.sub(r'\bxxxx\b', '', text)           # remove de-identification tokens
    text = re.sub(r'(?<!\d)\d+\.\s*', '', text)    # remove numbered list markers (1. 2. 3.)
    text = re.sub(r'[^a-z0-9.\s]', '', text)       # keep letters, digits, periods
    text = re.sub(r'\s+', ' ', text)               # collapse extra whitespace
    return text.strip()

df_raw['findings']   = df_raw['findings'].fillna('')
df_raw['impression'] = df_raw['impression'].fillna('')

df_raw['findings_clean']   = df_raw['findings'].apply(clean_text)
df_raw['impression_clean'] = df_raw['impression'].apply(clean_text)

# Drop records with no usable text in either field
df_clean = df_raw[
    (df_raw['findings_clean'] != '') &
    (df_raw['impression_clean'] != '')
].copy()

print(f"  Records after removing empty findings/impressions: {len(df_clean)}")

# =============================================================================
# SECTION 5: TRAIN / VAL / TEST SPLIT
# =============================================================================
# NOTE: Split is performed AFTER cleaning to ensure split ratios are accurate.
# Splitting on report_id prevents patient-level data leakage across sets.
print("\n" + "=" * 60)
print("SECTION 5: Creating train / val / test split...")
print("=" * 60)

unique_reports = df_clean['report_id'].unique()
train_ids, test_ids = train_test_split(unique_reports, test_size=0.20, random_state=42)
train_ids, val_ids  = train_test_split(train_ids,      test_size=0.10, random_state=42)

df_clean['split'] = 'train'
df_clean.loc[df_clean['report_id'].isin(val_ids),  'split'] = 'val'
df_clean.loc[df_clean['report_id'].isin(test_ids), 'split'] = 'test'

print(df_clean['split'].value_counts().to_string())

# =============================================================================
# SECTION 6: FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 6: Engineering text features...")
print("=" * 60)

# Word and sentence counts
df_clean['findings_word_count']       = df_clean['findings_clean'].apply(lambda x: len(x.split()))
df_clean['impression_word_count']     = df_clean['impression_clean'].apply(lambda x: len(x.split()))
df_clean['total_word_count']          = df_clean['findings_word_count'] + df_clean['impression_word_count']

# Sentence count: split on ". " or end-of-string period (avoids counting decimal points)
def count_sentences(text: str) -> int:
    sentences = re.split(r'(?<=[a-z])\.\s+', text)
    return max(len(sentences), 1)

df_clean['findings_sentence_count']   = df_clean['findings_clean'].apply(count_sentences)
df_clean['impression_sentence_count'] = df_clean['impression_clean'].apply(count_sentences)

# Report length categories
def report_length_category(words: int) -> str:
    if words < 20:
        return "short"
    elif words < 60:
        return "medium"
    else:
        return "long"

df_clean['report_length_category'] = df_clean['total_word_count'].apply(report_length_category)

# =============================================================================
# SECTION 7: PROVISIONAL LABELING — NORMAL vs ABNORMAL
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 7: Applying Normal / Abnormal labels...")
print("=" * 60)

NORMAL_PHRASES = [
    "no acute cardiopulmonary abnormality",
    "no acute cardiopulmonary findings",
    "no acute cardiopulmonary abnormalities",
    "no active disease",
    "no acute disease",
    "no acute cardiopulmonary disease",
    "no acute cardiopulmonary process",
    "no evidence of active disease",
    "negative for acute abnormality",
    "normal chest",
    "no acute findings",
    "no acute pulmonary abnormality",
]

def label_normal_abnormal(impression: str) -> str:
    impression_lower = impression.lower()
    for phrase in NORMAL_PHRASES:
        if phrase in impression_lower:
            return "Normal"
    return "Abnormal"

df_clean['label'] = df_clean['impression_clean'].apply(label_normal_abnormal).astype('category')

label_counts = df_clean['label'].value_counts()
print(label_counts.to_string())

# Class imbalance ratio
abnormal = label_counts.get('Abnormal', 0)
normal   = label_counts.get('Normal', 0)
ratio    = abnormal / normal if normal > 0 else float('inf')
print(f"\n  Class imbalance ratio (Abnormal:Normal) = {ratio:.2f}:1")
print(f"  NOTE: pos_weight for BCEWithLogitsLoss should be ~{ratio:.2f}")

# =============================================================================
# SECTION 8: SAVE CLEANED DATASET
# =============================================================================
df_clean.to_csv(CLEAN_CSV, index=False)
print(f"\n  Cleaned dataset saved → {CLEAN_CSV}")
print(f"  Final shape: {df_clean.shape}")

# =============================================================================
# SECTION 9: DATA PROFILING SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 9: Data Profiling Summary")
print("=" * 60)

print(f"\n  Total records      : {len(df_clean)}")
print(f"  Unique reports     : {df_clean['report_id'].nunique()}")
print(f"  Unique images      : {df_clean['image_file'].nunique()}")
print(f"  Missing values     :\n{df_clean.isnull().sum().to_string()}")

print("\n  Numeric summary:")
numeric_cols = [
    'findings_word_count', 'impression_word_count',
    'total_word_count', 'findings_sentence_count', 'impression_sentence_count'
]
print(df_clean[numeric_cols].describe().round(2).to_string())

print(f"\n  Avg words — findings  : {df_clean['findings_word_count'].mean():.2f}")
print(f"  Avg words — impression: {df_clean['impression_word_count'].mean():.2f}")
print(f"  Avg words — total     : {df_clean['total_word_count'].mean():.2f}")

# Images per report
img_per_report = df_clean.groupby('report_id').size()
print(f"\n  Images per report (mean): {img_per_report.mean():.2f}")
print(f"  Images per report (max) : {img_per_report.max()}")

# Top clinical terms
def top_words(series, n=10):
    all_text = " ".join(series.tolist())
    words = re.findall(r'\b\w{4,}\b', all_text)
    return Counter(words).most_common(n)

print("\n  Top 10 clinical terms — Normal findings:")
for word, count in top_words(df_clean[df_clean['label'] == 'Normal']['findings_clean']):
    print(f"    {word}: {count}")

print("\n  Top 10 clinical terms — Abnormal findings:")
for word, count in top_words(df_clean[df_clean['label'] == 'Abnormal']['findings_clean']):
    print(f"    {word}: {count}")

# Outlier reports (top 1%)
outliers = df_clean[df_clean['total_word_count'] > df_clean['total_word_count'].quantile(0.99)]
print(f"\n  Outlier reports (top 1% by word count): {len(outliers)}")
print(outliers[['report_id', 'total_word_count']].to_string(index=False))

# =============================================================================
# SECTION 10: EXPLORATORY DATA ANALYSIS — VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 10: Generating EDA plots...")
print("=" * 60)

# --- 10.1 Train / Val / Test Split ---
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='split', data=df_clean,
                   order=['train', 'val', 'test'], palette='Set2')
for container in ax.containers:
    ax.bar_label(container)
plt.title('Train / Val / Test Split Distribution')
plt.ylabel('Number of Records')
plt.xlabel('Split')
plt.tight_layout()
plt.savefig("plot_01_split_distribution.png", dpi=150)
plt.show()

# --- 10.2 Train / Val / Test Pie Chart ---
split_counts = df_clean['split'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(
    [split_counts.get('train', 0), split_counts.get('val', 0), split_counts.get('test', 0)],
    labels=['Train', 'Validation', 'Test'],
    autopct='%1.1f%%',
    startangle=140,
    colors=['#66c2a5', '#fc8d62', '#8da0cb'],
    explode=(0.05, 0.05, 0.05)
)
plt.title('Train / Validation / Test Split')
plt.tight_layout()
plt.savefig("plot_02_split_pie.png", dpi=150)
plt.show()

# --- 10.3 Normal vs Abnormal Label Distribution ---
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='label', data=df_clean, palette='pastel')
for container in ax.containers:
    ax.bar_label(container)
plt.title('Normal vs Abnormal Report Distribution')
plt.ylabel('Number of Records')
plt.xlabel('Label')
plt.tight_layout()
plt.savefig("plot_03_label_distribution.png", dpi=150)
plt.show()


# --- 10.4 Report Length Category Distribution ---
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='report_length_category', data=df_clean,
                   order=['short', 'medium', 'long'], palette='Set2')
for container in ax.containers:
    ax.bar_label(container)
plt.title('Report Length Categories')
plt.xlabel('Category')
plt.ylabel('Number of Records')
plt.tight_layout()
plt.savefig("plot_04_length_categories.png", dpi=150)
plt.show()

# --- 10.5 Report Length by Label ---
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='report_length_category', hue='label', data=df_clean,
                   order=['short', 'medium', 'long'], palette='Set2')
for container in ax.containers:
    ax.bar_label(container)
plt.title('Report Length Category by Label')
plt.ylabel('Number of Records')
plt.xlabel('Report Length Category')
plt.tight_layout()
plt.savefig("plot_05_length_by_label.png", dpi=150)
plt.show()

# --- 10.6 Total Word Count Distribution by Label ---
plt.figure(figsize=(10, 5))
plt.hist(df_clean[df_clean['label'] == 'Normal']['total_word_count'],
         bins=30, color='teal', alpha=0.7, label='Normal')
plt.hist(df_clean[df_clean['label'] == 'Abnormal']['total_word_count'],
         bins=30, color='orange', alpha=0.7, label='Abnormal')
mean_wc = df_clean['total_word_count'].mean()
plt.axvline(mean_wc, color='red', linestyle='dashed', label=f'Mean: {mean_wc:.1f} words')
plt.title('Total Word Count Distribution by Label')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig("plot_06_wordcount_distribution.png", dpi=150)
plt.show()

# --- 10.7 Images per Report ---
img_counts_series = df_clean.groupby('report_id').size()
plt.figure(figsize=(6, 4))
sns.countplot(x=img_counts_series, palette='Set2')
plt.title('Number of Images per Report')
plt.xlabel('Images per Report')
plt.ylabel('Number of Reports')
plt.tight_layout()
plt.savefig("plot_07_images_per_report.png", dpi=150)
plt.show()

# --- 10.8 Correlation Heatmap ---
corr_matrix = df_clean[numeric_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numeric Features')
plt.tight_layout()
plt.savefig("plot_08_correlation_heatmap.png", dpi=150)
plt.show()

# --- 10.9 Boxplot — Outlier Detection ---
plt.figure(figsize=(12, 5))
sns.boxplot(data=df_clean[numeric_cols], palette='Set3')
plt.title('Boxplot of Word and Sentence Counts (Outlier Detection)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plot_09_boxplot_outliers.png", dpi=150)
plt.show()

print("\n✅ ETL & EDA Complete! All plots saved as PNG files.")
print(f"   Final dataset: {CLEAN_CSV} | Shape: {df_clean.shape}")