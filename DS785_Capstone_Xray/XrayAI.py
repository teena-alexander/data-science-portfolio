# -*- coding: utf-8 -*-
"""
DS785 Project: Chest X-ray Image-Report Preprocessing
Created on Fri Feb 27 17:54:19 2026
Author: Teena Alexander
"""
# ---------------------------
# 1. IMPORT LIBRARIES
# ---------------------------
import matplotlib.pyplot as plt
import os
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

# ---------------------------
# 2. DEFINE PATHS
# ---------------------------
REPORT_DIR = "data/NLMCXR_reports/ecgen-radiology"
IMAGE_DIR = "data/NLMCXR_png"
FINAL_CSV = "nlmcxr_preprocessed.csv"

# ---------------------------
# 3. EXTRACT XML REPORTS AND LINK IMAGES
# ---------------------------
rows = []
all_images_on_disk = set(os.listdir(IMAGE_DIR))

for file in os.listdir(REPORT_DIR):
    if not file.endswith(".xml"): continue
    
    tree = ET.parse(os.path.join(REPORT_DIR, file))
    root = tree.getroot()
    
    # Extract findings and impressions specifically
    findings = ""
    impression = ""
    image_ids = []

    # Extract Image IDs (parentImage tags)
    for abstract in root.findall(".//AbstractText"):
        label = abstract.attrib.get("Label", "")
        if label == "FINDINGS":
            findings = abstract.text.strip() if abstract.text else ""
        elif label == "IMPRESSION":
            impression = abstract.text.strip() if abstract.text else ""

    # THE LINK: Append .png and verify it exists in the folder
    for img in root.findall(".//parentImage"):
        img_id = img.attrib.get("id", "")
        if img_id:
            filename_with_ext = f"{img_id}.png"
            
            if filename_with_ext in all_images_on_disk:
                rows.append({
                    "report_id": file.replace(".xml", ""),
                    "image_file": filename_with_ext, # The "Link"
                    "findings": findings,
                    "impression": impression
                })

df = pd.DataFrame(rows)

# ---------------------------
# 4. SPLIT DATA INTO TRAIN / VAL / TEST
# ---------------------------
# Ensure patient-level leakage is prevented by splitting on unique report IDs
# split by 'report_id' so the same patient's multiple views don't cross between sets.
unique_reports = df['report_id'].unique()
train_ids, test_ids = train_test_split(unique_reports, test_size=0.2, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=0.1, random_state=42)

df['split'] = 'train'
df.loc[df['report_id'].isin(val_ids), 'split'] = 'val'
df.loc[df['report_id'].isin(test_ids), 'split'] = 'test'

# ---------------------------
# 5. EXPORT PREPROCESSED DATA
# ---------------------------
df.to_csv(FINAL_CSV, index=False)
print(f"Preprocessing Complete. Total Records: {len(df)}")
print(df['split'].value_counts())

# ---------------------------
# 6.Loadcleaned data
# ---------------------------

df = pd.read_csv("nlmcxr_preprocessed.csv")

# ---------------------------
# 7. DATA PROFILING
# ---------------------------
num_reports = df['report_id'].nunique()
total_linked_images = len(df)
images_per_report = df.groupby('report_id').size()
avg_images_per_report = images_per_report.mean()
max_images_per_report = images_per_report.max()

print("\n📊 Data Profiling Summary (after exact image matching)")
print(f"Unique reports: {num_reports}")
print(f"Total linked images: {total_linked_images}")
print(f"Average images per report: {avg_images_per_report:.2f}")
print(f"Max images in a single report: {max_images_per_report}")

#  Basic Counts
print(f"Total Image-Report Pairs: {len(df)}")
print(f"Unique Reports: {df['report_id'].nunique()}")

#  Image Distribution (How many images per patient/report?)
img_counts = df.groupby('report_id').size().value_counts()
print("\nImages per Report Distribution:")
print(img_counts)

#  Text Length Analysis (Word counts for Findings)
df['findings_word_count'] = df['findings'].fillna('').apply(lambda x: len(x.split()))
print(f"\nAverage Words per Finding: {df['findings_word_count'].mean():.2f}")

import re
from collections import Counter
# Vocabulary Check (Top 10 most common clinical words)
def get_top_words(text_series, n=10):
    all_text = " ".join(text_series.fillna('').tolist()).lower()
    words = re.findall(r'\b\w{4,}\b', all_text) # Only words 4+ letters long
    return Counter(words).most_common(n)

print("\nTop 10 Common Clinical Terms:")
for word, count in get_top_words(df['findings']):
    print(f"{word}: {count}")

# ---------------------------
# 8. VISUALIZATIONS
# ---------------------------
plt.figure(figsize=(12, 5))

# Plot 1: Distribution of Finding Word Counts
plt.figure(figsize=(8, 5))
plt.hist(df['findings_word_count'], bins=30, color='teal', edgecolor='black', alpha=0.7)
plt.axvline(df['findings_word_count'].mean(), color='red', linestyle='dashed', label=f'Mean: {df["findings_word_count"].mean():.1f}')
plt.title('Distribution of Report Lengths (Word Count)', fontsize=14)
plt.xlabel('Number of Words', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('finding_length_distribution.png')
print("[Saved Plot: finding_length_distribution.png]")

# Plot 2: Train / Val / Test Split Distribution
plt.subplot(1, 2, 2)
df['split'].value_counts().plot(kind='bar', color=['teal', 'orange', 'red'])
plt.title('Train / Val / Test Split')
plt.ylabel('Number of Images')

plt.tight_layout()
plt.show()

# Plot 3: Images per Report Distribution
# This counts how many images each report_id has, then counts the frequency of those counts.
img_dist = df.groupby('report_id').size().value_counts().sort_index()
plt.figure(figsize=(9, 6))
bars = img_dist.plot(kind='bar', color='teal', edgecolor='black', alpha=0.8)
plt.title('Distribution of Images per Clinical Report', fontsize=15, pad=15)
plt.xlabel('Number of Images per Report', fontsize=12)
plt.ylabel('Frequency (Number of Reports)', fontsize=12)
plt.xticks(rotation=0)
for i, v in enumerate(img_dist):
    plt.text(i, v + 50, str(v), ha='center', fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.show()

