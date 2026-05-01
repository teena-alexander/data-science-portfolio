# -*- coding: utf-8 -*-
"""
DS785 Capstone Project: Chest X-Ray Analysis & Report Generation
Script 02 — Google Colab Version
ResNet50 Multi-Label Classifier + Grad-CAM

HOW TO USE:
1. Upload this file to Google Colab (or paste cell by cell)
2. Set Runtime → Change runtime type → T4 GPU
3. Run cells top to bottom
4. Model weights saved to your Google Drive automatically
"""

# =============================================================================
# CELL 1: CHECK GPU & MOUNT GOOGLE DRIVE
# =============================================================================
# @title Cell 1: Setup — Check GPU and Mount Google Drive
# Run this first!

import torch
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("GPU Memory:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB")
else:
    print("⚠️  No GPU detected. Go to Runtime → Change runtime type → T4 GPU")

# Mount Google Drive (your files will be saved here permanently)
from google.colab import drive
drive.mount('/content/drive')

# Create a project folder in your Drive
import os
PROJECT_DIR = "/content/drive/MyDrive/DS785_Capstone"
os.makedirs(PROJECT_DIR, exist_ok=True)
print(f"\n✅ Project folder ready: {PROJECT_DIR}")

# =============================================================================
# CELL 2: INSTALL DEPENDENCIES
# =============================================================================
# @title Cell 2: Install Dependencies

# PyTorch is pre-installed on Colab with GPU support
# Just install any missing packages

import subprocess
subprocess.run(["pip", "install", "Pillow", "scikit-learn", "seaborn", "-q"])
print("✅ Dependencies ready.")

# =============================================================================
# CELL 3: UPLOAD DATASET FILES
# =============================================================================
# @title Cell 3: Upload Your Dataset

# You have two options to get your data into Colab:
#
# OPTION A — Upload from your computer (small files only):
# from google.colab import files
# uploaded = files.upload()   # use for nlmcxr_cleaned_for_eda.csv
#
# OPTION B — Copy from Google Drive (recommended for images):
# First, zip and upload your data folder to Google Drive manually,
# then unzip here:

# Step 1: Upload nlmcxr_cleaned_for_eda.csv from your local machine
from google.colab import files
print("Please upload your nlmcxr_cleaned_for_eda.csv file:")
uploaded = files.upload()

# Step 2: Copy CSV to working directory
import shutil
for filename in uploaded.keys():
    shutil.copy(filename, f"/content/{filename}")
    print(f"  Copied {filename} to /content/")

# Step 3: Upload your image folder
# If your PNG folder is small enough (<1GB), zip it first on your PC:
#   Right-click NLMCXR_png folder → Send to → Zip
# Then upload the zip here:
print("\nNow upload your NLMCXR_png.zip image folder:")
uploaded_zip = files.upload()

# Unzip images
import zipfile
for filename in uploaded_zip.keys():
    print(f"Unzipping {filename}...")
    with zipfile.ZipFile(f"/content/{filename}", 'r') as z:
        z.extractall("/content/data/")
    print(f"  ✅ Unzipped to /content/data/")

# =============================================================================
# CELL 4: CONFIGURATION
# =============================================================================
# @title Cell 4: Configuration

import os

# Paths — all outputs saved to Google Drive
IMAGE_DIR  = "/content/data/NLMCXR_png"
CLEAN_CSV  = "/content/nlmcxr_cleaned_for_eda.csv"
MODEL_PATH = f"{PROJECT_DIR}/resnet50_chestxray.pth"   # Saved to Drive

# Training hyperparameters
# Colab T4 GPU can handle larger batches than local CPU
IMAGE_SIZE    = 224
BATCH_SIZE    = 32      # Increased from 16 — T4 GPU can handle this
NUM_EPOCHS    = 15      # More epochs since GPU is fast
LEARNING_RATE = 1e-4
NUM_CLASSES   = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device     : {DEVICE}")
print(f"Batch Size : {BATCH_SIZE}")
print(f"Epochs     : {NUM_EPOCHS}")
print(f"Model will be saved to: {MODEL_PATH}")

# =============================================================================
# CELL 5: IMPORTS
# =============================================================================
# @title Cell 5: Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

from PIL import Image
from sklearn.metrics import roc_auc_score, classification_report, roc_curve

sns.set(style="whitegrid")
print("✅ All libraries imported.")

# =============================================================================
# CELL 6: DATASET CLASS
# =============================================================================
# @title Cell 6: ChestXRayDataset Class

class ChestXRayDataset(Dataset):
    """
    PyTorch Dataset for IU Chest X-Ray images.
    Loads image from disk and returns it with its binary label.
    """
    def __init__(self, dataframe: pd.DataFrame, image_dir: str, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image_file'])

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(
            [1.0 if row['label'] == 'Abnormal' else 0.0],
            dtype=torch.float32
        )
        return image, label

print("✅ ChestXRayDataset class defined.")

# =============================================================================
# CELL 7: IMAGE TRANSFORMS
# =============================================================================
# @title Cell 7: Image Transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

print("✅ Transforms defined.")

# =============================================================================
# CELL 8: LOAD DATA
# =============================================================================
# @title Cell 8: Load Data & Build DataLoaders

df = pd.read_csv(CLEAN_CSV)
df['label'] = df['label'].astype(str)

train_df = df[df['split'] == 'train'].reset_index(drop=True)
val_df   = df[df['split'] == 'val'].reset_index(drop=True)
test_df  = df[df['split'] == 'test'].reset_index(drop=True)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

train_dataset = ChestXRayDataset(train_df, IMAGE_DIR, transform=train_transform)
val_dataset   = ChestXRayDataset(val_df,   IMAGE_DIR, transform=val_test_transform)
test_dataset  = ChestXRayDataset(test_df,  IMAGE_DIR, transform=val_test_transform)

# num_workers=2 speeds up data loading on Colab
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print("✅ DataLoaders ready.")

# =============================================================================
# CELL 9: CLASS IMBALANCE
# =============================================================================
# @title Cell 9: Compute pos_weight for Class Imbalance

n_abnormal = (train_df['label'] == 'Abnormal').sum()
n_normal   = (train_df['label'] == 'Normal').sum()
pos_weight  = torch.tensor([n_normal / n_abnormal], dtype=torch.float32).to(DEVICE)

print(f"Normal   : {n_normal}")
print(f"Abnormal : {n_abnormal}")
print(f"pos_weight (BCEWithLogitsLoss): {pos_weight.item():.4f}")

# =============================================================================
# CELL 10: BUILD MODEL
# =============================================================================
# @title Cell 10: Build ResNet50 Model

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer4 for fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace final FC layer
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(in_features, NUM_CLASSES)
)

model = model.to(DEVICE)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ ResNet50 loaded.")
print(f"   Trainable parameters: {trainable_params:,}")

# Loss, optimizer, scheduler
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=2, factor=0.5, verbose=True
)

print("   Loss: BCEWithLogitsLoss")
print("   Optimizer: Adam")
print("   Scheduler: ReduceLROnPlateau")

# =============================================================================
# CELL 11: TRAINING LOOP
# =============================================================================
# @title Cell 11: Train the Model
# ⏱️ Expected time on T4 GPU: ~10-20 minutes for 15 epochs

train_losses, val_losses = [], []
train_aucs,   val_aucs   = [], []
best_val_auc = 0.0

for epoch in range(NUM_EPOCHS):
    # --- Training ---
    model.train()
    running_loss = 0.0
    all_labels   = []
    all_probs    = []

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    epoch_train_loss = running_loss / len(train_dataset)
    train_auc = roc_auc_score(
        np.array(all_labels), np.array(all_probs)
    ) if len(set(np.array(all_labels).flatten())) > 1 else 0.5

    # --- Validation ---
    model.eval()
    val_loss   = 0.0
    val_labels = []
    val_probs  = []

    with torch.no_grad():
        for images, labels in val_loader:
            images  = images.to(DEVICE)
            labels  = labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs).cpu().numpy()
            val_probs.extend(probs)
            val_labels.extend(labels.cpu().numpy())

    epoch_val_loss = val_loss / len(val_dataset)
    val_auc = roc_auc_score(
        np.array(val_labels), np.array(val_probs)
    ) if len(set(np.array(val_labels).flatten())) > 1 else 0.5

    scheduler.step(epoch_val_loss)

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    train_aucs.append(train_auc)
    val_aucs.append(val_auc)

    print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
          f"Train Loss: {epoch_train_loss:.4f} | Train AUC: {train_auc:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f} | Val AUC: {val_auc:.4f}")

    # Save best model to Google Drive
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  ✅ Best model saved to Drive (Val AUC: {best_val_auc:.4f})")

print(f"\n🎉 Training complete! Best Val AUC: {best_val_auc:.4f}")
print(f"   Model saved to: {MODEL_PATH}")

# =============================================================================
# CELL 12: TRAINING CURVES
# =============================================================================
# @title Cell 12: Plot Training Curves

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_losses, label='Train Loss', color='royalblue')
axes[0].plot(val_losses,   label='Val Loss',   color='tomato')
axes[0].set_title('Loss Curve')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(train_aucs, label='Train AUC', color='royalblue')
axes[1].plot(val_aucs,   label='Val AUC',   color='tomato')
axes[1].set_title('AUC-ROC per Epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('AUC')
axes[1].legend()

plt.suptitle('ResNet50 Training Curves — DS785 Capstone')
plt.tight_layout()
plt.savefig(f"{PROJECT_DIR}/plot_training_curves.png", dpi=150)
plt.show()
print(f"✅ Training curves saved to Drive.")

# =============================================================================
# CELL 13: TEST SET EVALUATION
# =============================================================================
# @title Cell 13: Evaluate on Test Set

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

test_labels = []
test_probs  = []

with torch.no_grad():
    for images, labels in test_loader:
        images  = images.to(DEVICE)
        outputs = model(images)
        probs   = torch.sigmoid(outputs).cpu().numpy()
        test_probs.extend(probs)
        test_labels.extend(labels.numpy())

test_labels = np.array(test_labels).flatten()
test_probs  = np.array(test_probs).flatten()
test_preds  = (test_probs >= 0.5).astype(int)

test_auc = roc_auc_score(test_labels, test_probs)
print(f"Test AUC-ROC: {test_auc:.4f}\n")
print("Classification Report:")
print(classification_report(test_labels, test_preds,
                             target_names=['Normal', 'Abnormal']))

# ROC Curve
fpr, tpr, _ = roc_curve(test_labels, test_probs)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='royalblue', lw=2, label=f'AUC = {test_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.title('ROC Curve — Test Set')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig(f"{PROJECT_DIR}/plot_roc_curve.png", dpi=150)
plt.show()
print(f"✅ ROC curve saved to Drive.")

# =============================================================================
# CELL 14: GRAD-CAM
# =============================================================================
# @title Cell 14: Grad-CAM Heatmaps

class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[:, 0].backward()

        pooled_grads = self.gradients.mean(dim=[0, 2, 3])
        heatmap      = self.activations[0].cpu().numpy()
        for i, w in enumerate(pooled_grads.cpu().numpy()):
            heatmap[i] *= w

        heatmap = np.mean(heatmap, axis=0)
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        return heatmap


def show_gradcam(image_path, heatmap, pred_prob, true_label):
    img        = Image.open(image_path).convert("RGB").resize((224, 224))
    img_np     = np.array(img) / 255.0
    heatmap_r  = np.array(Image.fromarray(
        np.uint8(255 * heatmap)).resize((224, 224))) / 255.0
    colored    = cm.get_cmap('jet')(heatmap_r)[:, :, :3]
    overlay    = np.clip(0.6 * img_np + 0.4 * colored, 0, 1)
    pred_label = "Abnormal" if pred_prob >= 0.5 else "Normal"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title(f"Original\nTrue: {true_label}")
    axes[0].axis('off')
    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM\nPred: {pred_label} ({pred_prob:.1%})")
    axes[1].axis('off')
    plt.suptitle("Grad-CAM Explainability — ResNet50")
    plt.tight_layout()

    save_path = f"{PROJECT_DIR}/gradcam_{true_label.lower()}_{pred_label.lower()}.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  ✅ Saved: {save_path}")


# Run Grad-CAM on 3 sample test images
grad_cam   = GradCAM(model, model.layer4)
sample_df  = test_df.sample(3, random_state=42).reset_index(drop=True)

print("Generating Grad-CAM heatmaps for 3 test images...\n")
for _, row in sample_df.iterrows():
    image_path = os.path.join(IMAGE_DIR, row['image_file'])
    if not os.path.exists(image_path):
        print(f"  ⚠️  Not found: {image_path}")
        continue

    img_tensor = val_test_transform(
        Image.open(image_path).convert("RGB")
    ).unsqueeze(0).to(DEVICE)
    img_tensor.requires_grad_()

    with torch.enable_grad():
        heatmap  = grad_cam.generate(img_tensor)
        prob_out = torch.sigmoid(model(img_tensor)).item()

    show_gradcam(image_path, heatmap, prob_out, row['label'])

# =============================================================================
# CELL 15: DOWNLOAD MODEL TO LOCAL MACHINE
# =============================================================================
# @title Cell 15: Download Model Weights to Your PC

# Your model is already saved to Google Drive at:
# /content/drive/MyDrive/DS785_Capstone/resnet50_chestxray.pth
#
# To also download it directly to your PC, run:

from google.colab import files
files.download(MODEL_PATH)
print(f"✅ Model download started: {MODEL_PATH}")
print("\n🎉 Script 02 Complete!")
print(f"   All outputs saved to Google Drive: {PROJECT_DIR}")
print(f"   Files saved:")
print(f"     - resnet50_chestxray.pth  (model weights)")
print(f"     - plot_training_curves.png")
print(f"     - plot_roc_curve.png")
print(f"     - gradcam_*.png  (Grad-CAM heatmaps)")
