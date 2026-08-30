#%%
import torch._dynamo
torch._dynamo.config.suppress_errors = True


import os
import copy
import csv
import json
import random
import time
import xml.sax.saxutils as saxutils
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import albumentations as A
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image as RLImage, Table, TableStyle,
)

# ============================================================
# CONFIGURATION - edit these values to match your setup
# ============================================================

DATASET_ROOT = r"C:\Users\Admin\Desktop\Garbage\Data"
DATA_DIR = os.path.join(DATASET_ROOT, "original")   # must contain ONLY clean, non-augmented images

OUTPUT_DIR = "./training_outputs"
CV_SUBDIR = os.path.join(OUTPUT_DIR, "cross_validation")

MODEL_NAME = "convnext_tiny"   # or "convnext_small"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4                 # set to 0 on Windows if you hit DataLoader issues

# ---- Fixed train/val/test split ratio ----
TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10

# ---- K-Fold cross-validation on the (train+val) pool ----
CV_FOLDS = 5
CV_MAX_EPOCHS = 20               # safety ceiling per fold; early stopping usually stops sooner
CV_WARMUP_EPOCHS = 2
CV_EARLY_STOPPING_PATIENCE = 4

# ---- Final training with early stopping ----
FINAL_MAX_EPOCHS = 20            # safety ceiling; early stopping decides the real stop point
FINAL_WARMUP_EPOCHS = 3
FINAL_EARLY_STOPPING_PATIENCE = 4
FINAL_HOLD_OUT_VAL_FRACTION = 0.10  # slice of the trainval pool reserved only to monitor early stopping

EARLY_STOPPING_MIN_DELTA = 1e-4   # minimum improvement in val LOSS to reset patience (best checkpoint selection)

# ---- Differential learning rates ----
BACKBONE_LR = 1e-5   # pretrained ConvNeXt feature extractor (small LR to preserve ImageNet features)
HEAD_LR = 3e-4        # new classifier head (higher LR to adapt quickly)
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1

# ---- Hyperparameter tuning (LR / Weight Decay / Label Smoothing only) ----
# Reuses the existing 5-Fold CV implementation below, just with a smaller
# epoch ceiling/patience for speed. "Learning rate" here means HEAD_LR;
# BACKBONE_LR is scaled with it so the original differential-LR ratio
# (backbone always much smaller than head) is preserved unchanged.
BACKBONE_HEAD_LR_RATIO = BACKBONE_LR / HEAD_LR
HPT_MAX_EPOCHS = 5
HPT_EARLY_STOPPING_PATIENCE = 2
HYPERPARAM_SEARCH_SPACE = [
    {"head_lr": 3e-4, "weight_decay": 0.05, "label_smoothing": 0.1},
    {"head_lr": 1e-4, "weight_decay": 0.01, "label_smoothing": 0.05},
    {"head_lr": 5e-4, "weight_decay": 0.10, "label_smoothing": 0.10},
    {"head_lr": 1e-4, "weight_decay": 0.05, "label_smoothing": 0.00},
]

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# AMP (Automatic Mixed Precision) + torch.compile helpers
# ============================================================

def create_grad_scaler(device):
    """Creates a GradScaler enabled only on CUDA; a disabled GradScaler is
    a safe no-op on CPU (scale()/step()/update() just pass through)."""
    enabled = (device.type == "cuda")
    try:
        # Modern unified API (PyTorch >= 2.0 recommended form)
        return torch.amp.GradScaler(device.type if enabled else "cpu", enabled=enabled)
    except (TypeError, AttributeError):
        # Fallback for older PyTorch versions
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device):
    """Returns the recommended autocast context manager for the installed
    PyTorch version. Enabled only on CUDA; a strict no-op (FP32) on CPU."""
    try:
        return torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"))
    except (TypeError, AttributeError):
        # Fallback for older PyTorch versions
        if device.type == "cuda":
            return torch.cuda.amp.autocast(enabled=True)
        return torch.cuda.amp.autocast(enabled=False)


def maybe_compile_model(model):
    """Compiles the model once with torch.compile() if the installed
    PyTorch version supports it, leaving the architecture and behavior
    exactly the same. Silently skips compilation (falls back to the
    eager module) if torch.compile is unavailable or fails to engage."""
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        print("  torch.compile not available in this PyTorch version - skipping (no effect on results).")
        return model
    try:
        compiled_model = compile_fn(model)
        print("  torch.compile() applied to the model.")
        return compiled_model
    except Exception as e:
        print(f"  torch.compile() unavailable/failed ({e}) - continuing without compilation.")
        return model


# ============================================================
# TRANSFORMS
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard torchvision preprocessing, applied to EVERY image (real or synthetic)
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Used for validation/test AND for single-image inference in predict_image.py
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

balancing_augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=25, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
    A.GaussNoise(var_limit=(10.0, 40.0), p=0.2),
    A.Blur(blur_limit=3, p=0.15),
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05), shear=(-5, 5), p=0.3),
])


class TransformSubset(Dataset):
    """Wraps a Subset of an ImageFolder so val/test can use a plain,
    non-augmenting transform (always the same underlying real images)."""

    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class BalancedTrainDataset(Dataset):

    def __init__(self, base_dataset, indices, targets, seed=RANDOM_SEED):
        self.base_dataset = base_dataset
        self.rng = random.Random(seed)

        class_to_indices = defaultdict(list)
        for idx in indices:
            class_to_indices[int(targets[idx])].append(int(idx))

        self.counts_before = {cls: len(idxs) for cls, idxs in class_to_indices.items()}
        self.target_count = max(self.counts_before.values()) if self.counts_before else 0

        self.samples = []  # (path, label, is_synthetic)
        for cls, idxs in class_to_indices.items():
            for idx in idxs:
                path, label = self.base_dataset.samples[idx]
                self.samples.append((path, label, False))
            needed = self.target_count - len(idxs)
            for _ in range(needed):
                src_idx = self.rng.choice(idxs)
                path, label = self.base_dataset.samples[src_idx]
                self.samples.append((path, label, True))

        self.counts_after = dict(Counter(label for _, label, _ in self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, is_synthetic = self.samples[index]
        image = Image.open(path).convert("RGB")

        if is_synthetic:
            image_np = np.array(image)
            augmented_np = balancing_augmentation_pipeline(image=image_np)["image"]
            image = Image.fromarray(augmented_np)

        image_tensor = train_transform(image)
        return image_tensor, label


def build_balanced_train_loader(base_dataset, indices, targets, batch_size, num_workers, seed=RANDOM_SEED):
    dataset = BalancedTrainDataset(base_dataset, indices, targets, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=num_workers, pin_memory=True, drop_last=True)
    return loader, dataset.counts_before, dataset.counts_after, dataset.target_count


def build_eval_loader(base_dataset, indices, batch_size, num_workers):
    subset = TransformSubset(Subset(base_dataset, indices), eval_transform)
    return DataLoader(subset, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)


def print_class_count_table(counts, class_names, title):
    print(f"\n{title}")
    for cls_idx, name in enumerate(class_names):
        print(f"  {name:<20} {counts.get(cls_idx, 0)}")


# ============================================================
# DATA LOADING + DYNAMIC CLASS DISCOVERY
# ============================================================

def load_full_dataset(data_dir):
    base_dataset = ImageFolder(data_dir, transform=None)
    class_names = base_dataset.classes            # discovered dynamically, never hard-coded
    num_classes = len(class_names)                 # discovered dynamically
    targets = np.array(base_dataset.targets)

    print(f"Discovered {num_classes} classes automatically: {class_names}")
    print(f"Total images found in '{data_dir}': {len(base_dataset)}")
    return base_dataset, class_names, num_classes, targets


def split_dataset_stratified(all_idx, targets, train_ratio, val_ratio, test_ratio, seed):
    """
    A single stratified split of the CLEAN dataset (no augmented images
    exist yet at this point) into train/val/test using the fixed ratio.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    y = targets[all_idx]
    train_idx, temp_idx = train_test_split(all_idx, train_size=train_ratio, stratify=y, random_state=seed)

    val_share_of_temp = val_ratio / (val_ratio + test_ratio)
    temp_y = targets[temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, train_size=val_share_of_temp, stratify=temp_y, random_state=seed)

    return train_idx, val_idx, test_idx


# ============================================================
# MODEL: ConvNeXt with a dynamically-sized head
# ============================================================

def build_model(model_name: str, num_classes: int):
    if model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
    elif model_name == "convnext_small":
        weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        model = models.convnext_small(weights=weights)
    else:
        raise ValueError(f"Unsupported MODEL_NAME: {model_name}")

    # classifier = [0] LayerNorm2d, [1] Flatten, [2] Linear(in_features, 1000)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def build_differential_optimizer(model, backbone_lr, head_lr, weight_decay):
    """AdamW with two param groups: a tiny LR for the pretrained backbone,
    a much higher LR for the freshly-initialized classifier head."""
    param_groups = [
        {"params": model.features.parameters(), "lr": backbone_lr, "name": "backbone"},
        {"params": model.classifier.parameters(), "lr": head_lr, "name": "classifier_head"},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def build_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """Linear warm-up then cosine annealing, applied as a relative multiplier
    to each param group so the differential-LR ratio is preserved."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = max(total_epochs * steps_per_epoch, warmup_steps + 1)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(progress, 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# TRAIN / VALIDATION LOOP (with early stopping)
# ============================================================

def run_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, train: bool, scaler=None):
    model.train() if train else model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    phase_name = "Train" if train else "Val  "
    progress_bar = tqdm(dataloader, desc=phase_name, leave=False)

    with torch.set_grad_enabled(train):
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            with autocast_context(device):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total_samples += batch_size

            progress_bar.set_postfix(
                loss=f"{running_loss / total_samples:.4f}",
                acc=f"{running_correct / total_samples:.4f}",
            )

    return running_loss / total_samples, running_correct / total_samples


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                 max_epochs, checkpoint_path, patience, min_delta=1e-4,
                 class_names=None, model_name=None, verbose=True):
   
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")   # now the metric that decides the best checkpoint/early stopping
    best_val_acc = 0.0             # kept for reporting only (value at the best-val-loss epoch)
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    stopped_epoch = max_epochs

    # AMP GradScaler for this training run (enabled on CUDA, disabled/no-op on CPU)
    scaler = create_grad_scaler(device)

    for epoch in range(1, max_epochs + 1):
        start_time = time.time()
        if verbose:
            print(f"\nEpoch {epoch}/{max_epochs}")

        train_loss, train_acc = run_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, train=True, scaler=scaler
        )
        val_loss, val_acc = run_one_epoch(
            model, val_loader, criterion, optimizer=None, scheduler=None, device=device, train=False, scaler=scaler
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start_time
        if verbose:
            print(f"  Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}  ||  "
                  f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}  ({elapsed:.1f}s)")

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_val_acc = val_acc   # reporting only - recorded at the best-val-loss epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            checkpoint = {
                "epoch": epoch, "model_state_dict": best_model_weights,
                "val_acc": val_acc, "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                # Extra state below makes the checkpoint resumable (Improvement 7)
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            }
            if scaler is not None:
                checkpoint["scaler_state_dict"] = scaler.state_dict()
            if class_names is not None:
                checkpoint["class_names"] = class_names
            if model_name is not None:
                checkpoint["model_name"] = model_name
            torch.save(checkpoint, checkpoint_path)

            if verbose:
                print(f"  -> New best model saved (val_loss={val_loss:.4f}, val_acc={val_acc:.4f}) "
                      f"-> {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if verbose:
                print(f"  No improvement for {epochs_without_improvement}/{patience} epoch(s).")

        if epochs_without_improvement >= patience:
            stopped_epoch = epoch
            if verbose:
                print(f"\n*** EARLY STOPPING triggered at epoch {epoch} "
                      f"(no improvement in {patience} consecutive epochs). ***")
            break

    model.load_state_dict(best_model_weights)
    return model, history, best_val_acc, best_val_loss, stopped_epoch


# ============================================================
# METRICS HELPERS
# ============================================================

@torch.no_grad()
def collect_predictions(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


@torch.no_grad()
def collect_predictions_and_probs(model, dataloader, device):

    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.append(probs)
    return np.array(all_labels), np.array(all_preds), np.concatenate(all_probs, axis=0)


def compute_summary_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy,
        "precision_macro": p_macro, "recall_macro": r_macro, "f1_macro": f1_macro,
        "precision_weighted": p_weighted, "recall_weighted": r_weighted, "f1_weighted": f1_weighted,
    }


# ============================================================
# STEP 1: K-FOLD CROSS-VALIDATION
# ============================================================

def run_cross_validation(base_dataset, trainval_idx, targets, num_classes, class_names,
                          head_lr=HEAD_LR, backbone_lr=BACKBONE_LR, weight_decay=WEIGHT_DECAY,
                          label_smoothing=LABEL_SMOOTHING, max_epochs=CV_MAX_EPOCHS,
                          warmup_epochs=CV_WARMUP_EPOCHS, patience=CV_EARLY_STOPPING_PATIENCE,
                          checkpoint_dir=CV_SUBDIR, verbose=True):
    os.makedirs(checkpoint_dir, exist_ok=True)
    if verbose:
        print("\n" + "=" * 70)
        print(f"STEP 1/3: {CV_FOLDS}-FOLD STRATIFIED CROSS-VALIDATION "
              f"(early stopping per fold, patience={patience})")
        print("=" * 70)

    y = targets[trainval_idx]
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []
    for fold_i, (train_pos, val_pos) in enumerate(skf.split(trainval_idx, y), start=1):
        if verbose:
            print(f"\n--- Fold {fold_i}/{CV_FOLDS} ---")

        fold_train_idx = trainval_idx[train_pos]
        fold_val_idx = trainval_idx[val_pos]   # originals only, never augmented - leakage-safe

        train_loader, counts_before, counts_after, target_count = build_balanced_train_loader(
            base_dataset, fold_train_idx, targets, BATCH_SIZE, NUM_WORKERS, seed=RANDOM_SEED + fold_i
        )
        val_loader = build_eval_loader(base_dataset, fold_val_idx, BATCH_SIZE, NUM_WORKERS)

        if verbose:
            print_class_count_table(counts_before, class_names, f"Fold {fold_i} - train class counts BEFORE balancing:")
            print(f"  Target count per class (= largest class in this fold): {target_count}")
            print_class_count_table(counts_after, class_names, f"Fold {fold_i} - train class counts AFTER balancing:")
            print(f"  Fold val (originals only, never augmented): {len(fold_val_idx)}")

        set_seed(RANDOM_SEED + fold_i)
        model = build_model(MODEL_NAME, num_classes).to(DEVICE)
        model = maybe_compile_model(model)   # no-op fallback if torch.compile is unsupported
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = build_differential_optimizer(model, backbone_lr, head_lr, weight_decay)
        scheduler = build_scheduler(optimizer, warmup_epochs, max_epochs, len(train_loader))

        checkpoint_path = os.path.join(checkpoint_dir, f"fold_{fold_i}.pth")
        model, history, best_val_acc, best_val_loss, stopped_epoch = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, DEVICE,
            max_epochs, checkpoint_path, patience=patience,
            min_delta=EARLY_STOPPING_MIN_DELTA, class_names=class_names, model_name=MODEL_NAME, verbose=verbose,
        )

        fold_labels, fold_preds = collect_predictions(model, val_loader, DEVICE)
        fold_metrics = compute_summary_metrics(fold_labels, fold_preds)

        fold_results.append({"fold": fold_i, "stopped_epoch": stopped_epoch, "val_loss": best_val_loss, **fold_metrics})
        if verbose:
            print(f"  Fold {fold_i} result -> Accuracy: {fold_metrics['accuracy']:.4f} | "
                  f"F1(weighted): {fold_metrics['f1_weighted']:.4f} | val_loss: {best_val_loss:.4f} | "
                  f"stopped at epoch {stopped_epoch}")

    return fold_results


def print_and_save_cv_results(fold_results, output_dir):
    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["f1_weighted"] for r in fold_results]
    f1m = [r["f1_macro"] for r in fold_results]
    mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
    mean_f1w, std_f1w = float(np.mean(f1s)), float(np.std(f1s))
    mean_f1m, std_f1m = float(np.mean(f1m)), float(np.std(f1m))

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)
    print(f"{'Fold':<8}{'Accuracy':<12}{'F1-macro':<12}{'F1-weighted':<14}{'Stopped@epoch':<14}")
    for r in fold_results:
        print(f"{r['fold']:<8}{r['accuracy']:<12.4f}{r['f1_macro']:<12.4f}"
              f"{r['f1_weighted']:<14.4f}{r['stopped_epoch']:<14}")
    print("-" * 60)
    print(f"Mean Accuracy     : {mean_acc:.4f}  (+/- {std_acc:.4f})")
    print(f"Mean F1 (macro)   : {mean_f1m:.4f}  (+/- {std_f1m:.4f})")
    print(f"Mean F1 (weighted): {mean_f1w:.4f}  (+/- {std_f1w:.4f})")

    csv_path = os.path.join(output_dir, "cross_validation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "accuracy", "f1_macro", "f1_weighted", "stopped_epoch"])
        for r in fold_results:
            writer.writerow([r["fold"], r["accuracy"], r["f1_macro"], r["f1_weighted"], r["stopped_epoch"]])
        writer.writerow(["mean", mean_acc, mean_f1m, mean_f1w, ""])
        writer.writerow(["std", std_acc, std_f1m, std_f1w, ""])
    print(f"\nSaved cross-validation table -> {csv_path}")

    folds = [r["fold"] for r in fold_results]
    x = np.arange(len(folds))
    width = 0.35
    plt.figure(figsize=(9, 5.5))
    plt.bar(x - width / 2, accs, width, label="Accuracy", color="mediumseagreen")
    plt.bar(x + width / 2, f1s, width, label="F1-score (weighted)", color="salmon")
    plt.axhline(mean_acc, color="mediumseagreen", linestyle="--", linewidth=1,
                label=f"Mean Accuracy = {mean_acc:.3f}")
    plt.axhline(mean_f1w, color="salmon", linestyle="--", linewidth=1,
                label=f"Mean F1-weighted = {mean_f1w:.3f}")
    plt.xticks(x, [f"Fold {f}" for f in folds])
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title(f"{len(folds)}-Fold Cross-Validation Results")
    plt.legend(fontsize=8)
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "cross_validation_results.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"Saved cross-validation chart -> {chart_path}")

    return {"mean_accuracy": mean_acc, "std_accuracy": std_acc,
            "mean_f1_macro": mean_f1m, "std_f1_macro": std_f1m,
            "mean_f1_weighted": mean_f1w, "std_f1_weighted": std_f1w}

#%%
# ============================================================
# STEP 2: FINAL TRAINING WITH EARLY STOPPING
# ============================================================

def run_hyperparameter_search(base_dataset, train_idx, val_idx, targets, num_classes, class_names):
    print("\n" + "=" * 70)
    print(f"STEP 0/3: HYPERPARAMETER TUNING on a FIXED Train/Val split "
          f"(max {HPT_MAX_EPOCHS} epochs, patience={HPT_EARLY_STOPPING_PATIENCE})")
    print("=" * 70)

    search_dir = os.path.join(OUTPUT_DIR, "hyperparameter_tuning")
    os.makedirs(search_dir, exist_ok=True)

    # Built ONCE and reused for every candidate - fixed split.
    train_loader, counts_before, counts_after, target_count = build_balanced_train_loader(
        base_dataset, train_idx, targets, BATCH_SIZE, NUM_WORKERS, seed=RANDOM_SEED
    )
    val_loader = build_eval_loader(base_dataset, val_idx, BATCH_SIZE, NUM_WORKERS)

    print_class_count_table(counts_before, class_names, "HPT - train class counts BEFORE balancing:")
    print(f"  Target count per class (= largest class): {target_count}")
    print_class_count_table(counts_after, class_names, "HPT - train class counts AFTER balancing:")
    print(f"  Fixed validation set (originals only, never augmented): {len(val_idx)}")

    candidate_scores = []
    for cand_i, candidate in enumerate(HYPERPARAM_SEARCH_SPACE, start=1):
        head_lr = candidate["head_lr"]
        weight_decay = candidate["weight_decay"]
        label_smoothing = candidate["label_smoothing"]
        backbone_lr = head_lr * BACKBONE_HEAD_LR_RATIO   # preserve original differential-LR ratio

        print(f"\n--- Candidate {cand_i}/{len(HYPERPARAM_SEARCH_SPACE)}: "
              f"lr={head_lr}, weight_decay={weight_decay}, label_smoothing={label_smoothing} ---")

        set_seed(RANDOM_SEED)
        model = build_model(MODEL_NAME, num_classes).to(DEVICE)
        model = maybe_compile_model(model)   # no-op fallback if torch.compile is unsupported
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = build_differential_optimizer(model, backbone_lr, head_lr, weight_decay)
        scheduler = build_scheduler(optimizer, CV_WARMUP_EPOCHS, HPT_MAX_EPOCHS, len(train_loader))

        checkpoint_path = os.path.join(search_dir, f"candidate_{cand_i}.pth")
        _, _, _, best_val_loss, stopped_epoch = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, DEVICE,
            HPT_MAX_EPOCHS, checkpoint_path, patience=HPT_EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA, class_names=class_names, model_name=MODEL_NAME, verbose=False,
        )
        print(f"  Candidate {cand_i} validation loss: {best_val_loss:.4f} (stopped at epoch {stopped_epoch})")

        candidate_scores.append({
            "head_lr": head_lr, "backbone_lr": backbone_lr,
            "weight_decay": weight_decay, "label_smoothing": label_smoothing,
            "val_loss": best_val_loss, "stopped_epoch": stopped_epoch,
        })

    best = min(candidate_scores, key=lambda r: r["val_loss"])

    results_path = os.path.join(search_dir, "hyperparameter_search_results.json")
    with open(results_path, "w") as f:
        json.dump({"candidates": candidate_scores, "selected": best}, f, indent=2)
    print(f"\nSaved hyperparameter search results -> {results_path}")

    print("\n" + "-" * 70)
    print("SELECTED HYPERPARAMETERS (lowest validation loss on fixed val split):")
    print(f"  Learning Rate    : {best['head_lr']}")
    print(f"  Weight Decay     : {best['weight_decay']}")
    print(f"  Label Smoothing  : {best['label_smoothing']}")
    print("-" * 70)

    return best


def train_final_model(base_dataset, trainval_idx, targets, num_classes, class_names,
                       head_lr=HEAD_LR, backbone_lr=BACKBONE_LR, weight_decay=WEIGHT_DECAY,
                       label_smoothing=LABEL_SMOOTHING):
    print("\n" + "=" * 70)
    print(f"STEP 2/3: FINAL TRAINING WITH EARLY STOPPING "
          f"(max {FINAL_MAX_EPOCHS} epochs, patience={FINAL_EARLY_STOPPING_PATIENCE})")
    print("=" * 70)

    set_seed(RANDOM_SEED)

    y = targets[trainval_idx]
    # Carve out a small slice purely to monitor early stopping - the real test
    # set stays completely untouched until Step 3.
    final_train_raw_idx, es_val_idx = train_test_split(
        trainval_idx, test_size=FINAL_HOLD_OUT_VAL_FRACTION, stratify=y, random_state=RANDOM_SEED
    )

    train_loader, counts_before, counts_after, target_count = build_balanced_train_loader(
        base_dataset, final_train_raw_idx, targets, BATCH_SIZE, NUM_WORKERS, seed=RANDOM_SEED
    )
    es_val_loader = build_eval_loader(base_dataset, es_val_idx, BATCH_SIZE, NUM_WORKERS)

    print_class_count_table(counts_before, class_names, "Final training - class counts BEFORE balancing:")
    print(f"  Target count per class (= largest class): {target_count}")
    print_class_count_table(counts_after, class_names, "Final training - class counts AFTER balancing:")
    print(f"  Early-stopping monitor set (originals only, never augmented): {len(es_val_idx)}")

    model = build_model(MODEL_NAME, num_classes).to(DEVICE)
    model = maybe_compile_model(model)   # no-op fallback if torch.compile is unsupported
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = build_differential_optimizer(model, backbone_lr, head_lr, weight_decay)
    scheduler = build_scheduler(optimizer, FINAL_WARMUP_EPOCHS, FINAL_MAX_EPOCHS, len(train_loader))

    checkpoint_path = os.path.join(OUTPUT_DIR, "best_convnext_model.pth")
    model, history, best_val_acc, best_val_loss, stopped_epoch = train_model(
        model, train_loader, es_val_loader, criterion, optimizer, scheduler, DEVICE,
        FINAL_MAX_EPOCHS, checkpoint_path, patience=FINAL_EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA, class_names=class_names, model_name=MODEL_NAME, verbose=True,
    )

    print(f"\nFinal model best validation accuracy: {best_val_acc:.4f} (stopped at epoch {stopped_epoch})")
    return (model, history, best_val_acc, stopped_epoch,
            counts_before, counts_after, target_count, len(es_val_idx))

#%%
# ============================================================
# STEP 3: COMPREHENSIVE TEST EVALUATION
# ============================================================

def evaluate_on_test_set(model, test_loader, class_names, output_dir):
    print("\n" + "=" * 70)
    print("STEP 3/3: COMPREHENSIVE EVALUATION ON THE TEST SET (never seen, never augmented)")
    print("=" * 70)

    labels, preds, probs = collect_predictions_and_probs(model, test_loader, DEVICE)
    metrics = compute_summary_metrics(labels, preds)

    print(f"Overall Accuracy       : {metrics['accuracy']:.4f}")
    print(f"Precision (macro)      : {metrics['precision_macro']:.4f}")
    print(f"Recall    (macro)      : {metrics['recall_macro']:.4f}")
    print(f"F1-score  (macro)      : {metrics['f1_macro']:.4f}")
    print(f"Precision (weighted)   : {metrics['precision_weighted']:.4f}")
    print(f"Recall    (weighted)   : {metrics['recall_weighted']:.4f}")
    print(f"F1-score  (weighted)   : {metrics['f1_weighted']:.4f}")

    report_dict = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0)
    report_str = classification_report(labels, preds, target_names=class_names, zero_division=0)
    print("\nDetailed Classification Report:")
    print(report_str)

    report_csv_path = os.path.join(output_dir, "classification_report.csv")
    with open(report_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for class_name in class_names:
            row = report_dict[class_name]
            writer.writerow([class_name, row["precision"], row["recall"], row["f1-score"], row["support"]])
        for avg_name in ["macro avg", "weighted avg"]:
            row = report_dict[avg_name]
            writer.writerow([avg_name, row["precision"], row["recall"], row["f1-score"], row["support"]])
    print(f"\nSaved detailed report -> {report_csv_path}")

    cm = confusion_matrix(labels, preds, normalize="true")
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar_kws={"label": "Proportion"})
    plt.xlabel("Predicted label"); plt.ylabel("True label")
    plt.title("Normalized Confusion Matrix - Test Set")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix chart -> {cm_path}")

    per_class_f1 = [report_dict[name]["f1-score"] for name in class_names]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, per_class_f1, color="steelblue")
    plt.ylim(0, 1.0)
    plt.ylabel("F1-score")
    plt.title("Per-Class F1-Score - Test Set")
    plt.xticks(rotation=45, ha="right")
    for bar, score in zip(bars, per_class_f1):
        plt.text(bar.get_x() + bar.get_width() / 2, score + 0.01, f"{score:.2f}",
                  ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    f1_path = os.path.join(output_dir, "per_class_f1.png")
    plt.savefig(f1_path, dpi=200)
    plt.close()
    print(f"Saved per-class F1 chart -> {f1_path}")

  
    num_classes = len(class_names)
    top_k = min(5, num_classes)
    top_k_acc = top_k_accuracy_score(labels, probs, k=top_k, labels=np.arange(num_classes))
    metrics[f"top{top_k}_accuracy"] = float(top_k_acc)
    print(f"Top-{top_k} Accuracy          : {top_k_acc:.4f}")


    if num_classes == 2:

        y_bin = np.zeros((len(labels), 2), dtype=int)
        y_bin[np.arange(len(labels)), labels] = 1
    else:
        y_bin = label_binarize(labels, classes=np.arange(num_classes))

    fpr, tpr, roc_auc_per_class = {}, {}, {}
    precision_curve, recall_curve, ap_per_class = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc_per_class[i] = auc(fpr[i], tpr[i])
        precision_curve[i], recall_curve[i], _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ap_per_class[i] = average_precision_score(y_bin[:, i], probs[:, i])

    roc_auc_macro = float(np.mean(list(roc_auc_per_class.values())))
    ap_macro = float(np.mean(list(ap_per_class.values())))
    metrics["roc_auc_macro"] = roc_auc_macro
    metrics["average_precision_macro"] = ap_macro
    print(f"ROC-AUC (macro, OvR)   : {roc_auc_macro:.4f}")
    print(f"Avg Precision (macro, OvR): {ap_macro:.4f}")

    plt.figure(figsize=(9, 7))
    for i, name in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], lw=1.5, label=f"{name} (AUC={roc_auc_per_class[i]:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves (One-vs-Rest) - Test Set (macro-AUC = {roc_auc_macro:.3f})")
    plt.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print(f"Saved ROC curve chart -> {roc_path}")

    plt.figure(figsize=(9, 7))
    for i, name in enumerate(class_names):
        plt.plot(recall_curve[i], precision_curve[i], lw=1.5, label=f"{name} (AP={ap_per_class[i]:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curves (One-vs-Rest) - Test Set (macro-AP = {ap_macro:.3f})")
    plt.legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    pr_path = os.path.join(output_dir, "precision_recall_curve.png")
    plt.savefig(pr_path, dpi=200)
    plt.close()
    print(f"Saved Precision-Recall curve chart -> {pr_path}")

    return metrics, report_dict


def plot_training_curves(history, output_dir):
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs_range, history["val_loss"], label="Validation Loss", marker="o")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training vs. Validation Loss (final model)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_range, history["train_acc"], label="Train Accuracy", marker="o")
    axes[1].plot(epochs_range, history["val_acc"], label="Validation Accuracy", marker="o")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training vs. Validation Accuracy (final model)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    curves_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(curves_path, dpi=200)
    plt.close()
    print(f"Saved training curves chart -> {curves_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Dataset directory (clean, no pre-existing augmented images expected): {os.path.abspath(DATA_DIR)}")

    base_dataset, class_names, num_classes, targets = load_full_dataset(DATA_DIR)
    all_idx = np.arange(len(base_dataset))

    # ---- Stratified split of the CLEAN data (fixed 75/15/10 ratio) ----
    print("\n" + "=" * 70)
    print(f"STRATIFIED SPLIT OF CLEAN DATA: "
          f"{int(TRAIN_RATIO*100)}% train / {int(VAL_RATIO*100)}% val / {int(TEST_RATIO*100)}% test")
    print("=" * 70)
    train_idx, val_idx, test_idx = split_dataset_stratified(
        all_idx, targets, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # Diagnostic preview: per-class counts of the raw 75% train split, before any
    # balancing/CV/final-training logic runs. The actual dynamic balancing happens
    # separately inside each cross-validation fold and inside the final training run
    # (each re-balances its own training portion to its own largest class).
    preview_counts = Counter(int(c) for c in targets[train_idx])
    print_class_count_table(preview_counts, class_names,
                             "Preview - raw TRAIN split class counts (before balancing):")
    if preview_counts:
        print(f"  (For reference, largest class in this raw split has "
              f"{max(preview_counts.values())} images.)")

    trainval_idx = np.concatenate([train_idx, val_idx])

    # ---- Step 0: hyperparameter tuning (Learning Rate / Weight Decay / Label Smoothing) ----
    # Uses a FIXED train/val split (fast, one run per candidate) - the test set is untouched.
    best_hparams = run_hyperparameter_search(base_dataset, train_idx, val_idx, targets, num_classes, class_names)
    print("\nFinal selected hyperparameters (used for CV reporting and final training below):")
    print(f"  Learning Rate    : {best_hparams['head_lr']}")
    print(f"  Weight Decay     : {best_hparams['weight_decay']}")
    print(f"  Label Smoothing  : {best_hparams['label_smoothing']}")

    # ---- Step 1: cross-validation on the (train+val) pool ----
    cv_fold_results = run_cross_validation(
        base_dataset, trainval_idx, targets, num_classes, class_names,
        head_lr=best_hparams["head_lr"], backbone_lr=best_hparams["backbone_lr"],
        weight_decay=best_hparams["weight_decay"], label_smoothing=best_hparams["label_smoothing"],
    )
    cv_summary = print_and_save_cv_results(cv_fold_results, OUTPUT_DIR)

    # ---- Step 2: final training with early stopping ----
    (model, history, final_val_acc, final_stopped_epoch,
     final_counts_before, final_counts_after, final_target_count, final_es_val_size) = train_final_model(
        base_dataset, trainval_idx, targets, num_classes, class_names,
        head_lr=best_hparams["head_lr"], backbone_lr=best_hparams["backbone_lr"],
        weight_decay=best_hparams["weight_decay"], label_smoothing=best_hparams["label_smoothing"],
    )

    history_path = os.path.join(OUTPUT_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history -> {history_path}")

    plot_training_curves(history, OUTPUT_DIR)

    # ---- Step 3: comprehensive test evaluation ----
    test_loader = build_eval_loader(base_dataset, test_idx, BATCH_SIZE, NUM_WORKERS)
    test_metrics, report_dict = evaluate_on_test_set(model, test_loader, class_names, OUTPUT_DIR)

    metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Saved test metrics -> {metrics_path}")


    print("\nAll done. Check the output folder for checkpoints and charts:")
    print(f"  {os.path.abspath(OUTPUT_DIR)}")
    print("\nTo test the trained model on your own image, run:")
    print('  python predict_image.py --image "path/to/your_photo.jpg"')


if __name__ == "__main__":
    main()

# %%
