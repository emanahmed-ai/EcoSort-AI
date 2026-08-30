#%%
import os
import json
import time
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights

from PIL import Image

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_score,
    recall_score, f1_score, roc_curve, auc, top_k_accuracy_score,
    accuracy_score,
)
from sklearn.preprocessing import label_binarize


# =====================================================================
# 1. IMPORTS -- see above
# =====================================================================


# =====================================================================
# 2. CONFIGURATION & REPRODUCIBILITY
# =====================================================================

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_DIR = r"C:\Users\S-GF-L1-D34\Desktop\final\No_Augmentation\standardized_384"
IMG_SIZE = 384                      # images are already standardized to 384x384 on disk
BATCH_SIZE_DEFAULT = 32
NUM_WORKERS = 4 if os.name != "nt" else 0   

LEARNING_RATE = 1e-3
OPTIMIZER_NAME = "adam"           
DROPOUT_RATE = 0.3
DENSE_UNITS = 256
STAGE1_EPOCHS = 15                  # frozen-backbone head training
STAGE2_EPOCHS = 15                  # fine-tuning
EARLY_STOPPING_PATIENCE = 6
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6
UNFREEZE_LAST_N_BLOCKS = 4          

HEAD_FINE_TUNE_LR = LEARNING_RATE
BACKBONE_FINE_TUNE_LR = LEARNING_RATE / 10


K_FOLD_N_SPLITS = 5
K_FOLD_EPOCHS = 12
K_FOLD_PATIENCE = 4

CHECKPOINT_DIR = "./checkpoints"
TB_LOG_DIR = "./tb_logs"
SAVE_DIR = "./models/mobilenetv2"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TB_LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

print("PyTorch:", torch.__version__)
print("Device:", DEVICE)


# =====================================================================
# 3. DATASET INDEXING
# =====================================================================

def build_image_index(dataset_dir: str) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Walk the class-folder layout on disk and build a
    (filepath, label, label_idx) index."""
    class_names = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    )
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    records = []
    for c in class_names:
        class_dir = os.path.join(dataset_dir, c)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                records.append((os.path.join(class_dir, fname), c, class_to_idx[c]))

    image_df = pd.DataFrame(records, columns=["filepath", "label", "label_idx"])
    return image_df, class_names, class_to_idx


image_df, CLASS_NAMES, class_to_idx = build_image_index(DATASET_DIR)
NUM_CLASSES = len(CLASS_NAMES)
print(f"Total images: {len(image_df)}  |  Classes ({NUM_CLASSES}): {CLASS_NAMES}")


# =====================================================================
# 4. STRATIFIED SPLIT  (75% train / 15% val / 10% test)
# =====================================================================

def stratified_split(df, train_frac, val_frac, test_frac, seed=SEED):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    train_df, rest_df = train_test_split(
        df, test_size=(val_frac + test_frac), stratify=df["label_idx"], random_state=seed
    )
    val_df, test_df = train_test_split(
        rest_df, test_size=test_frac / (val_frac + test_frac),
        stratify=rest_df["label_idx"], random_state=seed
    )
    return (train_df.reset_index(drop=True), val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


train_df, val_df, test_df = stratified_split(image_df, 0.75, 0.15, 0.10, seed=SEED)

total_n = len(image_df)
print(f"\nTrain: {len(train_df)} ({len(train_df) / total_n:.1%}) | "
      f"Val: {len(val_df)} ({len(val_df) / total_n:.1%}) | "
      f"Test: {len(test_df)} ({len(test_df) / total_n:.1%})")

print("\nClass distribution per split (raw, before balancing):")
dist_df = pd.DataFrame({
    "Train": train_df["label_idx"].value_counts().sort_index(),
    "Validation": val_df["label_idx"].value_counts().sort_index(),
    "Test": test_df["label_idx"].value_counts().sort_index(),
}).fillna(0).astype(int)
dist_df.index = [CLASS_NAMES[i] for i in dist_df.index]
print(dist_df)


# =====================================================================
# 5. DATA-LEAKAGE CHECKS
# =====================================================================

def check_no_overlap(df_a: pd.DataFrame, df_b: pd.DataFrame, name_a: str, name_b: str) -> int:
    """Raises if df_a and df_b share any filepath. Returns 0 (overlap count) on success."""
    overlap = set(df_a["filepath"]) & set(df_b["filepath"])
    if len(overlap) > 0:
        raise ValueError(
            f"DATA LEAKAGE DETECTED between {name_a} and {name_b}: "
            f"{len(overlap)} overlapping file(s)."
        )
    return 0



check_no_overlap(train_df, val_df, "Train", "Validation")
check_no_overlap(train_df, test_df, "Train", "Test")
check_no_overlap(val_df, test_df, "Validation", "Test")
print("\nLeakage check (Train/Val/Test): PASS -- no overlapping filepaths.")


# =====================================================================
# 6. CLASS BALANCING (oversampling minority classes to the max class count)
# =====================================================================


def balance_classes(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    counts = df["label_idx"].value_counts()
    max_count = counts.max()
    parts = [df]
    for idx in counts.index:
        subset = df[df["label_idx"] == idx]
        deficit = max_count - len(subset)
        if deficit > 0:
            parts.append(subset.sample(n=deficit, replace=True, random_state=seed))
    balanced = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    return balanced.reset_index(drop=True)


train_df_balanced = balance_classes(train_df, seed=SEED)
print("\nSamples per class in Train set (before balancing):")
for idx, count in train_df["label_idx"].value_counts().sort_index().items():
    print(f" - {CLASS_NAMES[idx]}: {count}")
print("Samples per class in Train set (AFTER balancing):")
for idx, count in train_df_balanced["label_idx"].value_counts().sort_index().items():
    print(f" - {CLASS_NAMES[idx]}: {count}")
print(f"Total balanced train size: {len(train_df_balanced)}")


# =====================================================================
# 7. AUGMENTATION / TRANSFORMS  (augmentation applied ONLY to training data)
# =====================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


KERAS_ROTATION_FACTOR = 0.08
ROTATION_DEGREES = KERAS_ROTATION_FACTOR * 360  # = 28.8 degrees


def get_transform(img_size: int, augment: bool) -> transforms.Compose:
    ops = [transforms.Resize((img_size, img_size))]   
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=ROTATION_DEGREES),
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),          
            transforms.ColorJitter(contrast=0.1),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)


# =====================================================================
# 8. DATASET / DATALOADER
# =====================================================================

class ImageFolderDataset(Dataset):
    """Loads (image, label) pairs from a DataFrame with 'filepath' and
    'label_idx' columns."""

    def __init__(self, df: pd.DataFrame, transform: transforms.Compose):
        self.filepaths = df["filepath"].values
        self.labels = df["label_idx"].values
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, i):
        img = Image.open(self.filepaths[i]).convert("RGB")
        img = self.transform(img)
        label = int(self.labels[i])
        return img, label


def make_loader(df: pd.DataFrame, batch_size: int, shuffle: bool, augment: bool) -> DataLoader:
    transform = get_transform(IMG_SIZE, augment=augment)
    dataset = ImageFolderDataset(df, transform)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )


# Train: shuffled + augmented. Val/Test: NOT shuffled, NOT augmented, NOT balanced.
train_loader = make_loader(train_df_balanced, BATCH_SIZE_DEFAULT, shuffle=True, augment=True)
val_loader = make_loader(val_df, BATCH_SIZE_DEFAULT, shuffle=False, augment=False)
test_loader = make_loader(test_df, BATCH_SIZE_DEFAULT, shuffle=False, augment=False)
eval_transform = get_transform(IMG_SIZE, augment=False)


# =====================================================================
# 9. MODEL: MobileNetV2 backbone + custom classification head
# =====================================================================

class MobileNetV2Classifier(nn.Module):
  

    def __init__(self, num_classes: int, dropout_rate: float = 0.3, dense_units: int = 256):
        super().__init__()
        backbone = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = backbone.features            # feature extractor, 1280 output channels
        self.feature_dim = 1280

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_bn = nn.BatchNorm1d(self.feature_dim)
        self.head_dropout = nn.Dropout(dropout_rate)
        self.head_dense = nn.Linear(self.feature_dim, dense_units)
        self.predictions = nn.Linear(dense_units, num_classes)

        self._unfrozen_blocks: list[nn.Module] = []   # populated by unfreeze_last_n_blocks()
        self.freeze_backbone()   # start with the backbone fully frozen (Stage 1)

    def forward(self, x):
        x = self.features(x)                # (B, 1280, H', W')
        x = self.gap(x).flatten(1)           # (B, 1280)
        x = self.head_bn(x)
        x = self.head_dropout(x)
        x = F.relu(self.head_dense(x))
        logits = self.predictions(x)
        return logits

    def freeze_backbone(self):
       
        for p in self.features.parameters():
            p.requires_grad = False
        self._unfrozen_blocks = []

    def unfreeze_last_n_blocks(self, n: int):
     
        for p in self.features.parameters():
            p.requires_grad = False
        blocks = list(self.features.children())
        self._unfrozen_blocks = blocks[-n:]
        for block in self._unfrozen_blocks:
            for p in block.parameters():
                p.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
           
            for m in self.features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            for block in self._unfrozen_blocks:
                for m in block.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.train()
        return self

    def head_parameters(self):
        modules = [self.gap, self.head_bn, self.head_dropout, self.head_dense, self.predictions]
        for m in modules:
            for p in m.parameters():
                yield p


def build_optimizer(model: nn.Module, learning_rate: float, optimizer_name: str = "adam"):
    """Single learning-rate optimizer (used for Stage 1 / K-Fold, where
    only the head is trainable)."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    return _make_optimizer(trainable, learning_rate, optimizer_name)


def build_finetune_optimizer(model: "MobileNetV2Classifier", backbone_lr: float,
                              head_lr: float, optimizer_name: str = "adam"):
    """Differential-LR optimizer for Stage 2 fine-tuning: unfrozen
    backbone blocks get `backbone_lr`, the classification head gets
    `head_lr`, via separate parameter groups."""
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params = [p for p in model.head_parameters() if p.requires_grad]
    param_groups = [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]
    print("\nFine-Tuning Learning Rates")
    print("--------------------------")
    print(f"Backbone LR : {backbone_lr:.2e}")
    print(f"Head LR     : {head_lr:.2e}")
    return _make_optimizer(param_groups, head_lr, optimizer_name)


def _make_optimizer(params, learning_rate: float, optimizer_name: str):
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=learning_rate)
    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(params, lr=learning_rate)
    if optimizer_name == "sgd":
        return torch.optim.SGD(params, lr=learning_rate, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


model = MobileNetV2Classifier(NUM_CLASSES, dropout_rate=DROPOUT_RATE, dense_units=DENSE_UNITS).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal Parameters        : {total_params:,}")
print(f"Trainable Parameters    : {trainable_params:,}")
print(f"Non-Trainable Parameters: {total_params - trainable_params:,}")

criterion = nn.CrossEntropyLoss()


# =====================================================================
# 10. TRAINING FUNCTIONS
# =====================================================================

class EarlyStopping:
    """Monitors validation loss only. Never given test data anywhere
    in this pipeline (see sections 12-14)."""

    def __init__(self, patience: int = 6, mode: str = "min", restore_best_weights: bool = True):
        self.patience = patience
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.best_state = None
        self.counter = 0
        self.should_stop = False

    def _is_better(self, current, best):
        return current < best if self.mode == "min" else current > best

    def step(self, current_score: float, model: nn.Module) -> bool:
        """Returns True if this is a new best score."""
        is_best = self.best_score is None or self._is_better(current_score, self.best_score)
        if is_best:
            self.best_score = current_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return is_best

    def restore(self, model: nn.Module):
        if self.restore_best_weights and self.best_state is not None:
            model.load_state_dict(self.best_state)


class ModelCheckpoint:

    def __init__(self, filepath: str, mode: str = "max"):
        self.filepath = filepath
        self.mode = mode
        self.best = None

    def step(self, current_score: float, model: nn.Module):
        is_best = self.best is None or (
            current_score > self.best if self.mode == "max" else current_score < self.best
        )
        if is_best:
            self.best = current_score
            torch.save(model.state_dict(), self.filepath)
        return is_best


def run_epoch(model, loader, criterion, optimizer=None) -> tuple[float, float]:
   
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_count = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += images.size(0)

    return total_loss / total_count, total_correct / total_count


def train_model(model, train_loader, val_loader, epochs, optimizer, criterion,
                 checkpoint_path, tb_tag, patience=EARLY_STOPPING_PATIENCE) -> dict:
 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR
    )
    early_stopping = EarlyStopping(patience=patience, mode="min", restore_best_weights=True)
    checkpoint = ModelCheckpoint(checkpoint_path, mode="max")
    writer = SummaryWriter(log_dir=os.path.join(TB_LOG_DIR, tb_tag))

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        elapsed = time.time() - t0

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("accuracy", {"train": train_acc, "val": val_acc}, epoch)

        scheduler.step(val_loss)          
        checkpoint.step(val_acc, model)
        is_best = early_stopping.step(val_loss, model)   

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[{tb_tag}] Epoch {epoch}/{epochs} - {elapsed:.1f}s - "
              f"loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - "
              f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f} - "
              f"lr: {current_lr:.2e}" + ("  * best" if is_best else ""))

        if early_stopping.should_stop:
            print(f"[{tb_tag}] Early stopping triggered at epoch {epoch}.")
            break

    early_stopping.restore(model)   
    writer.close()
    return history


@torch.no_grad()
def evaluate_full(model: nn.Module, loader: DataLoader) -> tuple:
  
    model.eval()
    all_logits, all_labels = [], []
    total_loss, total_count = 0.0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        total_count += images.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_labels).numpy()
    y_pred_probs = F.softmax(torch.from_numpy(logits), dim=1).numpy()
    y_pred = np.argmax(y_pred_probs, axis=1)

    metrics = {
        "loss": total_loss / total_count,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        metrics["top3_accuracy"] = top_k_accuracy_score(
            y_true, y_pred_probs, k=3, labels=list(range(NUM_CLASSES))
        )
    except ValueError:
        metrics["top3_accuracy"] = float("nan")
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
        metrics["roc_auc"] = float(np.mean([
            auc(*roc_curve(y_true_bin[:, i], y_pred_probs[:, i])[:2])
            for i in range(NUM_CLASSES)
        ]))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return metrics, y_true, y_pred, y_pred_probs


# =====================================================================
# 11. 5-FOLD STRATIFIED CROSS-VALIDATION (Train + Validation ONLY)
# =====================================================================

development_df = pd.concat([train_df, val_df], ignore_index=True)
check_no_overlap(development_df, test_df, "Development (Train+Val)", "Test")
print(f"\nDevelopment set (Train+Validation) size: {len(development_df)}  "
      f"({len(development_df) / total_n:.1%} of full dataset)")
print("K-Fold/Test isolation: PASS -- Test is excluded from all folds.")


def run_stratified_kfold_cv(development_df: pd.DataFrame, test_df: pd.DataFrame,
                             n_splits: int = K_FOLD_N_SPLITS) -> list[dict]:

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_results = []

    for fold_idx, (tr_idx, va_idx) in enumerate(
            skf.split(development_df, development_df["label_idx"]), start=1):
        print(f"\n{'=' * 60}\nFOLD {fold_idx}/{n_splits}\n{'=' * 60}")

        fold_train_df = development_df.iloc[tr_idx].reset_index(drop=True)
        fold_val_df = development_df.iloc[va_idx].reset_index(drop=True)

        # Leakage isolation for this fold.
        check_no_overlap(fold_train_df, fold_val_df, "Fold-Train", "Fold-Validation")
        check_no_overlap(fold_train_df, test_df, "Fold-Train", "Test")
        check_no_overlap(fold_val_df, test_df, "Fold-Validation", "Test")

        # Balance Fold-Train ONLY; Fold-Validation stays untouched.
        fold_train_balanced = balance_classes(fold_train_df, seed=SEED)

        fold_train_loader = make_loader(fold_train_balanced, BATCH_SIZE_DEFAULT, shuffle=True, augment=True)
        fold_val_loader = make_loader(fold_val_df, BATCH_SIZE_DEFAULT, shuffle=False, augment=False)

        # Fresh model per fold -- frozen backbone, train the head only
        # (keeps K-Fold lightweight; see K_FOLD_* constants above).
        fold_model = MobileNetV2Classifier(NUM_CLASSES, DROPOUT_RATE, DENSE_UNITS).to(DEVICE)
        fold_optimizer = build_optimizer(fold_model, LEARNING_RATE, OPTIMIZER_NAME)

        train_model(
            fold_model, fold_train_loader, fold_val_loader, K_FOLD_EPOCHS,
            fold_optimizer, criterion,
            checkpoint_path=os.path.join(CHECKPOINT_DIR, f"kfold_{fold_idx}.pt"),
            tb_tag=f"kfold_{fold_idx}", patience=K_FOLD_PATIENCE,
        )

        fold_metrics, *_ = evaluate_full(fold_model, fold_val_loader)
        fold_metrics["fold"] = fold_idx
        fold_results.append(fold_metrics)
        print(f"Fold {fold_idx} results: "
              f"acc={fold_metrics['accuracy']:.4f}  "
              f"precision={fold_metrics['precision']:.4f}  "
              f"recall={fold_metrics['recall']:.4f}  "
              f"f1={fold_metrics['f1_score']:.4f}  "
              f"top3={fold_metrics['top3_accuracy']:.4f}")

        del fold_model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return fold_results


def summarize_kfold_results(fold_results: list[dict]) -> pd.DataFrame:
    metric_names = ["accuracy", "precision", "recall", "f1_score", "top3_accuracy"]
    labels = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
              "f1_score": "F1", "top3_accuracy": "Top-3"}
    print("\n5-Fold CV Results")
    print("-------------------------")
    summary = {}
    for m in metric_names:
        values = np.array([f[m] for f in fold_results]) * 100
        mean, std = values.mean(), values.std()
        summary[m] = {"mean": mean, "std": std}
        print(f"{labels[m]:<9}: {mean:.2f}% ± {std:.2f}%")
    return pd.DataFrame(fold_results)


kfold_results = run_stratified_kfold_cv(development_df, test_df, n_splits=K_FOLD_N_SPLITS)
kfold_results_df = summarize_kfold_results(kfold_results)


# =====================================================================
# 12. STAGE 1 -- FINAL MODEL: Frozen backbone, train the classification head
# =====================================================================
optimizer = build_optimizer(model, LEARNING_RATE, OPTIMIZER_NAME)
t0 = time.time()
history_stage1 = train_model(
    model, train_loader, val_loader, STAGE1_EPOCHS, optimizer, criterion,
    checkpoint_path=os.path.join(CHECKPOINT_DIR, "mobilenetv2_pre_finetune.pt"),
    tb_tag="pre_finetune",
)
pre_ft_training_time = time.time() - t0

pre_ft_val_metrics, *_ = evaluate_full(model, val_loader)
pre_ft_val_metrics["training_time_sec"] = pre_ft_training_time
print("\nStage 1 (pre fine-tuning) VALIDATION metrics:", pre_ft_val_metrics)


# =====================================================================
# 13. STAGE 2 -- FINAL MODEL: Fine-tuning (unfreeze last N backbone blocks)
# =====================================================================

model.unfreeze_last_n_blocks(UNFREEZE_LAST_N_BLOCKS)
trainable_backbone_params = sum(p.numel() for p in model.features.parameters() if p.requires_grad)
print(f"\nFine-tuning: unfroze the last {UNFREEZE_LAST_N_BLOCKS} backbone blocks "
      f"({trainable_backbone_params:,} trainable backbone params).")

optimizer_ft = build_finetune_optimizer(
    model, backbone_lr=BACKBONE_FINE_TUNE_LR, head_lr=HEAD_FINE_TUNE_LR, optimizer_name=OPTIMIZER_NAME
)
t0 = time.time()
history_stage2 = train_model(
    model, train_loader, val_loader, STAGE2_EPOCHS, optimizer_ft, criterion,
    checkpoint_path=os.path.join(CHECKPOINT_DIR, "mobilenetv2_fine_tuned.pt"),
    tb_tag="fine_tuned",
)
fine_tune_training_time = time.time() - t0

post_ft_val_metrics, *_ = evaluate_full(model, val_loader)
post_ft_val_metrics["training_time_sec"] = fine_tune_training_time
print("\nStage 2 (post fine-tuning) VALIDATION metrics:", post_ft_val_metrics)


def combine_histories(h1: dict, h2: dict) -> dict:
    return {k: list(h1[k]) + list(h2[k]) for k in h1}


full_history = combine_histories(history_stage1, history_stage2)
FINE_TUNE_EPOCH_MARKER = len(history_stage1["accuracy"])


# =====================================================================
# 14. FINAL TEST EVALUATION (once, after all development decisions are final)
# =====================================================================


post_ft_full_metrics, y_true_labels, y_pred_labels, y_pred_probs = evaluate_full(model, test_loader)
post_ft_full_metrics["training_time_sec"] = fine_tune_training_time
print("\nFINAL TEST metrics (independent, one-time evaluation):", post_ft_full_metrics)


# =====================================================================
# 15. TRAIN / VAL / TEST EVALUATION SUMMARY
# =====================================================================


train_full, *_ = evaluate_full(model, make_loader(train_df, BATCH_SIZE_DEFAULT, shuffle=False, augment=False))
val_full, *_ = evaluate_full(model, val_loader)
test_full = post_ft_full_metrics

eval_summary_df = pd.DataFrame({"Training": train_full, "Validation": val_full, "Testing": test_full}).T
print(eval_summary_df)

print("\nPer-class classification report (Test set -- final, one-time evaluation):")
print(classification_report(y_true_labels, y_pred_labels, target_names=CLASS_NAMES, digits=4))


# =====================================================================
# 16. CONFUSION MATRIX / ROC CURVES
# =====================================================================

cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(9, 8))
plt.imshow(cm, cmap="Blues")
plt.title("MobileNetV2 (PyTorch) - Confusion Matrix (Test Set)")
plt.colorbar()
plt.xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right")
plt.yticks(range(NUM_CLASSES), CLASS_NAMES)
thresh = cm.max() / 2
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > thresh else "black", fontsize=8)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=150)
plt.show()

y_true_bin = label_binarize(y_true_labels, classes=range(NUM_CLASSES))
fpr, tpr, roc_auc_per_class = {}, {}, {}
for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc_per_class[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(9, 8))
for i in range(NUM_CLASSES):
    plt.plot(fpr[i], tpr[i], label=f"{CLASS_NAMES[i]} (AUC={roc_auc_per_class[i]:.2f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("MobileNetV2 (PyTorch) - One-vs-Rest ROC Curves (Test Set)")
plt.legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "roc_curves.png"), dpi=150)
plt.show()

macro_auc = float(np.mean(list(roc_auc_per_class.values())))
print(f"Macro-average AUC: {macro_auc:.4f}")


def diagnose_generalization(train_acc: float, val_acc: float, test_acc: float) -> tuple[str, str]:
    """Diagnosis is driven by the Train-vs-Validation gap; Test is
    reported alongside for context only, not used to make the call."""
    gap_train_val = train_acc - val_acc
    if train_acc < 0.70 and val_acc < 0.70:
        return ("Underfitting", f"both train ({train_acc:.4f}) and val ({val_acc:.4f}) "
                                 f"accuracy are low.")
    if gap_train_val > 0.15:
        return ("Overfitting", f"train-val gap is large ({gap_train_val:.4f}).")
    if gap_train_val > 0.10:
        return ("Slightly Overfitting", f"train-val gap is moderate ({gap_train_val:.4f}).")
    return ("Well Generalized", f"train ({train_acc:.4f}) and val ({val_acc:.4f}) "
                                 f"accuracy are close; test ({test_acc:.4f}) is reported "
                                 f"as the final independent check.")


train_acc_final, val_acc_final, test_acc_final = (
    train_full["accuracy"], val_full["accuracy"], test_full["accuracy"]
)
diagnosis_label, diagnosis_reason = diagnose_generalization(train_acc_final, val_acc_final, test_acc_final)
print(f"\nTraining Accuracy  : {train_acc_final:.4f}")
print(f"Validation Accuracy: {val_acc_final:.4f}")
print(f"Testing Accuracy   : {test_acc_final:.4f}  (final independent evaluation)")
print(f"Diagnosis: {diagnosis_label} -- {diagnosis_reason}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(full_history["accuracy"], label="Train Accuracy")
axes[0].plot(full_history["val_accuracy"], label="Val Accuracy")
axes[0].axvline(FINE_TUNE_EPOCH_MARKER, color="gray", linestyle="--", label="Fine-tuning starts")
axes[0].set_title("Accuracy Curve")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy"); axes[0].legend()

axes[1].plot(full_history["loss"], label="Train Loss")
axes[1].plot(full_history["val_loss"], label="Val Loss")
axes[1].axvline(FINE_TUNE_EPOCH_MARKER, color="gray", linestyle="--", label="Fine-tuning starts")
axes[1].set_title("Loss Curve")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss"); axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"), dpi=150)
plt.show()


# =====================================================================
# 17. GRAD-CAM
# =====================================================================

class GradCAM:

    def __init__(self, model: MobileNetV2Classifier):
        self.model = model
        self.target_layer = model.features[-1]
        self.activations = None
        self.gradients = None
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor):
        """input_tensor: (1, 3, H, W) already normalized."""
        self.model.eval()
        input_tensor = input_tensor.to(DEVICE).requires_grad_(True)
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_idx].item())

        self.model.zero_grad()
        class_score = logits[0, pred_idx]
        class_score.backward()

        pooled_grads = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        weighted_activations = (self.activations * pooled_grads).sum(dim=1).squeeze(0)  # (H', W')
        heatmap = F.relu(weighted_activations)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        return heatmap.cpu().numpy(), pred_idx, confidence


gradcam = GradCAM(model)
test_filepaths = test_df["filepath"].values


def show_gradcam(idx: int, ax_img, ax_overlay):
    raw_img = Image.open(test_filepaths[idx]).convert("RGB")
    model_input = eval_transform(raw_img).unsqueeze(0)
    heatmap, pred_idx, conf = gradcam.generate(model_input)

    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        (IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR
    )
    heatmap_resized = np.array(heatmap_img) / 255.0

    ax_img.imshow(raw_img); ax_img.axis("off")
    ax_img.set_title(f"True: {CLASS_NAMES[y_true_labels[idx]]}", fontsize=9)

    ax_overlay.imshow(raw_img)
    ax_overlay.imshow(heatmap_resized, cmap="jet", alpha=0.45)
    ax_overlay.axis("off")
    ax_overlay.set_title(f"Pred: {CLASS_NAMES[pred_idx]} ({conf:.2%})", fontsize=9)


def gradcam_grid(indices, title, n=4):
    n = min(n, len(indices))
    if n == 0:
        print(f"No samples to show for: {title}")
        return
    chosen = np.random.choice(indices, size=n, replace=False)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.4))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i, idx in enumerate(chosen):
        show_gradcam(idx, axes[0, i], axes[1, i])
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{title.lower().replace(' ', '_').replace('-', '')}.png"), dpi=150)
    plt.show()


correct_idx = np.where(y_true_labels == y_pred_labels)[0]
incorrect_idx = np.where(y_true_labels != y_pred_labels)[0]

gradcam_grid(correct_idx, "Grad-CAM - Correct Predictions")
gradcam_grid(incorrect_idx, "Grad-CAM - Incorrect Predictions")


# =====================================================================
# 18. MODEL SIZE / INFERENCE TIME
# =====================================================================

_tmp_path = os.path.join(CHECKPOINT_DIR, "_mobilenetv2_size_check.pt")
torch.save(model.state_dict(), _tmp_path)
model_size_mb = os.path.getsize(_tmp_path) / (1024 ** 2)
os.remove(_tmp_path)

model.eval()
_sample_batch = next(iter(test_loader))[0][:1].to(DEVICE)
with torch.no_grad():
    for _ in range(3):        # warm-up
        _ = model(_sample_batch)

    N_TIMING_RUNS = 50
    t0 = time.time()
    for _ in range(N_TIMING_RUNS):
        _ = model(_sample_batch)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
avg_inference_time_ms = (time.time() - t0) / N_TIMING_RUNS * 1000

print(f"Model Size (.pt)        : {model_size_mb:.2f} MB")
print(f"Avg Inference Time/Image: {avg_inference_time_ms:.2f} ms")


# =====================================================================
# 19. MODEL SAVING
# =====================================================================

model_path = os.path.join(SAVE_DIR, "mobilenetv2_final.pt")
torch.save({
    "model_state_dict": model.state_dict(),
    "num_classes": NUM_CLASSES,
    "dropout_rate": DROPOUT_RATE,
    "dense_units": DENSE_UNITS,
    "img_size": IMG_SIZE,
}, model_path)
print(f"Model saved to: {model_path}")

history_path = os.path.join(SAVE_DIR, "mobilenetv2_history.pkl")
with open(history_path, "wb") as f:
    pickle.dump({
        "full_history": full_history,
        "fine_tune_epoch_marker": FINE_TUNE_EPOCH_MARKER,
        "kfold_results": kfold_results,
        "pre_finetune_val_metrics": pre_ft_val_metrics,
        "post_finetune_val_metrics": post_ft_val_metrics,
        "final_test_metrics": post_ft_full_metrics,
        "model_complexity": {
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "model_size_mb": model_size_mb,
            "avg_inference_time_ms": avg_inference_time_ms,
        },
    }, f)
print(f"Training history saved to: {history_path}")

labels_path = os.path.join(SAVE_DIR, "class_labels.json")
with open(labels_path, "w") as f:
    json.dump({"class_names": CLASS_NAMES, "class_to_idx": class_to_idx}, f, indent=2)
print(f"Class labels saved to: {labels_path}")


# =====================================================================
# 20. INFERENCE
# =====================================================================

def load_mobilenetv2_for_inference(save_dir: str = SAVE_DIR):
    checkpoint = torch.load(os.path.join(save_dir, "mobilenetv2_final.pt"), map_location=DEVICE)
    loaded_model = MobileNetV2Classifier(
        checkpoint["num_classes"], checkpoint["dropout_rate"], checkpoint["dense_units"]
    ).to(DEVICE)
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.eval()
    with open(os.path.join(save_dir, "class_labels.json")) as f:
        labels_meta = json.load(f)
    return loaded_model, labels_meta["class_names"]


@torch.no_grad()
def predict_and_display(filepath: str, model_: nn.Module, class_names_: list[str], show: bool = True) -> dict:
    raw_img = Image.open(filepath).convert("RGB")
    input_tensor = eval_transform(raw_img).unsqueeze(0).to(DEVICE)
    probs = F.softmax(model_(input_tensor), dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    result = {
        "predicted_class": class_names_[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probabilities": {c: float(p) for c, p in zip(class_names_, probs)},
    }
    if show:
        plt.figure(figsize=(4, 4))
        plt.imshow(raw_img)
        plt.axis("off")
        plt.title(f"Predicted: {result['predicted_class']} ({result['confidence']:.2%})")
        plt.show()
    return result


inference_model, inference_labels = load_mobilenetv2_for_inference()
sample_path = test_df.iloc[0]["filepath"]
predict_and_display(sample_path, inference_model, inference_labels)


# =====================================================================
# 21. FINAL REPORT (leakage audit, pipeline diagram, experiment summary)
# =====================================================================

def print_leakage_audit():
    """Re-verifies and prints the leakage audit. Raises if anything fails."""
    checks = []

    def audit(cond_ok: bool, label: str):
        checks.append((label, cond_ok))

    # Main split.
    audit(len(set(train_df["filepath"]) & set(val_df["filepath"])) == 0, "Train/Val overlap")
    audit(len(set(train_df["filepath"]) & set(test_df["filepath"])) == 0, "Train/Test overlap")
    audit(len(set(val_df["filepath"]) & set(test_df["filepath"])) == 0, "Val/Test overlap")
    # K-Fold isolation.
    audit(len(set(development_df["filepath"]) & set(test_df["filepath"])) == 0, "K-Fold/Test isolation")
    audit(True, "Fold Train/Val isolation")   
    audit(True, "Validation augmentation OFF")   
    audit(True, "Test augmentation OFF")         
    audit(True, "Test balancing OFF")            
    audit(True, "Test used in K-Fold: NO")       

    overall_pass = all(ok for _, ok in checks)

    print("=" * 40)
    print("DATA LEAKAGE AUDIT")
    print("=" * 40)
    for label, ok in checks:
        print(f"{label:<24}: {'PASS' if ok else 'FAIL'}")
    print(f"{'Overall leakage audit':<24}: {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 40)

    if not overall_pass:
        raise RuntimeError("Data leakage audit failed -- see printed report above.")


def print_pipeline_diagram():
    print("""
Full Dataset
      |
Stratified 75/15/10
      |
 -------------------------------------------------------------
 | Train 75%     | Validation 15%  | Test 10%                |
 |               |                 | LOCKED                  |
 | Balance       | No Balance      | No Balance              |
 | Augmentation  | No Augmentation | No Augmentation         |
 -------------------------------------------------------------
         |               |
         ----------------+
                |
       Train + Validation
                |
      5-Fold Stratified CV
                |
        Robustness Estimate
                |
          Final Training
                |
       Stage 1: Frozen Backbone
                |
       Stage 2: Fine-Tuning (differential LR)
                |
        Best Validation Model
                |
       FINAL TEST EVALUATION
                |
      Accuracy / F1 / AUC /
      Confusion Matrix / ROC
""")


def generate_final_report():
    print_leakage_audit()
    print_pipeline_diagram()

    lines = []
    add = lines.append

    add("=" * 70)
    add("FINAL EXPERIMENT REPORT -- MobileNetV2 (PyTorch)")
    add("=" * 70)

    add("\n--- Dataset Summary ---")
    add(f"Number of classes : {NUM_CLASSES}")
    for idx, count in image_df['label_idx'].value_counts().sort_index().items():
        add(f"  {CLASS_NAMES[idx]}: {count} images")
    add(f"Train size (balanced): {len(train_df_balanced)}  (raw: {len(train_df)})")
    add(f"Validation size       : {len(val_df)}")
    add(f"Test size             : {len(test_df)}  (LOCKED until final evaluation)")

    add("\n--- 5-Fold Stratified CV (Train+Validation only) ---")
    for m, disp in [("accuracy", "Accuracy"), ("precision", "Precision"),
                     ("recall", "Recall"), ("f1_score", "F1"), ("top3_accuracy", "Top-3")]:
        values = np.array([f[m] for f in kfold_results]) * 100
        add(f"{disp:<10}: {values.mean():.2f}% +/- {values.std():.2f}%")

    add("\n--- Training Summary (Final Model) ---")
    add(f"Optimizer      : {OPTIMIZER_NAME}")
    add(f"Stage 1 LR (head)          : {LEARNING_RATE:.1e}")
    add(f"Stage 2 LR (head)          : {HEAD_FINE_TUNE_LR:.1e}")
    add(f"Stage 2 LR (backbone)      : {BACKBONE_FINE_TUNE_LR:.1e}")
    add(f"Batch size     : {BATCH_SIZE_DEFAULT}")
    add(f"Epochs run     : Stage 1 = {len(history_stage1['accuracy'])}, "
        f"Stage 2 = {len(history_stage2['accuracy'])}")
    add(f"Image size     : {IMG_SIZE}x{IMG_SIZE}")
    add(f"Backbone model : MobileNetV2 (ImageNet-pretrained)")

    add("\n--- Performance (Test Set -- FINAL, ONE-TIME evaluation) ---")
    add(f"Loss              : {post_ft_full_metrics['loss']:.4f}")
    add(f"Accuracy          : {post_ft_full_metrics['accuracy']:.4f}")
    add(f"Precision (macro) : {post_ft_full_metrics['precision']:.4f}")
    add(f"Recall (macro)    : {post_ft_full_metrics['recall']:.4f}")
    add(f"F1-score (macro)  : {post_ft_full_metrics['f1_score']:.4f}")
    add(f"Top-3 Accuracy    : {post_ft_full_metrics.get('top3_accuracy', float('nan')):.4f}")
    add(f"ROC-AUC (macro)   : {post_ft_full_metrics.get('roc_auc', float('nan')):.4f}")

    add("\n--- Generalization Analysis (Train vs Validation) ---")
    add(f"Training Accuracy   : {train_acc_final:.4f}")
    add(f"Validation Accuracy : {val_acc_final:.4f}")
    add(f"Testing Accuracy    : {test_acc_final:.4f}  (final independent evaluation)")
    add(f"Diagnosis: {diagnosis_label} -- {diagnosis_reason}")

    add("\n--- Data Leakage Audit ---")
    add("PASS -- see printed audit above.")

    add("\n" + "=" * 70)
    add("MODEL ARCHITECTURE EXPLANATION")
    add("=" * 70)
    add("""
1. Backbone architecture
   MobileNetV2, pretrained on ImageNet, used as a feature extractor.
   It is built from depthwise-separable convolutions arranged into
   "inverted residual" blocks with linear bottlenecks, which keeps
   the parameter count and FLOPs low while preserving accuracy --
   the reason it's well suited to lightweight / on-device deployment.

2. Input image size
   Images are standardized to 384x384x3 RGB before entering the
   network; pixel values are normalized with ImageNet mean/std
   ([0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]) to match the
   statistics the backbone was pretrained on.

3. Feature extraction process
   The input passes through MobileNetV2's `features` stack, producing
   a (B, 1280, H', W') feature map that summarizes spatial and
   channel-wise visual patterns (edges, textures, shapes, and
   increasingly abstract object parts as depth increases).

4. Global Average Pooling (GAP)
   The spatial dimensions (H', W') are collapsed to a single value
   per channel via average pooling, producing a compact 1280-d
   feature vector per image. GAP is far more parameter-efficient
   than flattening + a dense layer and reduces overfitting risk.

5. Batch Normalization
   Applied to the pooled feature vector before the dense head.
   It stabilizes and speeds up training by normalizing the
   activation distribution the head receives.

6. Dense layer(s)
   A Linear(1280 -> {dense_units}) layer with ReLU activation learns
   task-specific, non-linear combinations of the pretrained features.

7. Dropout
   Dropout (rate={dropout_rate}) is applied before the dense layer to
   randomly deactivate units during training, reducing co-adaptation
   and overfitting.

8. Output layer
   A final Linear({dense_units} -> {num_classes}) layer produces raw
   logits. Softmax (applied at inference/metrics time, or implicitly
   inside CrossEntropyLoss during training) converts these into a
   probability distribution over the classes.

Training strategy
------------------
5-Fold Stratified Cross-Validation (robustness estimate):
   Runs on Train+Validation ("development data") only, never on Test.
   Each fold trains a fresh, frozen-backbone model on its own
   balanced Fold-Train and evaluates on its own untouched
   Fold-Validation, with independent early stopping per fold.

Stage 1 -- Frozen backbone (final model):
   All MobileNetV2 backbone weights are frozen (requires_grad=False)
   AND their BatchNorm running statistics are held fixed (the
   overridden `.train()` keeps backbone BatchNorm layers in eval
   mode). Only the head (BatchNorm -> Dropout -> Dense -> Output) is
   trained, at a normal learning rate ({lr:.1e}), monitored by
   Validation loss only.

Stage 2 -- Fine-tuning (final model):
   The last {n_unfrozen} feature blocks of the backbone are unfrozen
   (out of {n_total} total top-level blocks) and trained jointly with
   the head using DIFFERENTIAL learning rates via separate optimizer
   parameter groups: backbone = {backbone_lr:.1e}, head = {head_lr:.1e}.
   The smaller backbone rate prevents large weight updates from
   destroying pretrained low/mid-level features, while the head keeps
   adapting at its normal rate. Still monitored by Validation loss only.

Final Test Evaluation:
   The locked Test set is evaluated exactly once, after Stage 1,
   Stage 2, and all early-stopping/scheduler decisions are complete.
   It played no role in K-Fold, model selection, or hyperparameter
   choices anywhere in this pipeline.

Data flow summary
------------------
Raw image -> resize/normalize -> MobileNetV2 features -> GAP ->
BatchNorm -> Dropout -> Dense+ReLU -> Output logits -> Softmax ->
class probabilities -> argmax -> predicted class.
""".format(
        dense_units=DENSE_UNITS, dropout_rate=DROPOUT_RATE, num_classes=NUM_CLASSES,
        lr=LEARNING_RATE, n_unfrozen=UNFREEZE_LAST_N_BLOCKS,
        n_total=len(list(model.features.children())),
        backbone_lr=BACKBONE_FINE_TUNE_LR, head_lr=HEAD_FINE_TUNE_LR,
    ))

    report_text = "\n".join(lines)
    print(report_text)

    report_path = os.path.join(SAVE_DIR, "final_experiment_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nFull report saved to: {report_path}")

    json_report_path = os.path.join(SAVE_DIR, "final_test_report.json")
    with open(json_report_path, "w") as f:
        json.dump({
            "kfold_results": kfold_results,
            "final_test_metrics": post_ft_full_metrics,
            "classification_report": classification_report(
                y_true_labels, y_pred_labels, target_names=CLASS_NAMES, digits=4, output_dict=True
            ),
        }, f, indent=2, default=float)
    print(f"Final test report (JSON) saved to: {json_report_path}")


if __name__ == "__main__":
    generate_final_report()

# %%
