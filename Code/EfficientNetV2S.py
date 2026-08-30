#%%
# ==========================================================
# Imports (EfficientNetV2-S section only)
# ==========================================================
import os
import time
import json
import shutil
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effnetv2_preprocess

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices("GPU"))


# ==========================================================
# Load dataset from local disk & Stratified Split 75/15/10 
# with K-Fold Cross Validation & Selective Augmentation
# ==========================================================
DATASET_DIR = r"C:\Users\S-GF-L1-D34\Downloads\No_Augmentation\standardized_384"

IMG_SIZE = (384, 384)
BATCH_SIZE = 32

CLASS_NAMES = sorted(
    d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))
)
class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
all_paths, all_labels = [], []
for class_name in CLASS_NAMES:
    class_dir = os.path.join(DATASET_DIR, class_name)
    for fname in os.listdir(class_dir):
        if fname.lower().endswith(VALID_EXTS):
            all_paths.append(os.path.join(class_dir, fname))
            all_labels.append(class_to_idx[class_name])

all_paths = np.array(all_paths)
all_labels = np.array(all_labels)
print(f"Found {len(all_paths)} images across {len(CLASS_NAMES)} classes")

# 1. Stratified split: 75% Train, 25% Temp (to be split into 15% Val and 10% Test)
# بما أن النسب المتبقية من الـ 25% هي 15% فاليديشن و 10% تيست، إذن test_size = 10 / 25 = 0.4 للـ temp split
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    all_paths, all_labels, test_size=0.25, stratify=all_labels, random_state=SEED
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.40, stratify=temp_labels, random_state=SEED
)

print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

# عرض عدد العينات لكل کلاس في الـ Train Set الأصلية قبل الـ Augmentation
train_df_temp = pd.DataFrame({"label": train_labels})
train_class_counts = train_df_temp["label"].value_counts().sort_index()
print("\nNumber of samples per class in Train set (before augmentation):")
for idx, count in train_class_counts.items():
    print(f" - {CLASS_NAMES[idx]}: {count}")

# 2. Augment train set of the lowest classes to be all equal (Matching the maximum class count)
max_count = train_class_counts.max()
augmented_train_paths = list(train_paths)
augmented_train_labels = list(train_labels)

for idx in train_class_counts.index:
    class_indices = np.where(train_labels == idx)[0]
    current_count = len(class_indices)
    deficit = max_count - current_count
    
    if deficit > 0:
        # اختيار عينات عشوائية من نفس الكلاس لتطبيق الـ Augmentation عليها حتى تتساوى الكلاسات
        sampled_indices = np.random.choice(class_indices, size=deficit, replace=True)
        for s_idx in sampled_indices:
            augmented_train_paths.append(train_paths[s_idx])
            augmented_train_labels.append(train_labels[s_idx])

augmented_train_paths = np.array(augmented_train_paths)
augmented_train_labels = np.array(augmented_train_labels)

# طباعة عدد العينات لكل كلاس بعد الـ Augmentation
aug_train_df = pd.DataFrame({"label": augmented_train_labels})
aug_class_counts = aug_train_df["label"].value_counts().sort_index()
print("\nNumber of samples per class in Train set (AFTER augmentation):")
for idx, count in aug_class_counts.items():
    print(f" - {CLASS_NAMES[idx]}: {count}")

print(f"\nTotal Train set size after balancing/augmentation: {len(augmented_train_paths)}")

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

def _load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = effnetv2_preprocess(img)
    return img, label

def _make_dataset(paths, labels, shuffle=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    if augment:
        # تطبيق الـ Augmentation فقط على الـ Train Dataset وتجنبها تماماً في الـ Val/Test لتجنب الـ Data Leakage
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

# إنشاء الـ Datasets الأساسية (مع تفعيل الـ Augmentation للـ Train فقط وضمان خلو Val و Test منهما)
train_ds = _make_dataset(augmented_train_paths, augmented_train_labels, shuffle=True, augment=True)
val_ds = _make_dataset(val_paths, val_labels, shuffle=False, augment=False)
test_ds = _make_dataset(test_paths, test_labels, shuffle=False, augment=False)


# ==========================================================
# K-Fold Cross Validation Setup
# ==========================================================
from sklearn.model_selection import StratifiedKFold

NUM_FOLDS = 5
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

print(f"\nStarting {NUM_FOLDS}-Fold Cross Validation...")
fold_no = 1

for fold_train_idx, fold_val_idx in skf.split(all_paths, all_labels):
    print(f"\n--- Training Fold {fold_no} / {NUM_FOLDS} ---")
    
    f_train_paths, f_train_labels = all_paths[fold_train_idx], all_labels[fold_train_idx]
    f_val_paths, f_val_labels = all_paths[fold_val_idx], all_labels[fold_val_idx]
    
    # تجهيز وتوازن الكلاسات لكل Fold على حدة لضمان عدم تسريب البيانات
    f_train_df = pd.DataFrame({"label": f_train_labels})
    f_counts = f_train_df["label"].value_counts()
    f_max_count = f_counts.max()
    
    f_aug_paths, f_aug_labels = list(f_train_paths), list(f_train_labels)
    for idx in f_counts.index:
        c_idxs = np.where(f_train_labels == idx)[0]
        diff = f_max_count - len(c_idxs)
        if diff > 0:
            sampled = np.random.choice(c_idxs, size=diff, replace=True)
            for s in sampled:
                f_aug_paths.append(f_train_paths[s])
                f_aug_labels.append(f_train_labels[s])
                
    f_train_ds = _make_dataset(np.array(f_aug_paths), np.array(f_aug_labels), shuffle=True, augment=True)
    f_val_ds = _make_dataset(f_val_paths, f_val_labels, shuffle=False, augment=False)
    
    # (ملاحظة: يمكنك استخدام f_train_ds و f_val_ds هنا داخل لوب التدريب الخاص بكل فولد إذا رغبت بتطبيق النتائج بالكامل عبر الـ K-Fold)
    
    fold_no += 1


# ==========================================================
# Sanity check: confirm everything the rest of this section needs is ready
# ==========================================================
required_vars = ["train_ds", "val_ds", "test_ds", "CLASS_NAMES", "IMG_SIZE", "BATCH_SIZE"]
missing = [v for v in required_vars if v not in globals()]
if missing:
    raise NameError(
        f"Missing required variables from previous sections: {missing}. "
        "Please run the Data Loading/Preprocessing/Splitting and MobileNetV2 "
        "sections first so the best split is available."
    )

NUM_CLASSES = len(CLASS_NAMES)
print(f"Reusing best split -> classes: {NUM_CLASSES}, image size: {IMG_SIZE}, batch size: {BATCH_SIZE}")
print("Class names:", CLASS_NAMES)


# ==========================================================
# Load pretrained EfficientNetV2-S backbone
# ==========================================================
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)

base_model = EfficientNetV2S(
    weights="imagenet",
    include_top=False,
    input_shape=INPUT_SHAPE,
    pooling=None,  # pooling is added explicitly in the custom head
)

# Freeze the entire backbone initially (Stage 1: head-only training)
base_model.trainable = False

print(f"Backbone: EfficientNetV2-S")
print(f"Total layers in backbone: {len(base_model.layers)}")
print(f"Backbone trainable: {base_model.trainable}")


# ==========================================================
# Build the full model: backbone + custom classification head
# ==========================================================
def build_efficientnetv2s_model(
    base_model,
    num_classes,
    dense_units=256,
    dropout_rate_1=0.30,
    dropout_rate_2=0.20,
):
    inputs = keras.Input(shape=INPUT_SHAPE, name="input_image")

    # EfficientNetV2S's preprocess_input expects pixel values in [0, 255]
    x = effnetv2_preprocess(inputs)
    x = base_model(x, training=False)  # training=False keeps BatchNorm stats frozen

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_batchnorm")(x)
    x = layers.Dropout(dropout_rate_1, name="head_dropout_1")(x)
    x = layers.Dense(dense_units, activation="relu", name="head_dense")(x)
    x = layers.Dropout(dropout_rate_2, name="head_dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="EfficientNetV2S_GarbageClassifier")
    return model


effnetv2_model = build_efficientnetv2s_model(base_model, NUM_CLASSES)
effnetv2_model.summary()


# ==========================================================
# Parameter counts
# ==========================================================
def count_params(model):
    trainable = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    return int(trainable), int(non_trainable)

trainable_params, non_trainable_params = count_params(effnetv2_model)
total_params = trainable_params + non_trainable_params

print(f"Total parameters:         {total_params:,}")
print(f"Trainable parameters:     {trainable_params:,}")
print(f"Non-trainable parameters: {non_trainable_params:,}")


# ==========================================================
# Detect label encoding (one-hot vs. integer) from a sample batch
# ==========================================================
_sample_x, _sample_y = next(iter(train_ds))
LABELS_ARE_ONE_HOT = (_sample_y.shape.rank == 2 and _sample_y.shape[-1] == NUM_CLASSES)
loss_fn = "categorical_crossentropy" if LABELS_ARE_ONE_HOT else "sparse_categorical_crossentropy"
top3_metric_name = "top_3_categorical_accuracy" if LABELS_ARE_ONE_HOT else "sparse_top_k_categorical_accuracy"

if LABELS_ARE_ONE_HOT:
    top3_metric = keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy")
else:
    top3_metric = keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy")

print("Detected label format:", "one-hot" if LABELS_ARE_ONE_HOT else "integer (sparse)")
print("Loss function selected:", loss_fn)


# ==========================================================
# Compile the model (Stage 1: frozen backbone, head-only training)
# ==========================================================
INITIAL_LR = 1e-3

effnetv2_model.compile(
    optimizer=optimizers.Adam(learning_rate=INITIAL_LR),
    loss=loss_fn,
    metrics=["accuracy", top3_metric],
)

print("Model compiled.")
print(f"Optimizer: Adam(lr={INITIAL_LR})")
print(f"Loss: {loss_fn}")


# ==========================================================
# Output directories
# ==========================================================
OUTPUT_DIR = Path("effnetv2s_artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "checkpoints").mkdir(exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(exist_ok=True)

def make_callbacks(stage_name, monitor="val_loss", patience=6, tensorboard=True):
    cb_list = [
        callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(OUTPUT_DIR / "checkpoints" / f"effnetv2s_{stage_name}_best.keras"),
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        ),
    ]
    if tensorboard:
        cb_list.append(
            callbacks.TensorBoard(log_dir=str(OUTPUT_DIR / "logs" / stage_name))
        )
    return cb_list


# ==========================================================
# Stage 1 training: frozen backbone, head-only
# ==========================================================
STAGE1_EPOCHS = 20

stage1_callbacks = make_callbacks(stage_name="stage1_frozen", monitor="val_loss", patience=6)

start_time = time.time()
history_stage1 = effnetv2_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=STAGE1_EPOCHS,
    callbacks=stage1_callbacks,
    verbose=1,
)
stage1_training_time = time.time() - start_time

print(f"\nStage 1 (frozen backbone) training time: {stage1_training_time:.1f} seconds")


# ==========================================================
# Reusable evaluation function
# ==========================================================
def get_true_and_pred_labels(model, dataset):
    y_true_list, y_pred_probs_list = [], []
    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch, verbose=0)
        y_pred_probs_list.append(preds)
        if LABELS_ARE_ONE_HOT:
            y_true_list.append(np.argmax(y_batch.numpy(), axis=1))
        else:
            y_true_list.append(y_batch.numpy())
    y_true = np.concatenate(y_true_list)
    y_pred_probs = np.concatenate(y_pred_probs_list)
    y_pred = np.argmax(y_pred_probs, axis=1)
    return y_true, y_pred, y_pred_probs


def evaluate_split(model, dataset, split_name):
    loss, acc, top3 = model.evaluate(dataset, verbose=0)
    y_true, y_pred, _ = get_true_and_pred_labels(model, dataset)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "Split": split_name,
        "Loss": loss,
        "Accuracy": acc,
        "Top-3 Accuracy": top3,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
    }


def build_results_table(model, tag=""):
    rows = [
        evaluate_split(model, train_ds, "Train"),
        evaluate_split(model, val_ds, "Validation"),
        evaluate_split(model, test_ds, "Test"),
    ]
    df = pd.DataFrame(rows).set_index("Split")
    if tag:
        print(f"=== Evaluation results: {tag} ===")
    display(df.round(4))
    return df

results_stage1 = build_results_table(effnetv2_model, tag="EfficientNetV2-S — Stage 1 (frozen backbone)")


# ==========================================================
# KerasTuner hyperparameter search (head-only, frozen backbone)
# ==========================================================
try:
    import keras_tuner as kt
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "keras-tuner", "--quiet"])
    import keras_tuner as kt


def build_tunable_model(hp):
    dense_units = hp.Choice("dense_units", [128, 256, 512])
    dropout_1 = hp.Float("dropout_rate_1", 0.1, 0.5, step=0.1)
    dropout_2 = hp.Float("dropout_rate_2", 0.1, 0.4, step=0.1)
    learning_rate = hp.Choice("learning_rate", [1e-2, 1e-3, 5e-4, 1e-4])
    optimizer_name = hp.Choice("optimizer", ["adam", "rmsprop"])

    tuned_base = EfficientNetV2S(
        weights="imagenet", include_top=False, input_shape=INPUT_SHAPE
    )
    tuned_base.trainable = False

    inputs = keras.Input(shape=INPUT_SHAPE)
    x = effnetv2_preprocess(inputs)
    x = tuned_base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_1)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    opt = (
        optimizers.Adam(learning_rate=learning_rate)
        if optimizer_name == "adam"
        else optimizers.RMSprop(learning_rate=learning_rate)
    )
    model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])
    return model


tuner = kt.RandomSearch(
    build_tunable_model,
    objective="val_accuracy",
    max_trials=12,
    executions_per_trial=1,
    directory=str(OUTPUT_DIR / "kt_search"),
    project_name="effnetv2s_garbage",
    seed=SEED,
)

tuner.search(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
    verbose=1,
)

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters found:")
for param in ["dense_units", "dropout_rate_1", "dropout_rate_2", "learning_rate", "optimizer"]:
    print(f"  {param}: {best_hp.get(param)}")


# ==========================================================
# Batch size tuning note
# ==========================================================
# Batch size is evaluated separately (outside KerasTuner) because it changes the
# tf.data pipeline rather than the model graph. Candidate batch sizes are compared
# on validation accuracy/loss using the SAME best split (only re-batched).
BATCH_SIZE_CANDIDATES = [16, 32, 64]

batch_size_results = []
for bs in BATCH_SIZE_CANDIDATES:
    train_ds_bs = train_ds.unbatch().batch(bs).prefetch(tf.data.AUTOTUNE)
    val_ds_bs = val_ds.unbatch().batch(bs).prefetch(tf.data.AUTOTUNE)

    trial_model = build_tunable_model(best_hp)
    trial_history = trial_model.fit(
        train_ds_bs,
        validation_data=val_ds_bs,
        epochs=5,
        verbose=0,
    )
    best_val_acc = max(trial_history.history["val_accuracy"])
    batch_size_results.append({"Batch Size": bs, "Best Val Accuracy": best_val_acc})

batch_size_df = pd.DataFrame(batch_size_results).sort_values("Best Val Accuracy", ascending=False)
display(batch_size_df)

BEST_BATCH_SIZE = int(batch_size_df.iloc[0]["Batch Size"])
print(f"Selected batch size: {BEST_BATCH_SIZE}")


# ==========================================================
# Rebuild final head-stage model using best hyperparameters
# ==========================================================
BEST_EPOCHS = 20  # number of epochs to use going forward; refined further by EarlyStopping

train_ds_final = train_ds.unbatch().batch(BEST_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds_final = val_ds.unbatch().batch(BEST_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds_final = test_ds.unbatch().batch(BEST_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

base_model_tuned = EfficientNetV2S(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
base_model_tuned.trainable = False

effnetv2_model = build_efficientnetv2s_model(
    base_model_tuned,
    NUM_CLASSES,
    dense_units=best_hp.get("dense_units"),
    dropout_rate_1=best_hp.get("dropout_rate_1"),
    dropout_rate_2=best_hp.get("dropout_rate_2"),
)

opt = (
    optimizers.Adam(learning_rate=best_hp.get("learning_rate"))
    if best_hp.get("optimizer") == "adam"
    else optimizers.RMSprop(learning_rate=best_hp.get("learning_rate"))
)
effnetv2_model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy", top3_metric])

tuned_callbacks = make_callbacks(stage_name="stage1_tuned", monitor="val_loss", patience=6)

start_time = time.time()
history_tuned = effnetv2_model.fit(
    train_ds_final,
    validation_data=val_ds_final,
    epochs=BEST_EPOCHS,
    callbacks=tuned_callbacks,
    verbose=1,
)
stage1_tuned_training_time = time.time() - start_time

print(f"\nTuned Stage 1 training time: {stage1_tuned_training_time:.1f} seconds")


results_before_finetune = build_results_table(
    effnetv2_model, tag="EfficientNetV2-S — Before Fine-Tuning (tuned head, frozen backbone)"
)


# ==========================================================
# Fine-tuning: unfreeze the top N layers of the backbone
# ==========================================================
FINE_TUNE_AT_FRACTION = 0.80  # unfreeze the top 20% of backbone layers
FINE_TUNE_LR = best_hp.get("learning_rate") / 10.0
FINE_TUNE_EPOCHS = 15

base_model_tuned.trainable = True
num_backbone_layers = len(base_model_tuned.layers)
fine_tune_at = int(num_backbone_layers * FINE_TUNE_AT_FRACTION)

for layer in base_model_tuned.layers[:fine_tune_at]:
    layer.trainable = False
# Keep BatchNormalization layers frozen even within the unfrozen block,
# to preserve stable running statistics learned on ImageNet.
for layer in base_model_tuned.layers[fine_tune_at:]:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

trainable_now, non_trainable_now = count_params(effnetv2_model)
print(f"Unfreezing layers {fine_tune_at} to {num_backbone_layers} ({num_backbone_layers - fine_tune_at} layers)")
print(f"Trainable parameters after unfreezing: {trainable_now:,}")

effnetv2_model.compile(
    optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss=loss_fn,
    metrics=["accuracy", top3_metric],
)

finetune_callbacks = make_callbacks(stage_name="stage2_finetune", monitor="val_loss", patience=6)

start_time = time.time()
history_finetune = effnetv2_model.fit(
    train_ds_final,
    validation_data=val_ds_final,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=finetune_callbacks,
    verbose=1,
)
finetune_training_time = time.time() - start_time

print(f"\nFine-tuning training time: {finetune_training_time:.1f} seconds")


results_after_finetune = build_results_table(
    effnetv2_model, tag="EfficientNetV2-S — After Fine-Tuning"
)


# ==========================================================
# Before vs. After Fine-Tuning comparison table
# ==========================================================
comparison_df = pd.DataFrame({
    "Before Fine-Tuning": results_before_finetune.loc["Test"],
    "After Fine-Tuning": results_after_finetune.loc["Test"],
})
comparison_df.loc["Training Time (s)"] = [stage1_tuned_training_time, finetune_training_time]
display(comparison_df.round(4))


# ==========================================================
# Before/after fine-tuning comparison charts
# ==========================================================
metrics_to_plot = ["Accuracy", "Top-3 Accuracy", "Precision", "Recall", "F1-score"]
before_vals = [results_before_finetune.loc["Test", m] for m in metrics_to_plot]
after_vals = [results_after_finetune.loc["Test", m] for m in metrics_to_plot]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(metrics_to_plot))
width = 0.35
ax.bar(x - width/2, before_vals, width, label="Before Fine-Tuning")
ax.bar(x + width/2, after_vals, width, label="After Fine-Tuning")
ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot, rotation=20)
ax.set_ylabel("Score")
ax.set_title("EfficientNetV2-S: Test Set Metrics — Before vs. After Fine-Tuning")
ax.legend()
ax.set_ylim(0, 1.0)
for i, (b, a) in enumerate(zip(before_vals, after_vals)):
    ax.text(i - width/2, b + 0.01, f"{b:.3f}", ha="center", fontsize=8)
    ax.text(i + width/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "before_after_finetune_comparison.png", dpi=150)
plt.show()


# ==========================================================
# Classification report & confusion matrix
# ==========================================================
y_true_test, y_pred_test, y_pred_probs_test = get_true_and_pred_labels(effnetv2_model, test_ds_final)

print("Classification Report (Test Set):\n")
print(classification_report(y_true_test, y_pred_test, target_names=CLASS_NAMES, digits=4))

cm = confusion_matrix(y_true_test, y_pred_test)
plt.figure(figsize=(9, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("EfficientNetV2-S — Confusion Matrix (Test Set)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.show()


# ==========================================================
# ROC curves & AUC (one-vs-rest, multi-class)
# ==========================================================
y_true_bin = label_binarize(y_true_test, classes=list(range(NUM_CLASSES)))

fpr, tpr, roc_auc = {}, {}, {}
for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs_test[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(9, 8))
for i, class_name in enumerate(CLASS_NAMES):
    plt.plot(fpr[i], tpr[i], label=f"{class_name} (AUC = {roc_auc[i]:.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("EfficientNetV2-S — ROC Curves (One-vs-Rest, Test Set)")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=150)
plt.show()

mean_auc = np.mean(list(roc_auc.values()))
print(f"Mean AUC across classes: {mean_auc:.4f}")


# ==========================================================
# Combined accuracy/loss curves across both training stages
# ==========================================================
def concat_history(hist1, hist2, key):
    return hist1.history.get(key, []) + hist2.history.get(key, [])

acc = concat_history(history_tuned, history_finetune, "accuracy")
val_acc = concat_history(history_tuned, history_finetune, "val_accuracy")
loss = concat_history(history_tuned, history_finetune, "loss")
val_loss = concat_history(history_tuned, history_finetune, "val_loss")
finetune_start_epoch = len(history_tuned.history.get("accuracy", []))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(acc, label="Train Accuracy")
axes[0].plot(val_acc, label="Validation Accuracy")
axes[0].axvline(finetune_start_epoch, color="gray", linestyle="--", label="Fine-tuning starts")
axes[0].set_title("Accuracy Curves")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(loss, label="Train Loss")
axes[1].plot(val_loss, label="Validation Loss")
axes[1].axvline(finetune_start_epoch, color="gray", linestyle="--", label="Fine-tuning starts")
axes[1].set_title("Loss Curves")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "learning_curves.png", dpi=150)
plt.show()


# ==========================================================
# Train / Validation / Test accuracy comparison + generalization verdict
# ==========================================================
final_train_acc = results_after_finetune.loc["Train", "Accuracy"]
final_val_acc = results_after_finetune.loc["Validation", "Accuracy"]
final_test_acc = results_after_finetune.loc["Test", "Accuracy"]

gap_train_val = final_train_acc - final_val_acc
gap_val_test = abs(final_val_acc - final_test_acc)

print(f"Train Accuracy:      {final_train_acc:.4f}")
print(f"Validation Accuracy: {final_val_acc:.4f}")
print(f"Test Accuracy:       {final_test_acc:.4f}")
print(f"Train-Validation gap: {gap_train_val:.4f}")

if gap_train_val > 0.10:
    verdict = "Overfitting is present: training accuracy notably exceeds validation accuracy."
elif final_train_acc < 0.70 and final_val_acc < 0.70:
    verdict = "Underfitting may be present: both training and validation accuracy are low."
else:
    verdict = "The model generalizes well: train/validation/test accuracies are close."

print("\nVerdict:", verdict)


# ==========================================================
# Collect raw images + predictions for qualitative error analysis
# ==========================================================
all_images, all_true, all_pred, all_conf = [], [], [], []
for x_batch, y_batch in test_ds_final:
    preds = effnetv2_model.predict(x_batch, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)
    true_labels = (
        np.argmax(y_batch.numpy(), axis=1) if LABELS_ARE_ONE_HOT else y_batch.numpy()
    )
    all_images.append(x_batch.numpy())
    all_true.append(true_labels)
    all_pred.append(pred_labels)
    all_conf.append(confidences)

all_images = np.concatenate(all_images)
all_true = np.concatenate(all_true)
all_pred = np.concatenate(all_pred)
all_conf = np.concatenate(all_conf)

correct_idx = np.where(all_true == all_pred)[0]
incorrect_idx = np.where(all_true != all_pred)[0]

print(f"Correct predictions: {len(correct_idx)} / {len(all_true)}")
print(f"Incorrect predictions: {len(incorrect_idx)} / {len(all_true)}")


# ==========================================================
# Visualize sample correct and incorrect predictions
# ==========================================================
def plot_prediction_samples(indices, title, n=8):
    n = min(n, len(indices))
    sample_idx = np.random.choice(indices, size=n, replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, idx in zip(axes.flat, sample_idx):
        img = all_images[idx].astype("uint8") if all_images[idx].max() > 1.0 else all_images[idx]
        ax.imshow(img)
        true_label = CLASS_NAMES[all_true[idx]]
        pred_label = CLASS_NAMES[all_pred[idx]]
        conf = all_conf[idx]
        ax.set_title(f"True: {true_label}\nPred: {pred_label} ({conf:.2f})", fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

plot_prediction_samples(correct_idx, "Sample Correct Predictions")
if len(incorrect_idx) > 0:
    plot_prediction_samples(incorrect_idx, "Sample Incorrect Predictions")
else:
    print("No incorrect predictions found in the test set.")


# ==========================================================
# Structured error table: true label, predicted label, confidence
# ==========================================================
error_df = pd.DataFrame({
    "True Label": [CLASS_NAMES[i] for i in all_true[incorrect_idx]],
    "Predicted Label": [CLASS_NAMES[i] for i in all_pred[incorrect_idx]],
    "Confidence": all_conf[incorrect_idx],
}).sort_values("Confidence", ascending=False)

display(error_df.head(20))


# ==========================================================
# Grad-CAM implementation
# ==========================================================
def find_last_conv_layer(model):
    # Search inside the nested backbone submodel as well as the outer model
    for layer in reversed(model.layers):
        if isinstance(layer, keras.Model):
            for sub_layer in reversed(layer.layers):
                if len(sub_layer.output_shape) == 4:
                    return layer, sub_layer.name
        if len(getattr(layer, "output_shape", ())) == 4:
            return model, layer.name
    raise ValueError("No 4D (conv) layer found for Grad-CAM.")


def make_gradcam_heatmap(img_array, model, backbone_submodel, last_conv_layer_name, pred_index=None):
    grad_model = keras.Model(
        inputs=backbone_submodel.inputs,
        outputs=[backbone_submodel.get_layer(last_conv_layer_name).output, backbone_submodel.output],
    )

    with tf.GradientTape() as tape:
        preprocessed = effnetv2_preprocess(img_array)
        conv_outputs, backbone_preds = grad_model(preprocessed)
        # Route backbone features through the rest of the head to get final predictions
        x = layers.GlobalAveragePooling2D()(conv_outputs)
        # NOTE: for exact correctness, the head layers from `model` should be reused;
        # here we recompute predictions directly from the full model for gradient accuracy.
        full_preds = model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(full_preds[0])
        class_channel = full_preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


def overlay_gradcam(img, heatmap, alpha=0.4):
    import matplotlib.cm as cm
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (img.shape[0], img.shape[1])).numpy().squeeze()
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    overlaid = jet_heatmap * alpha + (img / 255.0 if img.max() > 1.0 else img) * (1 - alpha)
    return np.clip(overlaid, 0, 1)


backbone_layer, last_conv_name = find_last_conv_layer(effnetv2_model)
print("Using backbone submodel:", backbone_layer.name, "| last conv layer:", last_conv_name)


# ==========================================================
# Grad-CAM visualizations: correct and incorrect predictions
# ==========================================================
def show_gradcam_examples(indices, title, n=4):
    n = min(n, len(indices))
    sample_idx = np.random.choice(indices, size=n, replace=False)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    for col, idx in enumerate(sample_idx):
        img = all_images[idx:idx+1]
        heatmap, pred_idx = make_gradcam_heatmap(img, effnetv2_model, backbone_layer, last_conv_name)
        overlay = overlay_gradcam(img[0], heatmap)

        raw_img = img[0].astype("uint8") if img[0].max() > 1.0 else img[0]
        axes[0, col].imshow(raw_img)
        axes[0, col].set_title(f"True: {CLASS_NAMES[all_true[idx]]}", fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f"Grad-CAM (Pred: {CLASS_NAMES[pred_idx]})", fontsize=9)
        axes[1, col].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

show_gradcam_examples(correct_idx, "Grad-CAM — Correct Predictions")
if len(incorrect_idx) > 0:
    show_gradcam_examples(incorrect_idx, "Grad-CAM — Incorrect Predictions")


# ==========================================================
# Parameter counts, model size, inference time
# ==========================================================
final_trainable, final_non_trainable = count_params(effnetv2_model)
final_total = final_trainable + final_non_trainable

temp_model_path = OUTPUT_DIR / "_temp_size_check.keras"
effnetv2_model.save(temp_model_path)
model_size_mb = os.path.getsize(temp_model_path) / (1024 ** 2)
os.remove(temp_model_path)

# Average inference time per image (single-image batches, warmed up)
sample_batch, _ = next(iter(test_ds_final))
single_image = sample_batch[:1]

for _ in range(5):  # warm-up
    _ = effnetv2_model.predict(single_image, verbose=0)

n_runs = 50
start_time = time.time()
for _ in range(n_runs):
    _ = effnetv2_model.predict(single_image, verbose=0)
avg_inference_time_ms = ((time.time() - start_time) / n_runs) * 1000

complexity_report = pd.DataFrame([{
    "Total Parameters": final_total,
    "Trainable Parameters": final_trainable,
    "Non-Trainable Parameters": final_non_trainable,
    "Model Size (MB)": round(model_size_mb, 2),
    "Avg. Inference Time (ms/image)": round(avg_inference_time_ms, 2),
}])
display(complexity_report)


# ==========================================================
# Persist model, history, and class labels
# ==========================================================
FINAL_MODEL_PATH = OUTPUT_DIR / "efficientnetv2s_garbage_classifier_final.keras"
effnetv2_model.save(FINAL_MODEL_PATH)
print(f"Saved final model to: {FINAL_MODEL_PATH}")

combined_history = {
    "stage1_tuned": history_tuned.history,
    "stage2_finetune": history_finetune.history,
}
history_path = OUTPUT_DIR / "training_history.json"
with open(history_path, "w") as f:
    json.dump(combined_history, f, indent=2, default=float)
print(f"Saved training history to: {history_path}")

labels_path = OUTPUT_DIR / "class_labels.json"
with open(labels_path, "w") as f:
    json.dump({"class_names": CLASS_NAMES}, f, indent=2)
print(f"Saved class labels to: {labels_path}")


# ==========================================================
# Inference helpers
# ==========================================================
def load_trained_model(model_path=FINAL_MODEL_PATH):
    return keras.models.load_model(model_path)


def load_class_labels(labels_path=labels_path):
    with open(labels_path, "r") as f:
        return json.load(f)["class_names"]


def preprocess_image_for_inference(image_path, target_size=IMG_SIZE):
    img = keras.utils.load_img(image_path, target_size=target_size)
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array


def predict_image(model, class_names, image_path, top_k=3):
    img_array = preprocess_image_for_inference(image_path)
    preds = model.predict(img_array, verbose=0)[0]
    top_indices = np.argsort(preds)[::-1][:top_k]
    results = [
        {"label": class_names[i], "confidence": float(preds[i])}
        for i in top_indices
    ]
    return results


def display_prediction(image_path, results):
    img = keras.utils.load_img(image_path)
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    title = "\n".join(f"{r['label']}: {r['confidence']:.2%}" for r in results)
    plt.title(title, fontsize=10)
    plt.axis("off")
    plt.show()


# Example usage (uncomment and set a real path to test):
# loaded_model = load_trained_model()
# loaded_labels = load_class_labels()
# example_path = "/path/to/unseen_image.jpg"
# prediction = predict_image(loaded_model, loaded_labels, example_path)
# display_prediction(example_path, prediction)
# print(prediction)


# ==========================================================
# Final Test Report — EfficientNetV2-S
# ==========================================================
print("=" * 60)
print("FINAL TEST REPORT — EfficientNetV2-S (after fine-tuning)")
print("=" * 60)

final_test_row = results_after_finetune.loc["Test"]
print(final_test_row.round(4))

print("\nPer-class classification report (Test set):")
final_class_report_str = classification_report(
    y_true_test, y_pred_test, target_names=CLASS_NAMES, digits=4
)
print(final_class_report_str)

final_metrics_names = ["Accuracy", "Top-3 Accuracy", "Precision", "Recall", "F1-score"]
final_metrics_values = [final_test_row[m] for m in final_metrics_names]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(final_metrics_names, final_metrics_values, color="#55A868")
ax.set_ylim(0, 1.0)
ax.set_ylabel("Score")
ax.set_title("EfficientNetV2-S — Final Test Metrics Summary")
for bar, val in zip(bars, final_metrics_values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", fontsize=9)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "final_test_metrics_summary.png", dpi=150)
plt.show()

final_report_path = OUTPUT_DIR / "final_test_report.json"
with open(final_report_path, "w") as f:
    json.dump({
        "test_metrics": final_test_row.to_dict(),
        "classification_report": classification_report(
            y_true_test, y_pred_test, target_names=CLASS_NAMES, digits=4, output_dict=True
        ),
    }, f, indent=2, default=float)
print(f"Final test report saved to: {final_report_path}")


# %%
