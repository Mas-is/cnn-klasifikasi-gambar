# =========================================================
# CNN IMAGE CLASSIFICATION - ANIMALS DATASET
# =========================================================

# =========================================================
# 1. IMPORT LIBRARY
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# =========================================================
# 2. REPRODUCIBILITY
# =========================================================

tf.random.set_seed(42)
np.random.seed(42)

print("=" * 50)
print(" TensorFlow version :", tf.__version__)
print(" GPU tersedia       :", len(tf.config.list_physical_devices('GPU')) > 0)
print("=" * 50)

# =========================================================
# 3. LOAD DATASET
# =========================================================

dataset_path = "dataset"

IMG_SIZE = 128
BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE

# Training Dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Validation Dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# =========================================================
# 4. INFORMASI DATASET
# =========================================================

CLASS_NAMES = train_dataset.class_names
NUM_CLASSES = len(CLASS_NAMES)

print("\nClass Names :", CLASS_NAMES)
print("Jumlah Kelas:", NUM_CLASSES)

# =========================================================
# 5. PREPROCESSING
# =========================================================

def preprocess(image, label):

    # Resize
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])

    # Normalisasi
    image = tf.cast(image, tf.float32) / 255.0

    return image, label

# Terapkan preprocessing
ds_train_proc = train_dataset.map(
    preprocess,
    num_parallel_calls=AUTOTUNE
)

ds_val_proc = validation_dataset.map(
    preprocess,
    num_parallel_calls=AUTOTUNE
)

# =========================================================
# 6. DATA AUGMENTATION
# =========================================================

def augment(image, label):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    image = tf.image.random_brightness(
        image,
        max_delta=0.2
    )

    image = tf.image.random_contrast(
        image,
        lower=0.8,
        upper=1.2
    )

    image = tf.image.random_saturation(
        image,
        lower=0.7,
        upper=1.3
    )

    image = tf.clip_by_value(
        image,
        0.0,
        1.0
    )

    return image, label

# =========================================================
# 7. FINAL DATASET PIPELINE
# =========================================================

ds_train_final = (

    ds_train_proc

    .map(
        augment,
        num_parallel_calls=AUTOTUNE
    )

    .cache()

    .shuffle(
        buffer_size=1000,
        seed=42
    )

    .prefetch(AUTOTUNE)
)

ds_val_final = (

    ds_val_proc

    .cache()

    .prefetch(AUTOTUNE)
)

# Validation digunakan juga sebagai test
ds_test_final = ds_val_final

print("\nData augmentation berhasil diterapkan!")

# =========================================================
# 8. VISUALISASI DATASET
# =========================================================

plt.figure(figsize=(12,8))

for images, labels in ds_train_proc.take(1):

    for i in range(6):

        ax = plt.subplot(2,3,i+1)

        plt.imshow(images[i].numpy())

        plt.title(CLASS_NAMES[labels[i]])

        plt.axis("off")

plt.tight_layout()

plt.show()

# =========================================================
# 9. RANCANGAN ARSITEKTUR CNN
# =========================================================

"""
INPUT (128x128x3)

BLOCK 1:
Conv2D(32) → Conv2D(32)
→ MaxPooling → BatchNormalization

BLOCK 2:
Conv2D(64) → Conv2D(64)
→ MaxPooling → BatchNormalization

BLOCK 3:
Conv2D(128) → Conv2D(128)
→ MaxPooling → BatchNormalization

Flatten

Dense(256) → Dropout(0.5)

Dense(128) → Dropout(0.3)

Output → Softmax
"""

# =========================================================
# 10. MEMBANGUN MODEL CNN
# =========================================================

def build_cnn_model(
    num_classes,
    input_shape=(128,128,3)
):

    model = models.Sequential(
        name='CNN_Animals_Classifier'
    )

    # =====================================================
    # BLOCK 1
    # =====================================================

    model.add(

        layers.Conv2D(
            32,
            (3,3),
            padding='same',
            activation='relu',
            input_shape=input_shape
        )
    )

    model.add(

        layers.Conv2D(
            32,
            (3,3),
            padding='same',
            activation='relu'
        )
    )

    model.add(

        layers.MaxPooling2D((2,2))
    )

    model.add(

        layers.BatchNormalization()
    )

    model.add(

        layers.Dropout(0.25)
    )

    # =====================================================
    # BLOCK 2
    # =====================================================

    model.add(

        layers.Conv2D(
            64,
            (3,3),
            padding='same',
            activation='relu'
        )
    )

    model.add(

        layers.Conv2D(
            64,
            (3,3),
            padding='same',
            activation='relu'
        )
    )

    model.add(

        layers.MaxPooling2D((2,2))
    )

    model.add(

        layers.BatchNormalization()
    )

    model.add(

        layers.Dropout(0.25)
    )

    # =====================================================
    # BLOCK 3
    # =====================================================

    model.add(

        layers.Conv2D(
            128,
            (3,3),
            padding='same',
            activation='relu'
        )
    )

    model.add(

        layers.Conv2D(
            128,
            (3,3),
            padding='same',
            activation='relu'
        )
    )

    model.add(

        layers.MaxPooling2D((2,2))
    )

    model.add(

        layers.BatchNormalization()
    )

    model.add(

        layers.Dropout(0.4)
    )

    # =====================================================
    # FLATTEN
    # =====================================================

    model.add(

        layers.Flatten()
    )

    # =====================================================
    # DENSE LAYER
    # =====================================================

    model.add(

        layers.Dense(
            256,
            activation='relu'
        )
    )

    model.add(

        layers.Dropout(0.5)
    )

    model.add(

        layers.Dense(
            128,
            activation='relu'
        )
    )

    model.add(

        layers.Dropout(0.3)
    )

    # =====================================================
    # OUTPUT LAYER
    # =====================================================

    model.add(

        layers.Dense(
            num_classes,
            activation='softmax'
        )
    )

    return model

# Bangun model
model = build_cnn_model(NUM_CLASSES)

print("\nModel berhasil dibangun!")

# Tampilkan summary
model.summary()

# =========================================================
# 11. KOMPILE MODEL
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']
)

print("\nModel berhasil dikompilasi!")

# =========================================================
# 12. CALLBACKS
# =========================================================

callbacks = [

    EarlyStopping(

        monitor='val_accuracy',

        patience=5,

        restore_best_weights=True,

        verbose=1
    ),

    ReduceLROnPlateau(

        monitor='val_loss',

        factor=0.5,

        patience=3,

        min_lr=1e-7,

        verbose=1
    )
]

# =========================================================
# 13. TRAINING MODEL
# =========================================================

EPOCHS = 10

print(f"\nMemulai training selama {EPOCHS} epoch...")

history = model.fit(

    ds_train_final,

    validation_data=ds_val_final,

    epochs=EPOCHS,

    callbacks=callbacks,

    verbose=1
)

# =========================================================
# 14. SAVE MODEL
# =========================================================

model.save("cnn_model.h5")

print("\nModel berhasil disimpan!")

# =========================================================
# 15. VISUALISASI ACCURACY & LOSS
# =========================================================

def plot_history(history):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14,5)
    )

    # Accuracy
    axes[0].plot(
        history.history['accuracy'],
        label='Training'
    )

    axes[0].plot(
        history.history['val_accuracy'],
        label='Validation'
    )

    axes[0].set_title('Model Accuracy')

    axes[0].set_xlabel('Epoch')

    axes[0].set_ylabel('Accuracy')

    axes[0].legend()

    # Loss
    axes[1].plot(
        history.history['loss'],
        label='Training'
    )

    axes[1].plot(
        history.history['val_loss'],
        label='Validation'
    )

    axes[1].set_title('Model Loss')

    axes[1].set_xlabel('Epoch')

    axes[1].set_ylabel('Loss')

    axes[1].legend()

    plt.tight_layout()

    plt.savefig(
        'training_history.png',
        dpi=100,
        bbox_inches='tight'
    )

    plt.show()

plot_history(history)

# =========================================================
# 16. EVALUASI MODEL
# =========================================================

test_loss, test_acc = model.evaluate(
    ds_test_final,
    verbose=0
)

print("\n" + "=" * 50)

print(f" Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")

print(f" Test Loss     : {test_loss:.4f}")

print("=" * 50)

# =========================================================
# 17. CONFUSION MATRIX
# =========================================================

y_pred = []
y_true = []

for images, labels in ds_test_final:

    preds = model.predict(
        images,
        verbose=0
    )

    y_pred.extend(
        np.argmax(preds, axis=1)
    )

    y_true.extend(
        labels.numpy()
    )

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(8,6))

sns.heatmap(

    cm,

    annot=True,

    fmt='d',

    cmap='Blues',

    xticklabels=CLASS_NAMES,

    yticklabels=CLASS_NAMES
)

plt.title('Confusion Matrix')

plt.xlabel('Predicted Label')

plt.ylabel('True Label')

plt.tight_layout()

plt.savefig(
    'confusion_matrix.png',
    dpi=100,
    bbox_inches='tight'
)

plt.show()

# =========================================================
# 18. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:\n")

print(

    classification_report(

        y_true,

        y_pred,

        target_names=CLASS_NAMES
    )
)

print("\nPROJECT CNN SELESAI!")