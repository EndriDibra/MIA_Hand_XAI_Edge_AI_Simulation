import os 
import gc
import time
import psutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import precision_recall_fscore_support

# ------------------------------------------------------------------------------
# 1. Dataset Setup
# ------------------------------------------------------------------------------
DATASET_DIR = r"C:\Users\User\Documents\AI_Robotics Projects\AAU\Summer\Synthetic_Dataset_Classical"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

num_classes = len(train_ds.class_names)

# ------------------------------------------------------------------------------
# 2. Complex CNN Architecture (With Data Augmentation Inside)
# ------------------------------------------------------------------------------
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

def build_complex_cnn(input_shape=(128, 128, 3), num_classes=num_classes):
    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="Complex_CNN")

# ------------------------------------------------------------------------------
# 3. Model Training
# ------------------------------------------------------------------------------
model = build_complex_cnn()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=10)

# ------------------------------------------------------------------------------
# 4. Export FP16 TFLite Only
# ------------------------------------------------------------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_fp16_model = converter.convert()

tflite_path = "complex_cnn_fp16.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_fp16_model)

print(f"\nFP16 TFLite model successfully saved to: {tflite_path}")

# ------------------------------------------------------------------------------
# 5. Evaluation Loop
# ------------------------------------------------------------------------------
def evaluate_tflite_fp16(tflite_model_path, dataset):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_true, y_pred, latencies = [], [], []
    proc = psutil.Process()
    ram_start = proc.memory_info().rss / (1024 * 1024)

    for images, labels in dataset:
        for i in range(len(images)):
            # Pass unscaled float32 images (Rescaling layer inside graph handles division)
            img = tf.cast(images[i:i+1], tf.float32)

            interpreter.set_tensor(input_details[0]['index'], img)
            
            t0 = time.perf_counter()
            interpreter.invoke()
            latencies.append((time.perf_counter() - t0) * 1000)

            output_data = interpreter.get_tensor(output_details[0]['index'])
            y_pred.append(np.argmax(output_data[0]))
            y_true.append(labels[i].numpy())

    ram_end = proc.memory_info().rss / (1024 * 1024)
    
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    
    del interpreter
    gc.collect()

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Size (MB)": os.path.getsize(tflite_model_path) / (1024 * 1024),
        "Avg Latency (ms)": np.mean(latencies),
        "RAM Usage (MB)": max(0, ram_end - ram_start)
    }

# Compute Metrics
metrics = evaluate_tflite_fp16(tflite_path, val_ds)

print("\n--- Benchmark Results for Complex CNN (FP16 Quantized) ---")
for key, val in metrics.items():
    print(f"{key}: {val:.4f}")

# ------------------------------------------------------------------------------
# 6. Save Metrics to CSV (synthetic_tf.csv)
# ------------------------------------------------------------------------------
csv_file = "synthetic_tf.csv"
variant_name = "complex_cnn_FP16"

# Structuring metrics to DataFrame
df_metrics = pd.DataFrame([metrics], index=[variant_name])
df_metrics.index.name = "Model_Variant"

# Append to existing file if available; otherwise create new file with header
file_exists = os.path.exists(csv_file)
df_metrics.to_csv(csv_file, mode='a', header=not file_exists)

print(f"\n Metrics appended to '{csv_file}' successfully!")