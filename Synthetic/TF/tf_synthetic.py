import os 
import time
import psutil
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import precision_recall_fscore_support

# ---------------------------------------------------------
# 1. SETUP & DATA LOADING
# --------------------------------------------------------- 
IMG_HEIGHT, IMG_WIDTH = 224, 224 
BATCH_SIZE = 32
DATASET_DIR = r"C:\Users\User\Documents\AI_Robotics Projects\AAU\Summer\Synthetic_Dataset_Classical"

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.3,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
], name="data_augmentation")

# ---------------------------------------------------------
# 2. ARCHITECTURE (MOBILENET V3-SMALL ONLY)
# ---------------------------------------------------------
def build_mobilenet_v3(num_classes=3):
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = data_augmentation(inputs)
    # Preprocessing layer explicitly included inside model graph
    x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
    
    base_model = tf.keras.applications.MobileNetV3Small(
        input_tensor=x,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Start frozen for initial head warm-up

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs=inputs, outputs=outputs), base_model

# ---------------------------------------------------------
# 3. TWO-STAGE TRAINING & FP16 QUANTIZATION PIPELINE
# ---------------------------------------------------------
results = {}
WARMUP_EPOCHS = 4
FINE_TUNE_EPOCHS = 10

print("\n================ Training MOBILENET_V3 ================")

model, base_model = build_mobilenet_v3()

# --- STAGE 1: Train Head Only ---
print("Stage 1: Warming up top classification head...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=WARMUP_EPOCHS
)

# --- STAGE 2: Fine-Tune Whole Network ---
print("Stage 2: Unfreezing base model for fine-tuning...")
base_model.trainable = True

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-4,
    decay_steps=FINE_TUNE_EPOCHS * len(train_ds)
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
]

model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=FINE_TUNE_EPOCHS, 
    callbacks=callbacks
)

keras_path = "mobilenet_v3.keras"
model.save(keras_path)
print(f" Saved original Keras model to '{keras_path}'")

print("Exporting FP16 TFLite variant for mobilenet_v3...")

# FP16 Export
conv_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
conv_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
conv_fp16.target_spec.supported_types = [tf.float16]
tflite_fp16 = conv_fp16.convert()

tflite_fp16_path = "mobilenet_v3_fp16.tflite"
with open(tflite_fp16_path, "wb") as f:
    f.write(tflite_fp16)

# Clean up session memory
del model, base_model
tf.keras.backend.clear_session()
gc.collect()

# ---------------------------------------------------------
# 4. BENCHMARKING EVALUATION
# ---------------------------------------------------------
def evaluate_tflite(tflite_path, dataset):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_true, y_pred, latencies = [], [], []
    proc = psutil.Process()
    ram_start = proc.memory_info().rss / (1024 * 1024)

    for images, labels in dataset:
        for i in range(len(images)):
            # Raw unscaled float image [0, 255] since model graph includes preprocessing
            img = tf.cast(images[i:i+1], tf.float32)

            interpreter.set_tensor(input_details[0]['index'], img)
            
            t0 = time.perf_counter()
            interpreter.invoke()
            latencies.append((time.perf_counter() - t0) * 1000)
            
            output_data = interpreter.get_tensor(output_details[0]['index'])

            y_pred.append(np.argmax(output_data[0]))
            y_true.append(labels[i].numpy())

    ram_end = proc.memory_info().rss / (1024 * 1024)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    
    del interpreter
    gc.collect()

    return {
        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
        "Size (MB)": os.path.getsize(tflite_path) / (1024 * 1024),
        "Avg Latency (ms)": np.mean(latencies),
        "RAM Usage (MB)": max(0, ram_end - ram_start)
    }

print("\nRunning Evaluation on MobileNetV3 FP16...")
results["mobilenet_v3_FP16"] = evaluate_tflite(tflite_fp16_path, val_ds)

# ---------------------------------------------------------
# 5. SAVE CSV & PLOT RESULTS
# ---------------------------------------------------------
csv_file = "synthetic_tf.csv"
df_results = pd.DataFrame.from_dict(results, orient='index')
df_results.index.name = "Model_Variant"

# Append to existing file if available; otherwise create new file with header
file_exists = os.path.exists(csv_file)
df_results.to_csv(csv_file, mode='a', header=not file_exists)
print(f"\n Metrics saved/appended to '{csv_file}' successfully!")

# Plot visualization
df_plot = df_results.reset_index().rename(columns={'index': 'Variant'})
df_plot['Architecture'] = 'MobileNetV3-Small'
df_plot['QuantType'] = 'FP16'

metrics = ["Accuracy", "Precision", "Recall", "F1", "Size (MB)", "Avg Latency (ms)", "RAM Usage (MB)"]

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 4, figsize=(24, 11))
axes = axes.flatten()

palettes = ["viridis", "plasma", "inferno", "magma", "cividis", "rocket", "mako"]

for idx, metric in enumerate(metrics):
    sns.barplot(
        data=df_plot, 
        x='Architecture', 
        y=metric, 
        hue='QuantType', 
        ax=axes[idx], 
        palette=palettes[idx]
    )
    axes[idx].set_title(f"Comparison: {metric}", fontsize=13, fontweight='bold')
    axes[idx].set_xlabel("")
    if metric in ["Accuracy", "Precision", "Recall", "F1"]:
        axes[idx].set_ylim([0, 1.05])

fig.delaxes(axes[7])

plt.tight_layout()
plt.savefig("mobilenet_v3_fp16_metrics.png", dpi=300)
plt.show()
print(" Saved plot to 'mobilenet_v3_fp16_metrics.png'")