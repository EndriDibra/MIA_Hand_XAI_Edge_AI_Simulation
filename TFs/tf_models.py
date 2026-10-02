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
DATASET_DIR = r"C:\\Users\\User\\Documents\\AI_Robotics Projects\\AAU\\Summer\\Dataset_Classical"

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
# 2. ARCHITECTURES (MOBILENET V2 & V3-SMALL)
# ---------------------------------------------------------
def build_mobilenet_v2(num_classes=3):
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = data_augmentation(inputs)
    # Scaled internally from [0, 255] -> [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    
    base_model = tf.keras.applications.MobileNetV2(
        input_tensor=x,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Start frozen for initial head warm-up

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs=inputs, outputs=outputs), base_model

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

# Calibration dataset generator for INT8 quantization
def representative_dataset():
    for images, _ in train_ds.take(30):
        for i in range(images.shape[0]):
            img = tf.cast(images[i:i+1], tf.float32)
            yield [img]

# ---------------------------------------------------------
# 3. TWO-STAGE TRAINING & QUANTIZATION PIPELINE
# ---------------------------------------------------------
results = {}
arch_builders = {
    "mobilenet_v2": build_mobilenet_v2,
    "mobilenet_v3": build_mobilenet_v3
}

WARMUP_EPOCHS = 4
FINE_TUNE_EPOCHS = 20

for name, builder_func in arch_builders.items():
    print(f"\n================ Training {name.upper()} ================")
    
    model, base_model = builder_func()
    
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
    
    keras_path = f"{name}.keras"
    model.save(keras_path)
    print(f" Saved original Keras model to '{keras_path}'")
    
    print(f"Exporting TFLite variants for {name}...")

    # FP32 Export
    conv_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = conv_fp32.convert()
    with open(f"{name}_fp32.tflite", "wb") as f:
        f.write(tflite_fp32)
        
    # FP16 Export
    conv_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
    conv_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    conv_fp16.target_spec.supported_types = [tf.float16]
    tflite_fp16 = conv_fp16.convert()
    with open(f"{name}_fp16.tflite", "wb") as f:
        f.write(tflite_fp16)
        
    # INT8 Quantization with Depthwise Guardrails
    conv_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
    conv_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    conv_int8.representative_dataset = representative_dataset
    # Allow float fallback for depthwise convolutions sensitive to weight clipping
    conv_int8.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    conv_int8.inference_input_type = tf.int8
    conv_int8.inference_output_type = tf.int8
    
    try:
        tflite_int8 = conv_int8.convert()
        with open(f"{name}_int8.tflite", "wb") as f:
            f.write(tflite_int8)
    except Exception as e:
        print(f"INT8 conversion failed for {name}: {e}")

    # Clean up session memory
    del model, base_model
    tf.keras.backend.clear_session()
    gc.collect()

# ---------------------------------------------------------
# 4. BENCHMARKING EVALUATION
# ---------------------------------------------------------
def evaluate_keras(model_path, dataset):
    model = tf.keras.models.load_model(model_path)
    y_true, y_pred, latencies = [], [], []
    proc = psutil.Process()
    ram_start = proc.memory_info().rss / (1024 * 1024)

    for images, labels in dataset:
        for i in range(len(images)):
            img = images[i:i+1]
            t0 = time.perf_counter()
            preds = model.predict(img, verbose=0)
            latencies.append((time.perf_counter() - t0) * 1000)
            y_pred.append(np.argmax(preds[0]))
            y_true.append(labels[i].numpy())

    ram_end = proc.memory_info().rss / (1024 * 1024)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    
    del model
    tf.keras.backend.clear_session()
    gc.collect()

    return {
        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
        "Size (MB)": os.path.getsize(model_path) / (1024 * 1024),
        "Avg Latency (ms)": np.mean(latencies),
        "RAM Usage (MB)": max(0, ram_end - ram_start)
    }

def evaluate_tflite(tflite_path, dataset):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    is_int8 = input_details[0]['dtype'] == np.int8
    
    if is_int8:
        input_scale, input_zero_point = input_details[0]['quantization']
        output_scale, output_zero_point = output_details[0]['quantization']

    y_true, y_pred, latencies = [], [], []
    proc = psutil.Process()
    ram_start = proc.memory_info().rss / (1024 * 1024)

    for images, labels in dataset:
        for i in range(len(images)):
            # Raw unscaled float image [0, 255] since model graph includes preprocessing
            img = tf.cast(images[i:i+1], tf.float32)
            
            if is_int8:
                # Direct integer mapping without double normalization
                img = (img / input_scale) + input_zero_point
                img = tf.cast(img, tf.int8)

            interpreter.set_tensor(input_details[0]['index'], img)
            
            t0 = time.perf_counter()
            interpreter.invoke()
            latencies.append((time.perf_counter() - t0) * 1000)
            
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            if is_int8:
                output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

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

print("\nRunning Evaluation on MobileNet Variants...")
for name in ["mobilenet_v2", "mobilenet_v3"]:
    results[f"{name}_Original_Keras"] = evaluate_keras(f"{name}.keras", val_ds)
    results[f"{name}_FP32"] = evaluate_tflite(f"{name}_fp32.tflite", val_ds)
    results[f"{name}_FP16"] = evaluate_tflite(f"{name}_fp16.tflite", val_ds)
    if os.path.exists(f"{name}_int8.tflite"):
        results[f"{name}_INT8"] = evaluate_tflite(f"{name}_int8.tflite", val_ds)

# ---------------------------------------------------------
# 5. SAVE CSV & PLOT COMPARISONS
# ---------------------------------------------------------
df_results = pd.DataFrame.from_dict(results, orient='index')
df_results.to_csv("mobilenet_comparison_results.csv", index_label="Model_Variant")
print("\n Saved results to 'mobilenet_comparison_results.csv'")

df_plot = df_results.reset_index().rename(columns={'index': 'Variant'})
df_plot['Architecture'] = df_plot['Variant'].apply(lambda x: 'MobileNetV2' if 'v2' in x else 'MobileNetV3-Small')

# Fixed column name collision: using 'QuantType' instead of 'Precision'
df_plot['QuantType'] = df_plot['Variant'].apply(
    lambda x: 'Keras' if 'Keras' in x else ('FP32' if 'FP32' in x else ('FP16' if 'FP16' in x else 'INT8'))
)

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
plt.savefig("mobilenet_metrics_comparison.png", dpi=300)
plt.show()
print(" Saved plots to 'mobilenet_metrics_comparison.png'")