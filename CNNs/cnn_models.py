import os 
import time
import psutil
import gc 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import precision_recall_fscore_support

# ---------------------------------------------------------
# 1. SETUP & DATA LOADING
# --------------------------------------------------------- 
IMG_HEIGHT, IMG_WIDTH = 128, 128 
BATCH_SIZE = 32
DATASET_DIR = "C:\\Users\\User\\Documents\\AI_Robotics Projects\\AAU\\Summer\\Dataset_Classical"

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
])

# ---------------------------------------------------------
# 2. ARCHITECTURES (SIMPLE, COMPLEX & DEEPER RES-CNN)
# ---------------------------------------------------------
def build_simple_cnn(num_classes=3):
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        data_augmentation,
        layers.Rescaling(1./255),
        
        layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.MaxPooling2D(),
        
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.MaxPooling2D(),
        
        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.MaxPooling2D(),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_complex_cnn(num_classes=3):
    """Stacked Conv blocks stabilized with BatchNormalization."""
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        data_augmentation,
        layers.Rescaling(1./255),
        
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(momentum=0.9),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(momentum=0.9),
        layers.Activation('relu'),
        layers.MaxPooling2D(),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(momentum=0.9),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(momentum=0.9),
        layers.Activation('relu'),
        layers.MaxPooling2D(),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(momentum=0.9),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(momentum=0.9),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(),

        # Classifier Head
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_deeper_res_cnn(num_classes=3):
    """4-Block Residual CNN with Skip Connections to evaluate depth capacity."""
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)

    def res_block(tensor, filters):
        x = layers.Conv2D(filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4))(tensor)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        
        # Shortcut projection if channel dimension changes
        if tensor.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), padding='same')(tensor)
            shortcut = layers.BatchNormalization(momentum=0.9)(shortcut)
        else:
            shortcut = tensor
            
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return layers.MaxPooling2D()(x)

    x = res_block(x, 32)   # Block 1
    x = res_block(x, 64)   # Block 2
    x = res_block(x, 128)  # Block 3
    x = res_block(x, 256)  # Block 4

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs)

# Dynamic calibration dataset generator for INT8 quantization
def representative_dataset():
    for images, _ in train_ds.take(20):
        for img in images:
            norm_img = tf.cast(img, tf.float32) / 255.0
            yield [tf.expand_dims(norm_img, axis=0)]

# ---------------------------------------------------------
# 3. TRAINING & QUANTIZATION PIPELINE
# ---------------------------------------------------------
results = {}
arch_builders = {
    "simple_cnn": build_simple_cnn,
    "complex_cnn": build_complex_cnn,
    "deeper_cnn": build_deeper_res_cnn
}

EPOCHS = 25

for name, builder_func in arch_builders.items():
    print(f"\n================ Training {name.upper()} ================")
    
    model = builder_func()
    
    # Moderate LR (3e-4) to prevent BN explosion in Complex and Deeper models
    init_lr = 3e-4
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=init_lr,
        decay_steps=EPOCHS * len(train_ds)
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=7, 
            restore_best_weights=True
        )
    ]
    
    model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=EPOCHS, 
        callbacks=callbacks
    )
    
    keras_path = f"{name}.keras"
    model.save(keras_path)
    print(f" Saved original Keras model to '{keras_path}'")
    
    print(f"Exporting TFLite variants for {name}...")

    # FP32
    conv_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = conv_fp32.convert()
    with open(f"{name}_fp32.tflite", "wb") as f:
        f.write(tflite_fp32)
        
    # FP16
    conv_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
    conv_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    conv_fp16.target_spec.supported_types = [tf.float16]
    tflite_fp16 = conv_fp16.convert()
    with open(f"{name}_fp16.tflite", "wb") as f:
        f.write(tflite_fp16)
        
    # Fully Quantized INT8
    conv_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
    conv_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    conv_int8.representative_dataset = representative_dataset
    conv_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv_int8.inference_input_type = tf.int8
    conv_int8.inference_output_type = tf.int8
    tflite_int8 = conv_int8.convert()
    with open(f"{name}_int8.tflite", "wb") as f:
        f.write(tflite_int8)

    # Clean up graph and RAM before starting next training run
    del model
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
            img = tf.cast(images[i:i+1], tf.float32)
            
            if is_int8:
                img = (img / 255.0) / input_scale + input_zero_point
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

print("\nRunning Evaluation on All Model Variants...")
for name in ["simple_cnn", "complex_cnn", "deeper_cnn"]:
    results[f"{name}_Original_Keras"] = evaluate_keras(f"{name}.keras", val_ds)
    results[f"{name}_FP32"] = evaluate_tflite(f"{name}_fp32.tflite", val_ds)
    results[f"{name}_FP16"] = evaluate_tflite(f"{name}_fp16.tflite", val_ds)
    results[f"{name}_INT8"] = evaluate_tflite(f"{name}_int8.tflite", val_ds)

# ---------------------------------------------------------
# 5. SAVE CSV & PLOT ALL METRICS
# ---------------------------------------------------------
df_results = pd.DataFrame.from_dict(results, orient='index')
df_results.to_csv("cnns_results.csv", index_label="Model_Variant")
print("\n Saved complete results to 'cnns_results.csv'")

df_plot = df_results.reset_index().rename(columns={'index': 'Variant'})

def map_arch(x):
    if 'simple' in x: return 'Simple CNN'
    elif 'complex' in x: return 'Complex CNN'
    else: return 'Deeper Res-CNN'

df_plot['Architecture'] = df_plot['Variant'].apply(map_arch)
df_plot['Precision'] = df_plot['Variant'].apply(
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
        hue='Precision', 
        ax=axes[idx], 
        palette=palettes[idx]
    )
    axes[idx].set_title(f"Comparison: {metric}", fontsize=13, fontweight='bold')
    axes[idx].set_xlabel("")
    if metric in ["Accuracy", "Precision", "Recall", "F1"]:
        axes[idx].set_ylim([0, 1.05])

fig.delaxes(axes[7])

plt.tight_layout()
plt.savefig("all_metrics_bar_plots.png", dpi=300)
plt.show()
print(" Saved updated metric bar plots to 'all_metrics_bar_plots.png'")