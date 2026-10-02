import os 
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
DATASET_DIR = r"C:\Users\User\Documents\AI_Robotics Projects\AAU\Summer\Dataset_Simple"
MODEL_DIR = "."  # Directory containing .keras and .tflite files
OUTPUT_CSV = "crad_cam_results.csv"
IMAGE_OUTPUT_DIR = "crad_cam_images"

CLASSES = ["apple", "bottle", "cups"]
CLASS_TO_IDX = {"apple": 0, "bottle": 1, "cups": 2}

ARCHITECTURES = [
    "simple_cnn",
    "deeper_cnn",
    "complex_cnn",
    "mobilenet_v2",
    "mobilenet_v3",
]
PRECISIONS = ["fp32.tflite", "fp16.tflite", "int8.tflite"]


# ==========================================
# DYNAMIC RICH LAYER & SHAPE SELECTION
# ==========================================
def get_rich_conv_layer_name(model, min_spatial_dim=4):
    """
    Finds the deepest convolutional layer maintaining adequate spatial resolution
    (at least min_spatial_dim x min_spatial_dim) to avoid 1x1 feature maps.
    """
    for layer in reversed(model.layers):
        if isinstance(
            layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)
        ) or "conv" in layer.name.lower():
            out_shape = layer.output_shape
            if isinstance(out_shape, list):
                out_shape = out_shape[0]

            h, w = out_shape[1], out_shape[2]
            if (h is not None and h >= min_spatial_dim) and (
                w is not None and w >= min_spatial_dim
            ):
                return layer.name

    # Fallback to last conv layer if none met spatial criteria
    for layer in reversed(model.layers):
        if isinstance(
            layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)
        ) or "conv" in layer.name.lower():
            return layer.name

    raise ValueError(f"No convolutional layer found in model: {model.name}")


def get_model_input_size(model):
    """Extracts expected (height, width) tuple from model input shape."""
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    return (shape[1], shape[2])


# ==========================================
# DATASET LOADER (RAW [0, 255] FLOATS)
# ==========================================
def load_simple_dataset(dataset_dir, target_size=(128, 128)):
    dataset_samples = []

    for cls_name in CLASSES:
        cls_dir = os.path.join(dataset_dir, cls_name)
        if not os.path.exists(cls_dir):
            print(f"[Warning] Directory missing: {cls_dir}")
            continue

        valid_files = [
            f
            for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        if not valid_files:
            print(f"[Warning] No images found in: {cls_dir}")
            continue

        fname = sorted(valid_files)[0]
        img_path = os.path.join(cls_dir, fname)
        raw_img = cv2.imread(img_path)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        raw_img_resized = cv2.resize(raw_img, target_size)
        
        # Keep raw [0, 255] float values so embedded layers normalize correctly
        img_array = raw_img_resized.astype(np.float32)[np.newaxis, ...]

        label_idx = CLASS_TO_IDX[cls_name]
        dataset_samples.append({
            "img_array": img_array,
            "raw_img": raw_img_resized,
            "label_name": cls_name,
            "label_idx": label_idx,
            "filename": fname,
        })

    print(f"[Dataset Loaded] {len(dataset_samples)} images loaded.")
    return dataset_samples


# ==========================================
# CAM ENGINES WITH ROBUST NORMALIZATION
# ==========================================
def compute_keras_gradcam(model, img_array, target_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(target_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()

    # ReLU + 99th percentile normalization
    heatmap = np.maximum(heatmap, 0)
    p99 = np.percentile(heatmap, 99)
    if p99 > 0:
        heatmap = np.clip(heatmap / p99, 0, 1.0)
    else:
        max_val = np.max(heatmap)
        heatmap = heatmap / (max_val + 1e-8) if max_val > 0 else heatmap

    return heatmap, pred_index.numpy()


def compute_tflite_scorecam(
    tflite_path, keras_model, img_array, target_layer_name, top_k=16
):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    feat_extractor = tf.keras.models.Model(
        inputs=[keras_model.inputs],
        outputs=[keras_model.get_layer(target_layer_name).output],
    )
    feature_maps = feat_extractor(img_array)[0]

    input_data = prepare_tflite_input(img_array, input_details[0])
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    base_preds = interpreter.get_tensor(output_details[0]["index"])[0]
    target_class = np.argmax(base_preds)

    H, W, C = feature_maps.shape
    channel_means = tf.reduce_mean(feature_maps, axis=(0, 1)).numpy()
    top_channels = np.argsort(channel_means)[-min(top_k, C) :]

    cam = np.zeros((H, W), dtype=np.float32)

    for c in top_channels:
        fmap = feature_maps[:, :, c].numpy()
        mask = cv2.resize(fmap, (img_array.shape[2], img_array.shape[1]))
        denom = mask.max() - mask.min()
        mask = (mask - mask.min()) / (denom + 1e-8) if denom > 0 else mask

        masked_input = img_array * mask[np.newaxis, ..., np.newaxis]
        masked_tflite_input = prepare_tflite_input(
            masked_input, input_details[0]
        )

        interpreter.set_tensor(input_details[0]["index"], masked_tflite_input)
        interpreter.invoke()
        score = interpreter.get_tensor(output_details[0]["index"])[0][
            target_class
        ]

        cam += score * fmap

    cam = np.maximum(cam, 0)
    p99 = np.percentile(cam, 99)
    if p99 > 0:
        cam = np.clip(cam / p99, 0, 1.0)
    else:
        cam = cam / (cam.max() + 1e-8) if cam.max() > 0 else cam

    return cam, target_class


def prepare_tflite_input(img_array, input_detail):
    if input_detail["dtype"] == np.int8:
        scale, zero_point = input_detail["quantization"]
        if scale > 0:
            return (img_array / scale + zero_point).astype(np.int8)
        return img_array.astype(np.int8)
    elif input_detail["dtype"] == np.float32:
        return img_array.astype(np.float32)
    return img_array


# ==========================================
# METRICS & HIGH-CONTRAST OVERLAY
# ==========================================
def compute_topk_iou(map1, map2, top_k_percent=0.20):
    k1 = max(1, int(map1.size * top_k_percent))
    k2 = max(1, int(map2.size * top_k_percent))

    thresh1 = np.sort(map1.ravel())[-k1]
    thresh2 = np.sort(map2.ravel())[-k2]

    bin1 = map1 >= thresh1
    bin2 = map2 >= thresh2

    intersection = np.logical_and(bin1, bin2).sum()
    union = np.logical_or(bin1, bin2).sum()
    return intersection / (union + 1e-8)


def overlay_heatmap(heatmap, raw_img, threshold=0.15, alpha=0.5):
    heatmap_resized = cv2.resize(heatmap, (raw_img.shape[1], raw_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    active_mask = heatmap_resized > threshold
    output = raw_img.copy().astype(np.float32)

    output[active_mask] = (
        raw_img[active_mask] * (1 - alpha) + heatmap_colored[active_mask] * alpha
    )

    return np.uint8(np.clip(output, 0, 255))


# ==========================================
# EXECUTION PIPELINE
# ==========================================
def run_crad_cam_experiment():
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    results = []

    for arch in ARCHITECTURES:
        keras_path = os.path.join(MODEL_DIR, f"{arch}.keras")
        if not os.path.exists(keras_path):
            print(f"[Warning] Missing baseline: {keras_path}. Skipping.")
            continue

        print(f"\n---> Evaluating Architecture: {arch}")
        keras_model = tf.keras.models.load_model(keras_path)

        conv_layer = get_rich_conv_layer_name(keras_model, min_spatial_dim=4)
        target_size = get_model_input_size(keras_model)
        print(f"     Rich Feature Layer selected: '{conv_layer}'")
        print(f"     Target input shape: {target_size}")

        dataset_samples = load_simple_dataset(
            DATASET_DIR, target_size=target_size
        )

        fig, axes = plt.subplots(3, 4, figsize=(18, 12))
        fig.suptitle(
            f"GRAD-CAM vs Score-CAM (Rich Feature Maps): {arch}",
            fontsize=16,
            fontweight="bold",
        )

        for sample_idx, sample in enumerate(dataset_samples):
            img_array = sample["img_array"]
            raw_img = sample["raw_img"]
            gt_idx = sample["label_idx"]
            gt_name = sample["label_name"]
            fname = sample["filename"]

            # Baseline GRAD-CAM
            keras_cam, keras_class = compute_keras_gradcam(
                keras_model, img_array, conv_layer
            )
            keras_vis = overlay_heatmap(keras_cam, raw_img)

            axes[sample_idx, 0].imshow(keras_vis)
            axes[sample_idx, 0].set_title(
                f"Keras FP32 Baseline\nGT: {gt_name} | Pred: {CLASSES[keras_class]}"
            )
            axes[sample_idx, 0].axis("off")

            # Quantized Variants
            for prec_idx, prec in enumerate(PRECISIONS):
                col_idx = prec_idx + 1
                tflite_filename = f"{arch}_{prec}"
                tflite_path = os.path.join(MODEL_DIR, tflite_filename)
                precision_label = prec.replace(".tflite", "").upper()

                if not os.path.exists(tflite_path):
                    axes[sample_idx, col_idx].text(
                        0.5,
                        0.5,
                        f"Missing\n{tflite_filename}",
                        ha="center",
                        va="center",
                    )
                    axes[sample_idx, col_idx].axis("off")
                    continue

                tflite_cam, tflite_class = compute_tflite_scorecam(
                    tflite_path, keras_model, img_array, conv_layer
                )

                if tflite_cam.shape != keras_cam.shape:
                    tflite_cam = cv2.resize(
                        tflite_cam, (keras_cam.shape[1], keras_cam.shape[0])
                    )

                ssim_score = ssim(keras_cam, tflite_cam, data_range=1.0)
                iou_score = compute_topk_iou(keras_cam, tflite_cam)

                if np.all(keras_cam == keras_cam[0, 0]) or np.all(
                    tflite_cam == tflite_cam[0, 0]
                ):
                    spearman_corr = 0.0
                else:
                    spearman_corr, _ = spearmanr(
                        keras_cam.ravel(), tflite_cam.ravel()
                    )
                    if np.isnan(spearman_corr):
                        spearman_corr = 0.0

                keras_correct = keras_class == gt_idx
                tflite_correct = tflite_class == gt_idx

                if keras_correct and tflite_correct:
                    pred_status = "Both Correct"
                elif keras_correct and not tflite_correct:
                    pred_status = "FP32 Correct / Quantized Failed"
                elif not keras_correct and tflite_correct:
                    pred_status = "FP32 Failed / Quantized Correct"
                else:
                    pred_status = "Both Incorrect"

                results.append({
                    "architecture": arch,
                    "precision": precision_label.lower(),
                    "image_filename": fname,
                    "ground_truth_class": gt_name,
                    "keras_predicted_class": CLASSES[keras_class],
                    "tflite_predicted_class": CLASSES[tflite_class],
                    "prediction_status": pred_status,
                    "ssim": ssim_score,
                    "topk_iou": iou_score,
                    "spearman_consistency": spearman_corr,
                })

                tflite_vis = overlay_heatmap(tflite_cam, raw_img)
                axes[sample_idx, col_idx].imshow(tflite_vis)
                axes[sample_idx, col_idx].set_title(
                    f"TFLite {precision_label} Score-CAM\nPred: {CLASSES[tflite_class]} | IoU: {iou_score:.2f}"
                )
                axes[sample_idx, col_idx].axis("off")

        plt.tight_layout()
        plot_save_path = os.path.join(
            IMAGE_OUTPUT_DIR, f"{arch}_rich_cam_grid.png"
        )
        plt.savefig(plot_save_path, bbox_inches="tight")
        plt.close()
        print(f"     Saved figure: {plot_save_path}")

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"\n[Done] Pipeline complete. Saved results to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_crad_cam_experiment()