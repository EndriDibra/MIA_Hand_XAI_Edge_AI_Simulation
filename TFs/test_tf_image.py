import os 
import cv2
import numpy as np
import tensorflow as tf

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Update these paths to match your Windows directory layout
MODEL_PATH = "mobilenet_v3.keras"  # or "simple_cnn.keras" / "complex_cnn_fp16.tflite"
IS_CUSTOM_CNN = False              # Set to True for your Custom CNN, False for MobileNetV3

CLASS_NAMES = ["apple", "bottle", "cups"]

# Provide full or relative Windows paths to your test images
TEST_IMAGE_PATHS = [
    r"apple_0360.png",
    r"bottle_0011.png",
    r"cup_0320.png"
]
# ==============================================================================


def load_runner(model_path, is_custom_cnn):
    """Loads either Keras (.keras) or TFLite (.tflite) model."""
    if model_path.endswith(".tflite"):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return ("tflite", interpreter)
    else:
        model = tf.keras.models.load_model(model_path)
        return ("keras", model)


def predict_image(runner, image, is_custom_cnn):
    model_type, model_obj = runner

    # Standardize input dimensions
    if is_custom_cnn:
        target_h, target_w = 128, 128
    else:
        target_h, target_w = 224, 224  # Standard MobileNet resolution (or auto-detected)

    # Preprocess
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (target_w, target_h))
    tensor = np.expand_dims(resized.astype(np.float32), axis=0)

    # Run Inference
    if model_type == "tflite":
        input_details = model_obj.get_input_details()
        output_details = model_obj.get_output_details()
        
        # Check INT8 Quantization
        if input_details[0]['dtype'] == np.int8:
            scale, zero_point = input_details[0]['quantization']
            tensor = (tensor / scale + zero_point).astype(np.int8) if scale > 0 else tensor.astype(np.int8)

        model_obj.set_tensor(input_details[0]['index'], tensor)
        model_obj.invoke()
        output = model_obj.get_tensor(output_details[0]['index'])

        if input_details[0]['dtype'] == np.int8:
            out_scale, out_zero_point = output_details[0]['quantization']
            if out_scale > 0:
                output = (output.astype(np.float32) - out_zero_point) * out_scale
    else:
        output = model_obj.predict(tensor, verbose=0)

    # Postprocess probabilities
    output = np.squeeze(output)
    if not np.allclose(np.sum(output), 1.0, atol=1e-2):
        exp_out = np.exp(output - np.max(output))
        output = exp_out / np.sum(exp_out)

    pred_idx = np.argmax(output)
    return CLASS_NAMES[pred_idx], output[pred_idx]


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file missing: {MODEL_PATH}")
        return

    runner = load_runner(MODEL_PATH, IS_CUSTOM_CNN)
    print(f"[INFO] Successfully loaded model: {MODEL_PATH}")
    print("[INFO] Press any key in the GUI window to step to the next image (or 'q' to quit).\n")

    for img_path in TEST_IMAGE_PATHS:
        if not os.path.exists(img_path):
            print(f"[SKIP] Image not found: {img_path}")
            continue

        frame = cv2.imread(img_path)
        if frame is None:
            continue

        pred_class, confidence = predict_image(runner, frame, IS_CUSTOM_CNN)
        print(f"File: {os.path.basename(img_path)} | Prediction: {pred_class} ({confidence * 100:.1f}%)")

        # Overlay text on window
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)
        cv2.putText(frame, f"Pred: {pred_class} ({confidence * 100:.1f}%)", (15, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Windows Model Test", frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()