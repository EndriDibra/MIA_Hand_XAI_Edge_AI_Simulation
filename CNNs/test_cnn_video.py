import os 
import cv2
import numpy as np
import tensorflow as tf

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Examples: "simple_cnn.keras", "complex_cnn_fp32.tflite", "deeper_cnn_int8.tflite"
SELECTED_MODEL = "complex_cnn_fp16.tflite"
CLASS_NAMES = ["apple", "bottle", "cups"]
VIDEO_SOURCE = 0
IMG_HEIGHT, IMG_WIDTH = 128, 128
# ==============================================================================


class CNNModelRunner:
    """Unified wrapper for Custom CNN models with built-in Rescaling(1./255)."""
    def __init__(self, model_path):
        self.model_path = model_path
        self.is_tflite = model_path.endswith(".tflite")
        
        if self.is_tflite:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_shape = self.input_details[0]['shape']
            self.is_int8 = (self.input_details[0]['dtype'] == np.int8)
        else:
            self.model = tf.keras.models.load_model(model_path)
            self.input_shape = self.model.input_shape

    def preprocess(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_WIDTH, IMG_HEIGHT))
        
        # Raw float [0, 255] because layers.Rescaling(1./255) is in the model graph
        float_frame = resized.astype(np.float32)

        return np.expand_dims(float_frame, axis=0)

    def predict(self, frame):
        tensor = self.preprocess(frame)

        if self.is_tflite:
            if self.is_int8:
                scale, zero_point = self.input_details[0]['quantization']
                if scale > 0:
                    # Map [0, 255] raw floats into INT8 quantization bounds
                    tensor = (tensor / scale + zero_point).astype(np.int8)
                else:
                    tensor = tensor.astype(np.int8)

            self.interpreter.set_tensor(self.input_details[0]['index'], tensor)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])

            if self.is_int8:
                out_scale, out_zero_point = self.output_details[0]['quantization']
                if out_scale > 0:
                    output = (output.astype(np.float32) - out_zero_point) * out_scale
        else:
            output = self.model.predict(tensor, verbose=0)

        output = np.squeeze(output)

        # Apply Softmax manually if raw unnormalized logits are returned
        if not np.allclose(np.sum(output), 1.0, atol=1e-2):
            exp_out = np.exp(output - np.max(output))
            output = exp_out / np.sum(exp_out)

        pred_class_idx = np.argmax(output)
        confidence = output[pred_class_idx]
        class_name = CLASS_NAMES[pred_class_idx] if pred_class_idx < len(CLASS_NAMES) else f"Class {pred_class_idx}"

        return class_name, confidence


def main():
    if not os.path.exists(SELECTED_MODEL):
        print(f"[ERROR] Model file not found: {SELECTED_MODEL}")
        return

    print(f"[INFO] Initializing CNN Live Classifier with: {SELECTED_MODEL}")
    try:
        runner = CNNModelRunner(SELECTED_MODEL)
    except Exception as e:
        print(f"[ERROR] Failed to load model {SELECTED_MODEL}: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open video source: {VIDEO_SOURCE}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = cv2.getTickCount()
        class_name, confidence = runner.predict(frame)

        time_elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        fps = 1.0 / time_elapsed if time_elapsed > 0 else 0.0

        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)
        cv2.putText(frame, f"Model: {SELECTED_MODEL}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Pred: {class_name} ({confidence * 100:.1f}%)", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow("Custom CNN Live Classifier", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()