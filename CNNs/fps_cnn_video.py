import os 
import csv
import time
import cv2
import numpy as np
import tensorflow as tf

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODELS_TO_BENCHMARK = [
    "simple_cnn.keras",
    "simple_cnn_fp32.tflite",
    "simple_cnn_fp16.tflite",
    "simple_cnn_int8.tflite",
    "complex_cnn.keras",
    "complex_cnn_fp32.tflite",
    "complex_cnn_fp16.tflite",
    "complex_cnn_int8.tflite",
    "deeper_cnn.keras",
    "deeper_cnn_fp32.tflite",
    "deeper_cnn_fp16.tflite",
    "deeper_cnn_int8.tflite"
]

TEST_DURATION_SEC = 10.0                 # Benchmark duration per model
OUTPUT_CSV = "fps_results.csv"           # Output CSV file name
VIDEO_SOURCE = 0                         # 0 for webcam, or video path string
NORMALIZE_INPUT = True                   # True for [0.0, 1.0], False for [0, 255]
# ==============================================================================


class GenericModelRunner:
    """Unified wrapper for executing both Keras and TFLite inference."""
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
        target_h, target_w = self.input_shape[1], self.input_shape[2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (target_w, target_h))
        float_frame = resized.astype(np.float32)

        if NORMALIZE_INPUT:
            float_frame /= 255.0

        return np.expand_dims(float_frame, axis=0)

    def predict(self, frame):
        tensor = self.preprocess(frame)

        if self.is_tflite:
            if self.is_int8:
                scale, zero_point = self.input_details[0]['quantization']
                tensor = (tensor / scale + zero_point).astype(np.int8)

            self.interpreter.set_tensor(self.input_details[0]['index'], tensor)
            self.interpreter.invoke()
            _ = self.interpreter.get_tensor(self.output_details[0]['index'])
        else:
            _ = self.model.predict(tensor, verbose=0)


def benchmark_single_model(model_name, cap):
    """Runs a 10-second stream benchmark for a specific model."""
    print(f"\n[INFO] Initializing: {model_name}")
    try:
        runner = GenericModelRunner(model_name)
    except Exception as e:
        print(f"[ERROR] Failed to load {model_name}: {e}")
        return None, None

    frame_count = 0
    start_time = time.time()
    end_time = start_time + TEST_DURATION_SEC

    # Warmup frame to compile execution graphs or allocation buffers
    ret, frame = cap.read()
    if ret:
        runner.predict(frame)

    print(f"[RUNNING] Benchmarking for {TEST_DURATION_SEC} seconds...")

    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Video stream interrupted.")
            break

        runner.predict(frame)
        frame_count += 1

        elapsed = time.time() - start_time
        remaining = max(0.0, TEST_DURATION_SEC - elapsed)
        current_fps = frame_count / elapsed if elapsed > 0 else 0.0

        # HUD Visual Overlay
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
        cv2.putText(frame, f"Testing: {model_name}", (15, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {current_fps:.1f} | Time Left: {remaining:.1f}s", (15, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("CNN Benchmark Engine", frame)

        # Allow user to break manually by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[USER INTERRUPT] Exiting current model benchmark early.")
            break

    total_elapsed = time.time() - start_time
    avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0.0
    
    # Calculate average latency per frame in milliseconds
    avg_latency_ms = (total_elapsed / frame_count) * 1000.0 if frame_count > 0 else 0.0

    print(f"[COMPLETE] {model_name} -> Avg FPS: {avg_fps:.2f} | Avg Latency: {avg_latency_ms:.2f} ms")
    return avg_fps, avg_latency_ms


def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open video source: {VIDEO_SOURCE}")
        return

    benchmark_results = []

    print("==================================================")
    print("      AUTOMATED CNN FPS BENCHMARK SUITE          ")
    print("==================================================")

    for model_name in MODELS_TO_BENCHMARK:
        if not os.path.exists(model_name):
            print(f"[SKIP] File not found: {model_name}")
            benchmark_results.append({
                "model_name": model_name,
                "status": "NOT FOUND",
                "avg_fps": "N/A",
                "avg_latency_ms": "N/A"
            })
            continue

        avg_fps, avg_latency_ms = benchmark_single_model(model_name, cap)

        if avg_fps is not None:
            benchmark_results.append({
                "model_name": model_name,
                "status": "SUCCESS",
                "avg_fps": f"{avg_fps:.2f}",
                "avg_latency_ms": f"{avg_latency_ms:.2f}"
            })
        else:
            benchmark_results.append({
                "model_name": model_name,
                "status": "LOAD FAILED",
                "avg_fps": "N/A",
                "avg_latency_ms": "N/A"
            })

    cap.release()
    cv2.destroyAllWindows()

    # Save findings to CSV
    with open(OUTPUT_CSV, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["model_name", "status", "avg_fps", "avg_latency_ms"])
        writer.writeheader()
        writer.writerows(benchmark_results)

    print("\n==================================================")
    print(f"[SUCCESS] Results exported to '{OUTPUT_CSV}'")
    print("==================================================")


if __name__ == "__main__":
    main()