from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO(r"C:\Users\User\Documents\AI_Robotics Projects\AAU\Summer\Dataset_YOLO\runs\detect\train\weights\best.pt")

# Run validation using the real-world dataset's data.yaml file
metrics = model.val(data=r"C:\Users\User\Documents\AI_Robotics Projects\AAU\Summer\data.yaml")

# Extract primary evaluation metrics
map50 = metrics.box.map50
map50_95 = metrics.box.map
precision = metrics.box.mp
recall = metrics.box.mr
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

# Print formatted results matching the specified evaluation structure
print("\n--- Real-World Performance Evaluation ---")
print(f"Real-World mAP@50 Evaluation: {map50:.4f}")
print(f"Real-World mAP@50-95 Evaluation: {map50_95:.4f}")
print(f"Real-World F1-Score Evaluation: {f1_score:.4f}")
print(f"Real-World Precision & Recall: Precision = {precision:.2f}, Recall = {recall:.2f}")