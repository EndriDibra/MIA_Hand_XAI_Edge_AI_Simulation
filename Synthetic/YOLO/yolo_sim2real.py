import numpy as np 
import pandas as pd
from ultralytics import YOLO

# 1. Load fine-tuned synthetic model
model = YOLO("C:/Users/User/Documents/AI_Robotics Projects/AAU/Summer/Synthetic_Dataset_YOLO/runs/detect/train/weights/best.pt")

# 2. Run validation against the real-world validation set
metrics = model.val(data="data_2.yaml", split="val")

# 3. Extract Aggregated Metrics & Latency
precision = float(metrics.box.mp)
recall = float(metrics.box.mr)
map50 = float(metrics.box.map50)
map50_95 = float(metrics.box.map)
fitness = float(metrics.fitness)
f1_score = (2 * precision * recall) / (precision + recall + 1e-16)

speed = metrics.speed
prep_ms = float(speed.get('preprocess', 0.0))
inf_ms = float(speed.get('inference', 0.0))
post_ms = float(speed.get('postprocess', 0.0))
total_ms = prep_ms + inf_ms + post_ms
fps = 1000.0 / total_ms if total_ms > 0 else 0.0

# 4. Construct Structured Data Rows
data_rows = []

# Add Overall Summary Row
data_rows.append({
    "class_id": "all",
    "class_name": "overall",
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1_score, 4),
    "mAP50": round(map50, 4),
    "mAP50-95": round(map50_95, 4),
    "fitness": round(fitness, 4),
    "preprocess_ms": round(prep_ms, 2),
    "inference_ms": round(inf_ms, 2),
    "postprocess_ms": round(post_ms, 2),
    "total_latency_ms": round(total_ms, 2),
    "fps": round(fps, 1)
})

# Add Per-Class Detailed Rows Safely
class_indices = metrics.box.ap_class_index
for i, cls_idx in enumerate(class_indices):
    cls_name = metrics.names[cls_idx]
    p_cls = float(metrics.box.p[i])
    r_cls = float(metrics.box.r[i])
    f1_cls = (2 * p_cls * r_cls) / (p_cls + r_cls + 1e-16)
    map50_cls = float(metrics.box.ap50[i])
    map50_95_cls = float(metrics.box.ap[i])
    
    data_rows.append({
        "class_id": int(cls_idx),
        "class_name": cls_name,
        "precision": round(p_cls, 4),
        "recall": round(r_cls, 4),
        "f1_score": round(f1_cls, 4),
        "mAP50": round(map50_cls, 4),
        "mAP50-95": round(map50_95_cls, 4),
        "fitness": None,
        "preprocess_ms": None,
        "inference_ms": None,
        "postprocess_ms": None,
        "total_latency_ms": None,
        "fps": None
    })

# 5. Export DataFrame to CSV
df = pd.DataFrame(data_rows)
output_path = "sim2real.csv"
df.to_csv(output_path, index=False)

print(f"\nSuccess! Metrics exported without issues to '{output_path}'.")
print(df)

