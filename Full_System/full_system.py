import csv 
import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# 1. INITIALIZATION & SETUP (Standard PyTorch/Ultralytics) 
# model_custom_path = r"C:\Users\User\Documents\AI_Robotics Projects\AAU\Summer\Dataset_YOLO\runs\detect\train\weights\best.pt"
model_custom_path = r"C:\Users\User\Documents\AI_Robotics Projects\AAU\Summer\Synthetic_Dataset_YOLO_30\runs\detect\train-2\weights\best.pt"
model_custom = YOLO(model_custom_path)
model_coco = YOLO("yolo11n.pt")

# --- MODEL WARM-UP) ---
dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
_ = model_custom(dummy_img, imgsz=320, verbose=False)
_ = model_coco(dummy_img, imgsz=320, verbose=False)

# Speed benchmarks logging setup
SPEED_CSV_FILENAME = "models_speed_results.csv"
SPEED_CSV_HEADERS = ["Model Path", "Inference Time (ms)", "Pipeline FPS"]

if not os.path.exists(SPEED_CSV_FILENAME) or os.path.getsize(SPEED_CSV_FILENAME) == 0:
    with open(SPEED_CSV_FILENAME, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SPEED_CSV_HEADERS)

# Temporal Filtering, Graceful Degradation & Frame Decimation Parameters
REQUIRED_CONSECUTIVE_FRAMES = 3
class_history = [] 
geometry_history = [] 

cached_geo_res = None
frame_count = 0

# Persistent tracking variables for frame decimation (ROI holding)
last_box = None
last_cls_id = None
last_label_name = None
last_fallback_flag = False

# COCO Classes relevant to prosthetic hand grasp strategy EXCLUDING (bottle:39, cup:41, apple:47)
ALLOWED_COCO_CLASSES = {
    40: "wine glass",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    54: "donut",
    64: "mouse",
    65: "remote",
    67: "cell phone",
    73: "book",
    75: "vase",
    76: "scissors",
    79: "toothbrush",
}

CSV_FILENAME = "shape_orientation2.csv"
CSV_HEADERS = [
    "Class ID",
    "Class Name",
    "Object Geometry",
    "Object Orientation (deg)",
    "Wrist Rotation",
    "Grasp Type",
]

file_exists = os.path.exists(CSV_FILENAME)
file_is_empty = not file_exists or os.path.getsize(CSV_FILENAME) == 0

if file_is_empty:
    with open(CSV_FILENAME, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)
else:
    with open(CSV_FILENAME, mode="r", newline="") as f:
        first_line = f.readline().strip()

    if "Class ID" not in first_line:
        with open(CSV_FILENAME, mode="r", newline="") as f:
            existing_data = f.read()
        with open(CSV_FILENAME, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
            f.write(existing_data)

print(f"\nCSV Target: {CSV_FILENAME}")
print(f"CSV Columns: {', '.join(CSV_HEADERS)}\n")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# --- REALSENSE WARM-UP (Discards initial frames during sensor auto-exposure) ---
for _ in range(10):
    pipeline.wait_for_frames()

color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics = color_stream.get_intrinsics()
fx, fy = intrinsics.fx, intrinsics.fy
cx, cy = intrinsics.ppx, intrinsics.ppy

CLASS_COLORS = [
    (0, 255, 0),    # Green -> Class ID 0
    (0, 165, 255),  # Orange -> Class ID 1
    (0, 255, 255),  # Yellow -> Class ID 2
]


def get_class_color(cls_id, is_fallback=False):
    if is_fallback:
        return (255, 191, 0)
    if 0 <= cls_id < len(CLASS_COLORS):
        return CLASS_COLORS[cls_id]
    return (255, 255, 255)


def evaluate_grasp_strategy(class_name, shape_primitive, yaw_deg, dimensions_cm):
    c_name = class_name.lower()
    dx, dy, dz = dimensions_cm

    if c_name in [
        "bottle",
        "cup",
        "banana",
        "apple",
        "orange",
        "vase",
        "bowl",
        "donut",
    ]:
        class_grasp = "Full Grasp"
    elif c_name in [
        "toothbrush",
        "fork",
        "spoon",
        "scissors",
        "knife",
        "mouse",
        "wine glass",
    ]:
        class_grasp = "Pinch Grasp"
    elif c_name in ["book", "cell phone", "remote", "sandwich"]:
        class_grasp = "Lateral Grasp"
    else:
        class_grasp = None

    if shape_primitive == "Flat" or min(dx, dy, dz) < 3.0:
        geom_grasp = "Pinch Grasp"
    elif shape_primitive in ["Cylindrical", "Spherical"]:
        geom_grasp = "Full Grasp"
    else:
        geom_grasp = "Lateral Grasp"

    if class_grasp is not None and class_grasp == geom_grasp:
        grasp_type = class_grasp
    else:
        grasp_type = geom_grasp

    if 90.0 <= yaw_deg < 170.0:
        wrist_rotation = "Left"
    elif (170.0 <= yaw_deg <= 180.0) or (-180.0 <= yaw_deg < -169.0):
        wrist_rotation = "Center"
    else:
        wrist_rotation = "Right"

    return grasp_type, wrist_rotation


def segment_full_object_fast(color_img, depth_meters, roi):
    x1, y1, x2, y2 = roi
    h_img, w_img = color_img.shape[:2]

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    roi_depths = depth_meters[y1:y2, x1:x2]
    valid_mask = roi_depths > 0

    if not np.any(valid_mask):
        return None, None

    z_min_surf = np.percentile(roi_depths[valid_mask], 15)
    z_min = max(0.15, z_min_surf - 0.04)
    z_max = z_min_surf + 0.16

    raw_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    depth_pass = (roi_depths >= z_min) & (roi_depths <= z_max)
    raw_mask[y1:y2, x1:x2][depth_pass] = 255

    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    full_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_large)

    roi_mask = full_mask[y1:y2, x1:x2]
    if np.count_nonzero(roi_mask) < 50:
        return None, None

    v_indices, u_indices = np.where(full_mask == 255)
    step = 3
    v_indices = v_indices[::step]
    u_indices = u_indices[::step]

    z_m = depth_meters[v_indices, u_indices]
    z_m[z_m == 0] = z_min_surf
    x_m = (u_indices - cx) * z_m / fx
    y_m = (v_indices - cy) * z_m / fy

    pts_3d = np.column_stack((x_m, y_m, z_m))
    return full_mask, pts_3d


def analyze_object_geometry_numpy(pts_3d):
    if len(pts_3d) < 20:
        return None

    mean_pts = np.mean(pts_3d, axis=0)
    std_pts = np.std(pts_3d, axis=0)
    inlier_mask = np.all(np.abs(pts_3d - mean_pts) < 2.0 * std_pts, axis=1)
    clean_pts = pts_3d[inlier_mask]

    if len(clean_pts) < 15:
        clean_pts = pts_3d

    centroid = np.mean(clean_pts, axis=0)
    centered = clean_pts - centroid
    cov = np.cov(centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    sort_idx = np.argsort(eigenvalues)[::-1]

    major_axis = eigenvectors[:, sort_idx][:, 0]
    wrist_yaw_deg = np.degrees(np.arctan2(major_axis[1], major_axis[0]))

    extents = np.ptp(clean_pts, axis=0)
    dx, dy, dz = sorted(extents, reverse=True)

    aspect_ratio_1 = dx / (dy + 1e-6)
    aspect_ratio_2 = dy / (dz + 1e-6)

    if dz < 0.02:
        shape_primitive = "Flat"
    elif aspect_ratio_1 < 1.35 and aspect_ratio_2 < 1.35:
        shape_primitive = "Spherical"
    elif aspect_ratio_1 > 1.4:
        shape_primitive = "Cylindrical"
    else:
        shape_primitive = "Box / Complex"

    return {
        "centroid": centroid,
        "wrist_yaw_deg": wrist_yaw_deg,
        "dimensions_cm": (
            round(dx * 100, 1),
            round(dy * 100, 1),
            round(dz * 100, 1),
        ),
        "shape_primitive": shape_primitive,
        "volume_cm3": round(dx * dy * dz * 1e6, 1),
    }


# 3. UNIFIED REAL-TIME PIPELINE WITH FRAME DECIMATION & INSTANT CLEARING
try:
    print("Starting Pipeline with Frame Decimation & Instant Clearing...")
    prev_time = time.time()
    current_fps = 0.0

    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        depth_meters = depth_img * 0.001
        vis_img = color_img.copy()

        frame_count += 1
        run_inference = (frame_count % 2 == 0)  # Run YOLO inference every 2nd frame

        CONF_CUSTOM = 0.30
        CONF_COCO = 0.30

        detected_boxes = []

        if run_inference:
            t_start = time.perf_counter()
            results_custom = model_custom(color_img, imgsz=320, verbose=False)[0]
            t_end = time.perf_counter()
            custom_inf_time_ms = round((t_end - t_start) * 1000, 2)

            # Log model path, inference execution time, and total pipeline FPS to CSV
            with open(SPEED_CSV_FILENAME, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([model_custom_path, custom_inf_time_ms, round(current_fps, 1)])

            for box in results_custom.boxes:
                if float(box.conf[0]) >= CONF_CUSTOM:
                    detected_boxes.append((box, False))

            is_fallback = False
            active_model = model_custom

            if len(detected_boxes) == 0:
                results_coco = model_coco(color_img, imgsz=320, verbose=False)[0]
                active_model = model_coco
                is_fallback = True

                for box in results_coco.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    if conf >= CONF_COCO and cls_id in ALLOWED_COCO_CLASSES:
                        detected_boxes.append((box, True))

            if len(detected_boxes) > 0:
                box, fallback_flag = detected_boxes[0]
                last_box = box
                last_cls_id = int(box.cls[0])
                last_label_name = active_model.names[last_cls_id]
                last_fallback_flag = fallback_flag
            else:
                # Instantly clear bounding box and cached state if nothing is detected on inference frames
                last_box = None
                cached_geo_res = None
                class_history.clear()
                geometry_history.clear()

        # Process and render using current/cached box
        if last_box is not None:
            x1, y1, x2, y2 = map(int, last_box.xyxy[0].tolist())
            cls_id = last_cls_id
            label_name = last_label_name
            fallback_flag = last_fallback_flag

            if run_inference:
                class_history.append(label_name)
                if len(class_history) > REQUIRED_CONSECUTIVE_FRAMES:
                    class_history.pop(0)

            center_u, center_v = (x1 + x2) // 2, (y1 + y2) // 2
            z_center = depth_frame.get_distance(center_u, center_v)

            if z_center == 0:
                roi_depths = depth_meters[y1:y2, x1:x2]
                valid_depths = roi_depths[roi_depths > 0]
                z_center = np.median(valid_depths) if len(valid_depths) > 0 else 0

            if 0.2 < z_center <= 1.2:
                if run_inference:
                    mask, pts_3d = segment_full_object_fast(color_img, depth_meters, [x1, y1, x2, y2])
                    if mask is not None and pts_3d is not None:
                        geo_res = analyze_object_geometry_numpy(pts_3d)
                        if geo_res is not None and geo_res["volume_cm3"] <= 15000.0:
                            cached_geo_res = geo_res
                            shape = cached_geo_res["shape_primitive"]
                            geometry_history.append(shape)
                            if len(geometry_history) > REQUIRED_CONSECUTIVE_FRAMES:
                                geometry_history.pop(0)

                if cached_geo_res is not None:
                    shape = cached_geo_res["shape_primitive"]
                    class_color = get_class_color(cls_id, is_fallback=fallback_flag)
                    yaw = cached_geo_res["wrist_yaw_deg"]
                    dims = cached_geo_res["dimensions_cm"]

                    grasp_type, wrist_rot = evaluate_grasp_strategy(label_name, shape, yaw, dims)

                    if run_inference:
                        with open(CSV_FILENAME, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                cls_id,
                                label_name,
                                shape,
                                round(yaw, 2),
                                wrist_rot,
                                grasp_type,
                            ])

                    mask, pts_3d = segment_full_object_fast(color_img, depth_meters, [x1, y1, x2, y2])
                    if mask is not None:
                        mask_idx = mask == 255
                        vis_img[mask_idx] = cv2.addWeighted(
                            color_img[mask_idx], 0.4,
                            np.full_like(color_img[mask_idx], class_color), 0.6, 0
                        )
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(vis_img, contours, -1, class_color, 2)

                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), class_color, 2)
                    
                    # Updated display label showing classification, shape, grasp type, wrist rotation, and yaw angle
                    disp_label = f"{label_name} | {shape} | {grasp_type} | Wrist: {wrist_rot} | Deg: {int(yaw)}"
                    cv2.putText(
                        vis_img, disp_label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, class_color, 2
                    )
            else:
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (200, 200, 200), 2)

        # FPS Calculation & Display
        curr_time = time.time()
        current_fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time

        cv2.putText(
            vis_img, f"FPS: {current_fps:.1f}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        cv2.imshow("Decimated Perception Pipeline", vis_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()