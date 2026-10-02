import pyrealsense2 as rs 
import numpy as np
import cv2
import open3d as o3d
import csv
import os
import time
from ultralytics import YOLO

# ==========================================
# 1. INITIALIZATION & SETUP
# ==========================================
model_path = r"C:\Users\User\Documents\AI_Robotics Projects\AAU\Summer\Dataset_YOLO\runs\detect\train\weights\best.pt"
model = YOLO(model_path)

CSV_FILENAME = "shape_orientation_2.csv"

# Initialize CSV file with updated headers
if not os.path.exists(CSV_FILENAME):
    with open(CSV_FILENAME, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Class ID", "Class Name", "Object Geometry", 
            "Object Orientation (deg)", "Wrist Rotation", "Grasp Type",
            "Latency (ms)", "FPS"
        ])

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
align = rs.align(rs.stream.color)

color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics = color_stream.get_intrinsics()
fx, fy = intrinsics.fx, intrinsics.fy
cx, cy = intrinsics.ppx, intrinsics.ppy

CLASS_COLORS = [
    (0, 255, 0),    # Green -> Class ID 0
    (0, 165, 255),  # Orange -> Class ID 1
    (0, 255, 255)   # Yellow -> Class ID 2
]

def get_class_color(cls_id):
    if cls_id < len(CLASS_COLORS):
        return CLASS_COLORS[cls_id]
    return (255, 255, 255)

def evaluate_grasp_strategy(class_name, shape_primitive, yaw_deg):
    """Determines grasp type and wrist orientation based on geometric agreement."""
    c_name = class_name.lower()
    
    if c_name in ["bottle", "cup", "apple"] and shape_primitive in ["Cylindrical", "Spherical"]:
        grasp_type = "Full Grasp"
    elif shape_primitive == "Flat":
        grasp_type = "Pinch Grasp"
    else:
        grasp_type = "Lateral Grasp"

    if yaw_deg < -10.0:
        wrist_rotation = "Left"
    elif yaw_deg > 10.0:
        wrist_rotation = "Right"
    else:
        wrist_rotation = "Center"

    return grasp_type, wrist_rotation

# ==========================================
# 2. HYBRID SEGMENTATION (DEPTH + GRABCUT)
# ==========================================
def segment_full_object_robust(color_img, depth_meters, roi):
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
    closed_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_large)

    roi_mask = closed_mask[y1:y2, x1:x2]
    if np.count_nonzero(roi_mask) < 50:
        return None, None

    gc_mask = np.full((y2 - y1, x2 - x1), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[roi_mask == 255] = cv2.GC_PR_FGD
    
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    core_fg = cv2.erode(roi_mask, kernel_small, iterations=2)
    gc_mask[core_fg == 255] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    roi_color = color_img[y1:y2, x1:x2]

    try:
        cv2.grabCut(roi_color, gc_mask, None, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_MASK)
        final_roi_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        final_roi_mask = closed_mask[y1:y2, x1:x2]

    full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = final_roi_mask

    v_indices, u_indices = np.where(full_mask == 255)
    pts_3d = []

    for u, v in zip(u_indices, v_indices):
        z_m = depth_meters[v, u]
        if z_m == 0:
            z_m = z_min_surf
        x_m = (u - cx) * z_m / fx
        y_m = (v - cy) * z_m / fy
        pts_3d.append([x_m, y_m, z_m])

    return full_mask, np.array(pts_3d)

# ==========================================
# 3. 3D GEOMETRY & POSE ANALYSIS
# ==========================================
def analyze_object_geometry(pts_3d):
    if len(pts_3d) < 40:
        return None
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_3d)
    
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.2)
    pcd_clean = pcd.select_by_index(ind)
    
    if len(pcd_clean.points) < 30:
        return None

    try:
        plane_model, inliers = pcd_clean.segment_plane(
            distance_threshold=0.012, ransac_n=3, num_iterations=100
        )
        pcd_object = pcd_clean.select_by_index(inliers, invert=True)
        clean_pts = np.asarray(pcd_object.points)
    except Exception:
        clean_pts = np.asarray(pcd_clean.points)

    if len(clean_pts) < 20:
        clean_pts = np.asarray(pcd_clean.points)

    centroid = np.mean(clean_pts, axis=0)
    centered = clean_pts - centroid
    cov = np.cov(centered, rowvar=False)
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    sort_idx = np.argsort(eigenvalues)[::-1]
    major_axis = eigenvectors[:, sort_idx][:, 0]
    wrist_yaw_deg = np.degrees(np.arctan2(major_axis[1], major_axis[0]))

    pcd_final = o3d.geometry.PointCloud()
    pcd_final.points = o3d.utility.Vector3dVector(clean_pts)
    try:
        obb = pcd_final.get_minimal_oriented_bounding_box()
        extent = obb.extent
    except Exception:
        extent = np.ptp(clean_pts, axis=0)

    dx, dy, dz = sorted(extent, reverse=True)
    
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
        "dimensions_cm": (round(dx * 100, 1), round(dy * 100, 1), round(dz * 100, 1)),
        "shape_primitive": shape_primitive,
        "volume_cm3": round(dx * dy * dz * 1e6, 1)
    }

# ==========================================
# 4. UNIFIED REAL-TIME PIPELINE & CSV LOGGING
# ==========================================
try:
    print(f"\nStarting Perception Pipeline with FPS Tracking... Logging to '{CSV_FILENAME}'")
    
    while True:
        # Start Latency Timer for the full perception cycle
        start_time = time.perf_counter()

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
        
        results = model(color_img, verbose=False)[0]
        
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < 0.50:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            label_name = model.names[cls_id]
            class_color = get_class_color(cls_id)

            center_u, center_v = (x1 + x2) // 2, (y1 + y2) // 2
            z_center = depth_frame.get_distance(center_u, center_v)
            
            if z_center == 0:
                roi_depths = depth_meters[y1:y2, x1:x2]
                valid_depths = roi_depths[roi_depths > 0]
                z_center = np.median(valid_depths) if len(valid_depths) > 0 else 0
                
            if z_center <= 0.2 or z_center > 1.2:
                continue

            mask, pts_3d = segment_full_object_robust(color_img, depth_meters, [x1, y1, x2, y2])
            
            if mask is None or pts_3d is None:
                continue

            geo_res = analyze_object_geometry(pts_3d)
            
            if geo_res is not None:
                vol = geo_res["volume_cm3"]
                if vol > 15000.0:
                    continue

                yaw = geo_res["wrist_yaw_deg"]
                shape = geo_res["shape_primitive"]
                
                # Evaluate Grasp Type & Wrist Rotation
                grasp_type, wrist_rot = evaluate_grasp_strategy(label_name, shape, yaw)

                # Calculate Latency & FPS for full pipeline execution
                execution_time_sec = time.perf_counter() - start_time
                latency_ms = round(execution_time_sec * 1000.0, 2)
                fps = round(1.0 / execution_time_sec, 1) if execution_time_sec > 0 else 0.0

                # Log data including latency and FPS
                with open(CSV_FILENAME, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        cls_id, label_name, shape, 
                        round(yaw, 2), wrist_rot, grasp_type,
                        latency_ms, fps
                    ])

                # Overlay Segmented Mask & Bounding Boxes
                mask_idx = mask == 255
                vis_img[mask_idx] = cv2.addWeighted(
                    color_img[mask_idx], 0.4, 
                    np.full_like(color_img[mask_idx], class_color), 0.6, 0
                )
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_img, contours, -1, class_color, 2)

                cv2.rectangle(vis_img, (x1, y1), (x2, y2), class_color, 2)
                disp_label = f"{label_name} ({shape}) | Grasp: {grasp_type}"
                disp_dims = f"Yaw:{yaw:.1f}deg ({wrist_rot}) | {latency_ms}ms ({fps} FPS)"
                
                cv2.putText(vis_img, disp_label, (x1, max(y1 - 25, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, class_color, 2)
                cv2.putText(vis_img, disp_dims, (x1, max(y1 - 8, 35)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imshow("Multi-Class Perception & CSV Logging Pipeline", vis_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()