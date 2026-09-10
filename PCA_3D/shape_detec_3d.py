import pyrealsense2 as rs 
import numpy as np
import cv2
import open3d as o3d
from ultralytics import YOLO

# 1. Load your trained YOLO model
model_path = r"C:\Users\User\Documents\AI_Robotics Projects\AAU\Summer\Dataset_YOLO\runs\detect\train\weights\best.pt"
model = YOLO(model_path)

# 2. Initialize RealSense Pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

# Align depth stream to color stream
align = rs.align(rs.stream.color)

# Extract Intrinsic Parameters
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics = color_stream.get_intrinsics()

fx, fy = intrinsics.fx, intrinsics.fy
cx, cy = intrinsics.ppx, intrinsics.ppy

def analyze_object_geometry(pts_3d):
    """
    Strips the tabletop via RANSAC, computes PCA orientation, and extracts 
    true 3D physical dimensions (OBB) & primitive shape type.
    """
    if len(pts_3d) < 40:
        return None
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_3d)
    
    # 1. Statistical Outlier Removal (removes flying depth noise)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.2)
    pcd_clean = pcd.select_by_index(ind)
    
    if len(pcd_clean.points) < 30:
        return None

    # 2. RANSAC Plane Removal (Strips the table surface under/behind the object)
    try:
        plane_model, inliers = pcd_clean.segment_plane(
            distance_threshold=0.012, ransac_n=3, num_iterations=100
        )
        # Keep points that are NOT part of the flat table plane
        pcd_object = pcd_clean.select_by_index(inliers, invert=True)
        clean_pts = np.asarray(pcd_object.points)
    except Exception:
        clean_pts = np.asarray(pcd_clean.points)

    if len(clean_pts) < 20:
        clean_pts = np.asarray(pcd_clean.points) # Fallback if plane removal over-cleans

    # 3. Centroid & PCA Orientation
    centroid = np.mean(clean_pts, axis=0)
    centered = clean_pts - centroid
    cov = np.cov(centered, rowvar=False)
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    sort_idx = np.argsort(eigenvalues)[::-1]
    major_axis = eigenvectors[:, sort_idx][:, 0]
    wrist_yaw_deg = np.degrees(np.arctan2(major_axis[1], major_axis[0]))

    # 4. Minimal Oriented Bounding Box (OBB)
    pcd_final = o3d.geometry.PointCloud()
    pcd_final.points = o3d.utility.Vector3dVector(clean_pts)
    try:
        obb = pcd_final.get_minimal_oriented_bounding_box()
        extent = obb.extent
    except Exception:
        extent = np.ptp(clean_pts, axis=0)

    # Sort dimensions from largest to smallest (meters)
    dx, dy, dz = sorted(extent, reverse=True)
    
    # 5. Primitive Shape Classification via Aspect Ratios
    aspect_ratio_1 = dx / (dy + 1e-6)
    aspect_ratio_2 = dy / (dz + 1e-6)
    
    if aspect_ratio_1 < 1.35 and aspect_ratio_2 < 1.35:
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

try:
    print("\nStarting RealSense RGB-D Accurate Geometry & Pose Pipeline... Press 'q' to exit.")
    
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue
            
        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        
        # Run YOLO Object Detection
        results = model(color_img, verbose=False)[0]
        
        for box in results.boxes:
            conf = float(box.conf[0])
            
            # 1. Filter out low-confidence background detections
            if conf < 0.50:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            
            # --- Depth Gating Step ---
            center_u, center_v = (x1 + x2) // 2, (y1 + y2) // 2
            z_center = depth_frame.get_distance(center_u, center_v)
            
            if z_center == 0:
                roi_depths = depth_img[y1:y2, x1:x2]
                valid_depths = roi_depths[roi_depths > 0] * 0.001
                z_center = np.median(valid_depths) if len(valid_depths) > 0 else 0
                
            # 2. Ignore distant background objects beyond workspace (> 1.2m)
            if z_center <= 0.2 or z_center > 1.2:
                continue

            pts_3d = []
            z_min_gate = z_center - 0.12
            z_max_gate = z_center + 0.12

            for v in range(y1, y2, 2):
                for u in range(x1, x2, 2):
                    z_m = depth_frame.get_distance(u, v)
                    
                    if z_min_gate <= z_m <= z_max_gate:
                        x_m = (u - cx) * z_m / fx
                        y_m = (v - cy) * z_m / fy
                        pts_3d.append([x_m, y_m, z_m])
                            
            pts_3d = np.array(pts_3d)
            geo_res = analyze_object_geometry(pts_3d)
            
            if geo_res is not None:
                vol = geo_res["volume_cm3"]

                # 3. Filter out unrealistic volume spikes (> 15,000 cm³)
                if vol > 15000.0:
                    continue

                centroid = geo_res["centroid"]
                yaw = geo_res["wrist_yaw_deg"]
                dims = geo_res["dimensions_cm"]
                shape = geo_res["shape_primitive"]
                
                # Annotate Bounding Box & Shape Metadata on Frame
                cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                disp_label = f"{label} ({shape}) | Yaw:{yaw:.1f}deg"
                disp_dims = f"L:{dims[0]} W:{dims[1]} H:{dims[2]}cm"
                
                cv2.putText(color_img, disp_label, (x1, max(y1 - 25, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                cv2.putText(color_img, disp_dims, (x1, max(y1 - 8, 35)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                            
                # Output formatted real-world physical metrics
                print(f"[{label.upper()}] Shape: {shape} | Pos (X,Y,Z): ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})m | "
                      f"Dims (L,W,H): {dims} cm | Yaw: {yaw:.1f}° | Vol: {vol} cm³")

        cv2.imshow("RealSense D435 3D Object Geometry & Pose Pipeline", color_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()