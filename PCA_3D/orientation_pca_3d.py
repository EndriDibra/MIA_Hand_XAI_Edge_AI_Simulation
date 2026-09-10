import pyrealsense2 as rs 
import numpy as np
import cv2
import open3d as o3d
from ultralytics import YOLO

# 1. Load your fine-tuned synthetic model weights
model_path = r"C:\Users\User\Documents\AI_Robotics Projects\AAU\Summer\Dataset_YOLO\runs\detect\train\weights\best.pt"
model = YOLO(model_path)

# 2. Setup RealSense Pipeline with Depth Alignment
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

# Align depth stream to color stream
align_to = rs.stream.color
align = rs.align(align_to)

# Extract Intrinsics
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics = color_stream.get_intrinsics()

fx, fy = intrinsics.fx, intrinsics.fy
cx, cy = intrinsics.ppx, intrinsics.ppy

def compute_pca_orientation(pts_3d):
    """Computes PCA on a set of 3D points to determine object centroid and orientation."""
    if len(pts_3d) < 50:
        return None  # Insufficient points for robust PCA
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_3d)
    
    # Statistical outlier removal to clean depth noise
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    clean_pts = np.asarray(pcd.select_by_index(ind).points)
    
    if len(clean_pts) < 30:
        return None
        
    centroid = np.mean(clean_pts, axis=0)
    centered = clean_pts - centroid
    cov = np.cov(centered, rowvar=False)
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort eigenvalues/eigenvectors in descending order
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]
    
    major_axis = eigenvectors[:, 0]  # Dominant direction vector
    wrist_yaw = np.arctan2(major_axis[1], major_axis[0])  # Yaw angle in camera frame
    
    return centroid, major_axis, np.degrees(wrist_yaw), eigenvalues

try:
    print("\nRunning RGB-D + YOLO + PCA Pipeline... Press 'q' to exit.")
    
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
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            
            # Extract 3D points inside the 2D bounding box ROI
            pts_3d = []
            depth_roi = depth_img[y1:y2, x1:x2]
            
            # Sampling grid inside bounding box
            for v in range(y1, y2, 2):
                for u in range(x1, x2, 2):
                    z_m = depth_frame.get_distance(u, v)
                    
                    # Ignore invalid zero depth and distant background (> 2.0m)
                    if 0.2 < z_m < 2.0:
                        x_m = (u - cx) * z_m / fx
                        y_m = (v - cy) * z_m / fy
                        pts_3d.append([x_m, y_m, z_m])
                        
            pts_3d = np.array(pts_3d)
            pca_res = compute_pca_orientation(pts_3d)
            
            if pca_res is not None:
                centroid, major_axis, wrist_yaw_deg, evals = pca_res
                
                # Annotate Bounding Box & PCA Angles on image
                cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                disp_text = f"{label} {conf:.2f} | Z:{centroid[2]:.2f}m | Yaw:{wrist_yaw_deg:.1f}deg"
                cv2.putText(color_img, disp_text, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                print(f"[{label.upper()}] Pos (X,Y,Z): ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})m | Wrist Yaw: {wrist_yaw_deg:.1f}°")

        cv2.imshow("RealSense D435 Real-Time PCA & Pose Estimation", color_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()