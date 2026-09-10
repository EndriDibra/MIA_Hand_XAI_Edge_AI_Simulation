import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# 1. Load YOLO model
model_path = r"C:\Users\User\Documents\AI_Robotics Projects\AAU\Summer\Dataset_YOLO\runs\detect\train\weights\best.pt"
model = YOLO(model_path)

# 2. RealSense Setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
align = rs.align(rs.stream.color)

color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics = color_stream.get_intrinsics()
fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy

def segment_full_object_robust(color_img, depth_frame, roi):
    """
    Combines adaptive depth gating with color-guided GrabCut 
    to capture 100% of the object, including caps and specular edges.
    """
    x1, y1, x2, y2 = roi
    h_img, w_img = color_img.shape[:2]

    # Ensure coordinates stay in-bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    depth_img = np.asanyarray(depth_frame.get_data())
    depth_meters = depth_img * 0.001

    roi_depths = depth_meters[y1:y2, x1:x2]
    valid_mask = roi_depths > 0

    if not np.any(valid_mask):
        return None, None

    # 1. Adaptive Depth Bounds (Front surface to back edge + generous +15cm depth room)
    z_min_surf = np.percentile(roi_depths[valid_mask], 15)
    z_min = max(0.15, z_min_surf - 0.04)
    z_max = z_min_surf + 0.16  # Expanded window to catch full object body

    # Initial Depth Mask inside ROI
    raw_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    roi_depth_crop = depth_meters[y1:y2, x1:x2]
    
    # Pass valid depth ranges
    depth_pass = (roi_depth_crop >= z_min) & (roi_depth_crop <= z_max)
    raw_mask[y1:y2, x1:x2][depth_pass] = 255

    # 2. Fill Depth Dropouts/Shadows with Aggressive Morphological Closing
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_large)

    # 3. GrabCut Refinement (Uses color cues to extend mask into missing depth zones like caps)
    roi_mask = closed_mask[y1:y2, x1:x2]
    if np.count_nonzero(roi_mask) < 50:
        return None, None

    # Initialize GrabCut mask states
    gc_mask = np.full((y2 - y1, x2 - x1), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[roi_mask == 255] = cv2.GC_PR_FGD  # Probably Foreground
    
    # Strong core seeding
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    core_fg = cv2.erode(roi_mask, kernel_small, iterations=2)
    gc_mask[core_fg == 255] = cv2.GC_FGD     # Definite Foreground

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    roi_color = color_img[y1:y2, x1:x2]

    try:
        cv2.grabCut(roi_color, gc_mask, None, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_MASK)
        final_roi_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        final_roi_mask = closed_mask[y1:y2, x1:x2]

    # Reconstruct full-frame mask
    full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = final_roi_mask

    # 4. Extract 3D Points for Segmented Region
    v_indices, u_indices = np.where(full_mask == 255)
    pts_3d = []

    for u, v in zip(u_indices, v_indices):
        z_m = depth_meters[v, u]
        if z_m == 0:  # Interpolate depth for filled missing pixels
            z_m = z_min_surf
        x_m = (u - cx) * z_m / fx
        y_m = (v - cy) * z_m / fy
        pts_3d.append([x_m, y_m, z_m])

    return full_mask, np.array(pts_3d)

try:
    print("\nRunning Robust Full-Object Segmentation... Press 'q' to exit.")
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame, depth_frame = aligned.get_color_frame(), aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_img = np.asanyarray(color_frame.get_data())
        vis_img = color_img.copy()

        results = model(color_img, verbose=False)[0]

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < 0.50:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = f"{model.names[int(box.cls[0])]} {conf:.2f}"

            # Robust Segmentation
            mask, target_pts = segment_full_object_robust(color_img, depth_frame, [x1, y1, x2, y2])

            if mask is not None:
                # Green translucent overlay
                mask_indices = mask == 255
                vis_img[mask_indices] = cv2.addWeighted(
                    color_img[mask_indices], 0.4, 
                    np.full_like(color_img[mask_indices], (0, 255, 0)), 0.6, 0
                )

                # Contour outlining full body
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)

            # Blue 2D YOLO Bounding Box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis_img, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Full Geometric + Color Refined Segmentation", vis_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()