import pyrealsense2 as rs 
import numpy as np
import cv2

# Initialize pipeline and config
pipeline = rs.pipeline()
config = rs.config()

# Enable Color and Depth Streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

try:
    profile = pipeline.start(config)
    print("✓ RealSense D435 Connected Successfully on Windows!")
    
    # Extract Intrinsics
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intrinsics = depth_stream.get_intrinsics()
    
    print("\n--- Camera Intrinsics ---")
    print(f"Focal Length (fx, fy):   {intrinsics.fx:.2f}, {intrinsics.fy:.2f}")
    print(f"Principal Point (cx, cy): {intrinsics.ppx:.2f}, {intrinsics.ppy:.2f}")
    
    # Stream preview loop
    print("\nStreaming RGB-D preview... Press 'q' to stop.")
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue
            
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Colorize depth image for visual sanity check
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        
        # Display side-by-side stream
        images = np.hstack((color_image, depth_colormap))
        cv2.imshow("RealSense D435 Windows Test (RGB | Depth)", images)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()