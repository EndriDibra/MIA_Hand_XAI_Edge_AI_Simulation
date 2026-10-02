import cv2 
from ultralytics import YOLO

# 1. Load fine-tuned weights
weights_path = r"C:\Users\User\Documents\AI_Robotics Projects\AAU\Summer\Dataset_YOLO\runs\detect\train-2\weights\best.pt"
model = YOLO(weights_path)

# 2. Open default webcam (0 for built-in camera, 1 for external USB camera)
cap = cv2.VideoCapture(0)

# Set resolution (matches model training resolution)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab camera frame.")
        break

    # 3. Run inference on current frame
    results = model.predict(source=frame, conf=0.20, verbose=False)

    # 4. Draw bounding boxes on frame
    annotated_frame = results[0].plot()

    # 5. Display output window
    cv2.imshow("YOLOv8 Live Detection", annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()