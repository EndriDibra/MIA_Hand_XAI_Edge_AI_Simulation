import os, glob

img_dir = r"C:\Users\User\Documents\AI_Robotics Projects\AAU\Summer\Dataset_YOLO\train\images"
lbl_dir = r"C:\Users\User\Documents\AI_Robotics Projects\AAU\Summer\Dataset_YOLO\train\labels"

images = set(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(img_dir, "*.*")))
labels = set(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(lbl_dir, "*.txt")))

print(f"Total Images: {len(images)} | Total Labels: {len(labels)}")
print(f"Orphaned Images (no label): {len(images - labels)}")
print(f"Orphaned Labels (no image): {len(labels - images)}")