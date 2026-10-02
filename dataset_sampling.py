import os
import shutil
from collections import defaultdict

# ------------------------------------------------------------------------------
# 1. Configuration & Paths
# ------------------------------------------------------------------------------
SOURCE_DATASET_DIR = r"C:\Users\User\Documents\AI_Robotics Projects\AAU\Summer\Real_Dataset_YOLO"
OUTPUT_DATASET_DIR = r"C:\Users\User\Documents\AI_Robotics Projects\AAU\Summer\Real_Dataset_YOLO_150"

TARGET_PER_CLASS = 150
CLASSES = {0: "apple", 1: "bottle", 2: "cup"}
SPLITS = ["train", "val"]

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

# ------------------------------------------------------------------------------
# 2. Sampling & Copying Logic
# ------------------------------------------------------------------------------
def get_image_classes(label_path):
    """Reads a YOLO label file and returns the set of class IDs inside."""
    class_ids = set()
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        class_ids.add(int(float(parts[0])))
                    except ValueError:
                        continue
    return class_ids

def sample_split(split_name):
    src_img_dir = os.path.join(SOURCE_DATASET_DIR, split_name, "images")
    src_lbl_dir = os.path.join(SOURCE_DATASET_DIR, split_name, "labels")
    
    out_img_dir = os.path.join(OUTPUT_DATASET_DIR, split_name, "images")
    out_lbl_dir = os.path.join(OUTPUT_DATASET_DIR, split_name, "labels")
    
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    
    class_counts = defaultdict(int)
    copied_files = set()
    
    all_images = [f for f in os.listdir(src_img_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]
    
    print(f"\n--- Processing '{split_name}' split (Target: {TARGET_PER_CLASS} per class) ---")
    
    for img_name in all_images:
        if all(class_counts[cls_id] >= TARGET_PER_CLASS for cls_id in CLASSES.keys()):
            break
            
        base_name = os.path.splitext(img_name)[0]
        lbl_name = f"{base_name}.txt"
        lbl_path = os.path.join(src_lbl_dir, lbl_name)
        
        img_classes = get_image_classes(lbl_path)
        
        needed = any(class_counts[cls_id] < TARGET_PER_CLASS for cls_id in img_classes if cls_id in CLASSES)
        
        if needed:
            src_img_path = os.path.join(src_img_dir, img_name)
            dst_img_path = os.path.join(out_img_dir, img_name)
            shutil.copy2(src_img_path, dst_img_path)
            
            if os.path.exists(lbl_path):
                dst_lbl_path = os.path.join(out_lbl_dir, lbl_name)
                shutil.copy2(lbl_path, dst_lbl_path)
            
            for cls_id in img_classes:
                if cls_id in CLASSES:
                    class_counts[cls_id] += 1
            
            copied_files.add(img_name)

    print(f" Total images saved to '{split_name}': {len(copied_files)}")
    for cls_id, cls_name in CLASSES.items():
        print(f"   - {cls_name} (ID {cls_id}): {class_counts[cls_id]} instances")

# ------------------------------------------------------------------------------
# 3. Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    for split in SPLITS:
        sample_split(split)
        
    print(f"\n Sub-dataset successfully saved to: {OUTPUT_DATASET_DIR}")