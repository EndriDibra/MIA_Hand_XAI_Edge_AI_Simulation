import os
import glob
import cv2
import albumentations as A

# 1. Paths & Configuration
dataset_path = r"C:\Users\User\Documents\AI_Robotics Projects\AAU\Summer\Dataset_YOLO"
images_dir = os.path.join(dataset_path, "train", "images")
labels_dir = os.path.join(dataset_path, "train", "labels")

CLASS_NAMES = {0: "apple", 1: "bottle", 2: "cup"}

# Custom target image counts per class
TARGET_COUNTS = {
    0: 400,  # Apple
    1: 400,  # Bottle
    2: 650   # Cup (oversampled to aggressively boost recall)
}

# 2. STEP ONE: Purge previously generated augmented files to start clean
print("Cleaning old augmented files...")
deleted_count = 0
for f in glob.glob(os.path.join(images_dir, "*_aug_*")):
    os.remove(f)
    deleted_count += 1
for f in glob.glob(os.path.join(labels_dir, "*_aug_*")):
    os.remove(f)

print(f"Removed {deleted_count} old augmented image/label files.")

# 3. Helper to sanitize box coordinates
def sanitize_bbox(bbox):
    x, y, w, h = bbox
    x_min, x_max = max(0.0, x - w / 2), min(1.0, x + w / 2)
    y_min, y_max = max(0.0, y - h / 2), min(1.0, y + h / 2)
    new_w = x_max - x_min
    new_h = y_max - y_min
    new_x = x_min + new_w / 2
    new_y = y_min + new_h / 2
    return [
        min(max(new_x, 0.001), 0.999),
        min(max(new_y, 0.001), 0.999),
        min(max(new_w, 0.001), 0.999),
        min(max(new_h, 0.001), 0.999)
    ]

# 4. Count base original images per class
class_image_files = {cls_id: set() for cls_id in CLASS_NAMES}
label_files = glob.glob(os.path.join(labels_dir, "*.txt"))

for label_file in label_files:
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    base_name = os.path.splitext(os.path.basename(label_file))[0]
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        cls_id = int(parts[0])
        if cls_id in class_image_files:
            class_image_files[cls_id].add(base_name)

print("\n--- Base Training Set Image Counts ---")
for cls_id, files in class_image_files.items():
    print(f"Class {cls_id} ({CLASS_NAMES[cls_id]}): {len(files)} original images -> Target: {TARGET_COUNTS[cls_id]}")

# 5. Albumentations Pipeline
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), translate_percent=(-0.06, 0.06), p=0.5),
    A.HorizontalFlip(p=0.5)
], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'], min_visibility=0.3, clip=True))

# 6. Generate precise targeted augmentations
print("\n--- Generating Targeted Augmentations ---")
for cls_id, current_files in class_image_files.items():
    current_count = len(current_files)
    target_count = TARGET_COUNTS[cls_id]
    
    if current_count >= target_count or current_count == 0:
        continue
    
    needed = target_count - current_count
    multiplier = max(1, round(needed / current_count))
    
    print(f"Augmenting '{CLASS_NAMES[cls_id]}' ({current_count} -> ~{target_count})...")
    
    for base_name in current_files:
        label_file = os.path.join(labels_dir, base_name + ".txt")
        if not os.path.exists(label_file):
            continue

        with open(label_file, 'r') as f:
            lines = f.readlines()

        bboxes = []
        category_ids = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            bboxes.append(sanitize_bbox([float(x) for x in parts[1:5]]))
            category_ids.append(int(parts[0]))

        img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = os.path.join(images_dir, base_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            continue

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for i in range(multiplier):
            try:
                transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
                aug_img = transformed['image']
                aug_bboxes = transformed['bboxes']
                aug_cats = transformed['category_ids']

                if not aug_bboxes:
                    continue

                new_name = f"{base_name}_aug_{CLASS_NAMES[cls_id]}_{i}"
                cv2.imwrite(
                    os.path.join(images_dir, new_name + ".jpg"),
                    cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                )

                with open(os.path.join(labels_dir, new_name + ".txt"), 'w') as out_f:
                    for cat, box in zip(aug_cats, aug_bboxes):
                        out_f.write(f"{cat} {' '.join(f'{v:.6f}' for v in box)}\n")

            except Exception as e:
                pass

print("\nTargeted dataset creation complete!")