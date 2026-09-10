import os 
import shutil
import glob

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SOURCE_DIR = r"C:\Users\User\Documents\AI_Robotics Projects\AAU\Summer\Dataset_YOLO"
TARGET_DIR = r"C:\Users\User\Documents\AI_Robotics Projects\AAU\Summer\Dataset_YOLO_Formatted"

# Folder names in your source directory -> Target Class ID
CLASS_MAPPING = {
    "apple": 0,
    "bottle": 1,
    "cups": 2   # Matches your folder name 'cups'
}

SPLITS = ["train", "val"]
# ==============================================================================


def reindex_and_format_dataset():
    """Modifies class IDs in label files and structures dataset into standard YOLO format."""
    print("==================================================")
    print("      YOLO LABEL RE-INDEXER & DATASET FORMATTER   ")
    print("==================================================")

    if not os.path.exists(SOURCE_DIR):
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")

    total_images_copied = 0
    total_labels_updated = 0

    # Create root train/val subdirectories
    for split in SPLITS:
        os.makedirs(os.path.join(TARGET_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(TARGET_DIR, split, "labels"), exist_ok=True)

    for class_folder_name, target_class_id in CLASS_MAPPING.items():
        print(f"\n[INFO] Processing Class: '{class_folder_name.upper()}' -> Assigned Class ID: {target_class_id}")
        
        for split in SPLITS:
            src_img_dir = os.path.join(SOURCE_DIR, class_folder_name, split, "images")
            src_lbl_dir = os.path.join(SOURCE_DIR, class_folder_name, split, "labels")

            if not os.path.exists(src_img_dir):
                print(f"  [WARN] Image directory missing: {src_img_dir}")
                continue

            # Gather all images in the folder
            image_paths = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
                image_paths.extend(glob.glob(os.path.join(src_img_dir, ext)))

            split_img_count = 0
            split_lbl_count = 0

            for img_path in image_paths:
                base_name = os.path.basename(img_path)
                name_without_ext, ext = os.path.splitext(base_name)

                # Prefix class name to avoid identical name collisions across original datasets
                unique_name = f"{class_folder_name}_{name_without_ext}"

                # Copy image file to target structure
                dst_img_path = os.path.join(TARGET_DIR, split, "images", f"{unique_name}{ext}")
                shutil.copy2(img_path, dst_img_path)
                split_img_count += 1

                # Find corresponding text label file
                src_lbl_path = os.path.join(src_lbl_dir, f"{name_without_ext}.txt")
                dst_lbl_path = os.path.join(TARGET_DIR, split, "labels", f"{unique_name}.txt")

                if os.path.exists(src_lbl_path):
                    updated_lines = []
                    with open(src_lbl_path, "r") as f:
                        lines = f.readlines()

                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # Modify the 1st element (class ID) to match target_class_id
                            parts[0] = str(target_class_id)
                            updated_lines.append(" ".join(parts) + "\n")

                    # Save modified text file to target structure
                    with open(dst_lbl_path, "w") as f:
                        f.writelines(updated_lines)

                    split_lbl_count += 1
                else:
                    print(f"  [WARN] Missing label file for image: {base_name}")

            print(f"  -> [{split.upper()}] Copied {split_img_count} images, updated {split_lbl_count} labels.")
            total_images_copied += split_img_count
            total_labels_updated += split_lbl_count

    # Prepare formatted path string without backslashes inside the f-string block
    formatted_target_dir = TARGET_DIR.replace("\\", "/")

    # Create data.yaml automatically
    yaml_content = f"""path: {formatted_target_dir}
train: train/images
val: val/images

names:
  0: apple
  1: bottle
  2: cup
"""
    yaml_path = os.path.join(TARGET_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print("\n==================================================")
    print(f"[SUCCESS] Re-indexing and formatting complete!")
    print(f"[SUCCESS] Copied Images: {total_images_copied} | Updated Label Files: {total_labels_updated}")
    print(f"[SUCCESS] Output dataset path: '{TARGET_DIR}'")
    print(f"[SUCCESS] Generated config file: '{yaml_path}'")
    print("==================================================")


if __name__ == "__main__":
    reindex_and_format_dataset()