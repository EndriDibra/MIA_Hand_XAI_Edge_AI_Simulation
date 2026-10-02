import os 
import glob
import matplotlib.pyplot as plt
from PIL import Image

folder_path = "Architectures"
architectures = ["simple_cnn", "complex_cnn", "deeper_cnn"]
variants = ["keras", "fp32", "fp16", "int8"]

for arch in architectures:
    fig, axes = plt.subplots(1, 4, figsize=(20, 10))
    fig.suptitle(f"Architecture Visualizations: {arch.replace('_', ' ').title()}", fontsize=18, fontweight='bold')
    
    for idx, var in enumerate(variants):
        ax = axes[idx]
        
        # Build search patterns to catch .png, .jpg, or .jpeg extensions
        if var == "keras":
            pattern = os.path.join(folder_path, f"{arch}.keras.*")
        else:
            pattern = os.path.join(folder_path, f"{arch}_{var}.tflite.*")
        
        matches = glob.glob(pattern)
        
        if matches:
            filepath = matches[0]  # Take the first matching image file
            img = Image.open(filepath)
            ax.imshow(img)
            ax.set_title(var.upper(), fontsize=14, fontweight='semibold')
        else:
            expected_name = f"{arch}.keras.png" if var == "keras" else f"{arch}_{var}.tflite.png"
            ax.text(0.5, 0.5, f"Missing:\n{expected_name}", ha='center', va='center', fontsize=12, color='red')
            ax.set_title(var.upper(), fontsize=14)
            
        ax.axis('off')  # Hide ticks and borders

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, f"{arch}_comparison_grid.png"), dpi=300, bbox_inches='tight')
    plt.show()