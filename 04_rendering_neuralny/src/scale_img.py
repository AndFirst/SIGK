import os
from PIL import Image
import glob
from pathlib import Path

parent_path = Path(__file__).parent.parent
input_dir = parent_path / "output"
output_dir = parent_path / "output"

os.makedirs(output_dir, exist_ok=True)
image_files = glob.glob(os.path.join(input_dir, "image_*.png"))

for image_path in image_files:
    with Image.open(image_path) as img:
        img = img.convert("RGBA")
        
        black_bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img_no_alpha = Image.alpha_composite(black_bg, img)
        
        scaled_img = img_no_alpha.resize((128, 128), Image.Resampling.LANCZOS)
        
        scaled_img = scaled_img.convert("RGB")
        
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        scaled_img.save(output_path, "PNG", optimize=True)

print(f"Scaled {len(image_files)} images to 128x128 pixels")
