from PIL import Image
import os

# Paths
input_dir = r"C:\Users\me\Desktop\isotope-case\renders"
output_dir = os.path.join(input_dir, "jpg_folder")
os.makedirs(output_dir, exist_ok=True)

# Optional resize
resize_to = (224, 224)  # set to None if you don't want resizing

# Process all PNGs
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".png"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".jpg")
        
        # Open image
        img = Image.open(input_path).convert("RGBA")
        
        # Create white background
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # paste using alpha channel as mask
        
        # Resize if needed
        if resize_to:
            background = background.resize(resize_to, Image.LANCZOS)
        
        # Save as JPG
        background.save(output_path, "JPEG", quality=85)

print("Conversion complete!")
