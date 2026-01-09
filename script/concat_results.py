import sys
import os
from PIL import Image, ImageDraw, ImageFont

def concatenate_images():
    images = [
        ("inference_both_cfm.png", "CFM (ODE)"),
        ("inference_both_mean_flow.png", "MeanFlow (1-step)"),
        ("inference_both_mean_flow_rectified.png", "Rectified MeanFlow (1-step)")
    ]
    
    loaded_imgs = []
    for path, label in images:
        if os.path.exists(path):
            loaded_imgs.append((Image.open(path), label))
        else:
            print(f"Warning: {path} not found.")
            
    if not loaded_imgs:
        print("No images found.")
        return

    # Dimensions
    widths, heights = zip(*(i.size for i, l in loaded_imgs))
    
    total_width = sum(widths)
    max_height = max(heights)
    
    # Add space for text at top
    text_height = 40
    new_im = Image.new('RGB', (total_width, max_height + text_height), (255, 255, 255))
    
    draw = ImageDraw.Draw(new_im)
    
    # Try to load a font, otherwise use default
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    x_offset = 0
    for img, label in loaded_imgs:
        # Draw label centered above image
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_x = x_offset + (img.width - text_w) // 2
        draw.text((text_x, 5), label, fill="black", font=font)
        
        # Paste image
        new_im.paste(img, (x_offset, text_height))
        x_offset += img.width
        
    output_path = "inference_comparison.png"
    new_im.save(output_path)
    print(f"Saved concatenated image to {output_path}")

if __name__ == "__main__":
    concatenate_images()
