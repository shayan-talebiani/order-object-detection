from PIL import Image
import os

IMAGE_PATH = "images/8"

def convert_png_to_jpg_with_white_bg(png_path, output_path=None):
    # Open the PNG image
    image = Image.open(png_path).convert("RGBA")

    # Create a white background image
    white_bg = Image.new("RGB", image.size, (255, 255, 255))

    # Paste the image on the white background using the alpha channel as mask
    white_bg.paste(image, mask=image.split()[3])  # Use alpha channel as mask

    # Determine output path
    if not output_path:
        base = os.path.splitext(png_path)[0]
        output_path = base + ".jpg"

    # Save as JPG
    white_bg.save(output_path, "JPEG")

    print(f"Saved JPG with white background at: {output_path}")

# Example usage
convert_png_to_jpg_with_white_bg(f"{IMAGE_PATH}/drawing.png")