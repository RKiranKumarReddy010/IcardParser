from PIL import Image, ImageDraw, ImageFont
import os

def create_sample_id_card():
    # Create a new image with blue background
    width = 1000
    height = 600
    image = Image.new('RGB', (width, height), '#0052CC')  # Blue background color
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 36)
        title_font = ImageFont.truetype("arial.ttf", 48)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # Add title
    draw.text((50, 30), "ID Card - stu_001", fill='white', font=title_font)
    
    # Add sample text with the exact data
    sample_data = [
        ("Name:", "Nathan Henry"),
        ("College:", "JNTU Kakinada"),
        ("Roll number:", "22JNT5377"),
        ("Branch:", "Computer Science"),
        ("Valid upto:", "2028")
    ]
    
    # Draw white background for text area
    text_bg = Image.new('RGB', (width - 300, height - 100), 'white')
    image.paste(text_bg, (280, 80))
    
    # Add text
    y_position = 100
    for label, value in sample_data:
        draw.text((300, y_position), f"{label} {value}", fill='black', font=font)
        y_position += 80
    
    # Save the image
    os.makedirs("tests/data", exist_ok=True)
    image_path = "tests/data/valid_id.png"
    image.save(image_path)
    print(f"Test image created at: {image_path}")
    return image_path

if __name__ == "__main__":
    create_sample_id_card() 