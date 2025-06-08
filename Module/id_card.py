import os
from dotenv import load_dotenv
import json
import subprocess
import re
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

FONT_PATH = os.getenv("FONT_PATH")
FONT_SIZE = int(os.getenv("FONT_SIZE", 24))  
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output_images")
INPUT_DIR = os.getenv("INPUT_DIR", "json_data")
box_dir = os.getenv("box_dir")
gt_dir = os.getenv("gt_dir")
output_lstmf_dir = os.getenv("output_lstmf_dir")

class IdCard:
    @staticmethod
    def create_id_card(json_file_path):
        with open(json_file_path, "r") as f:
            data = json.load(f)

        user_id = data.get("user_id", "unknown_id")
        fields = data.get("extracted_fields", {})

        patterns = {
            "name": r"^\s*[A-Z][a-z]+(?: [A-Z][a-z]+)*\s*$", 
            "college": r"^[A-Za-z0-9\s.,&\-]+$",        
            "roll_number": r"^[A-Z0-9]{6,15}$",          
            "branch": r"^[A-Za-z\s]+$"                  
        }

        validated_fields = {}
        for key, value in fields.items():
            pattern = patterns.get(key)
            if pattern and not re.fullmatch(pattern, value):
                print(f"Warning: Field '{key}' with value '{value}' failed validation.")
                continue
            validated_fields[key] = value

        # Create image
        img = Image.new("RGB", (600, 300), color="white")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

        y = 20
        draw.text((20, y), f"ID Card - {user_id}", font=font, fill="black")
        y += 40
        for key, value in validated_fields.items():
            draw.text((20, y), f"{key.capitalize().replace('_', ' ')}: {value}", font=font, fill="black")
            y += 35

        output_path = os.path.join(OUTPUT_DIR, f"{user_id}.png")
        img.save(output_path)
        print(f"Saved: {output_path}")

    def box_convert():
        os.makedirs(box_dir, exist_ok=True)

        for filename in os.listdir(OUTPUT_DIR):
            if filename.endswith(".png"):
                image_path = os.path.join(OUTPUT_DIR, filename)
                base_name = os.path.splitext(filename)[0] 
                box_output_path = os.path.join(box_dir, f"{base_name}.box")
                
                print(f"Generating .box for: {filename}")

                subprocess.run([
                    "tesseract",
                    image_path,
                    os.path.join(box_dir, base_name),
                    "batch.nochop",
                    "makebox"
                ])

                
    def my_train_lstmf():
        for filename in os.listdir(OUTPUT_DIR):
            if filename.endswith(".png"):
                base = os.path.splitext(filename)[0]
                image_path = os.path.join(OUTPUT_DIR, filename)
                box_path = os.path.join(box_dir, f"{base}.box")
                gt_path = os.path.join(gt_dir, f"{base}.gt.txt")
                if not (os.path.exists(box_path) and os.path.exists(gt_path)):
                    print(f"Skipping {base}: missing .box or .gt.txt")
                    continue

                # Run lstm.train
                subprocess.run([
                    "tesseract",
                    image_path,
                    os.path.join(output_lstmf_dir, base),
                    "--psm", "7",
                    "lstm.train"
                ])

    def convert_to_gt():
        os.makedirs(gt_dir, exist_ok=True)
        for filename in os.listdir(INPUT_DIR):
            if filename.endswith(".json"):
                json_path = os.path.join(INPUT_DIR, filename)
                with open(json_path, "r") as f:
                    data = json.load(f)

                user_id = data.get("user_id", "unknown_id")
                fields = data.get("extracted_fields", {})

                lines = [f"ID Card - {user_id}"]
                for key, value in fields.items():
                    line = f"{key.capitalize().replace('_', ' ')}: {value}"
                    lines.append(line)

                base = os.path.splitext(filename)[0]
                gt_path = os.path.join(gt_dir, f"{base}.gt.txt")
                with open(gt_path, "w") as f:
                    f.write("\n".join(lines))

                print(f"Generated: {gt_path}")