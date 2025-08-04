import json
import argparse
import os
from PIL import Image
from PIL import ImageDraw

image_path = (
    "/workspace/Qwen2-VL-Finetune/photo-20250519T115650Z-1-001/photo/1.jpg"
)
crop = "/workspace/Qwen2-VL-Finetune/testjson.json"


def crop_image(image_path, boxes):
    img = Image.open(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    draw = ImageDraw.Draw(img)
    for i, box in enumerate(boxes):
        print
        # Assume each bounding box is [x, y, height, width]
        y = int(box["y"])
        x = int(box["x"])
        h = int(box["height"])
        w = int(box["width"])
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
    out_file = f"{base_name}_painted.png"
    img.save(out_file)
    print(f"Image with painted bboxes saved to {out_file}")


def main():
    with open(crop, "r") as f:
        boxes = json.load(f)

    crop_image(image_path, boxes)


if __name__ == "__main__":
    main()
