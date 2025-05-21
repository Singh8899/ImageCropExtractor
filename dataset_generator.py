import os
import json
import argparse
from PIL import Image

def generate_dataset(images_dir, answers_dir, output_file):
    entries = []
    
    # Get sorted list of image files assuming they are .jpg files.
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    for image_file in image_files:
        base_name, _ = os.path.splitext(image_file)
        image_path = os.path.join(images_dir, image_file)
        with Image.open(image_path) as img:
            height = img.height
            width = img.width
        # Build the expected answer file path (assumes .json extension)
        answer_file_path = os.path.join(answers_dir, base_name + '.json')
        if not os.path.exists(answer_file_path):
            print(f"Warning: Answer file for image {image_file} not found. Skipping.")
            continue

        with open(answer_file_path, 'r', encoding='utf-8') as af:
            answer_data = json.load(af)
            # Convert the list of bounding boxes to JSON format.
            new_coords = []
            for coord in answer_data:
                new_coord = {}
                new_coord['x1'] = coord['x']
                new_coord['y1'] = coord['y']
                new_coord['x2'] = coord['x']+coord['width']
                new_coord['y2'] = coord['y']+coord['height']
                new_coords.append(new_coord)
            answer_text = json.dumps(new_coords, ensure_ascii=False)

        # Build the dataset entry
        entry = {
            "id": base_name.zfill(8),
            "image": image_file,
            "prompt": prompt(height, width),
            "answer": answer_text
        }
        entries.append(entry)

    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump(entries, out_f, ensure_ascii=False, indent=2)
    print(f"Dataset with {len(entries)} entries has been written to {output_file}")


def prompt(height, width):
    return f"""You are given an image of height {height} and width {width} containing one or more people. Your task is to extract 1, 2, or 3 rectangular crops based on the number and importance of the persons in the image. The goal is to focus on the most important person(s) in the image. Follow these precise rules:
    Cropping Rules:
        If extracting 1 crop:
            The crop must have a 1:1 aspect ratio (square).
            It should center on the most important person or group.
        If extracting 2 crops:
            Each crop must have a 2:1 aspect ratio vertically (portrait style).
            Each crop should center on a different important person.
        If extracting 3 crops:
            Two crops must be 1:1 aspect ratio.
            One crop must be 2:1 vertical aspect ratio.
            Each crop should focus on a different important person. Assign the vertical crop to the most prominent one if possible.
    Importance Criteria:
        Importance is based on a combination of centrality, face visibility, size in the image, eye contact, and pose.
        Do not include irrelevant background or non-human subjects.
        Avoid overlapping crops unless it's necessary to focus on grouped individuals.
    Output Format:
    Return the crops as a JSON array, where each object contains:
        "y1": top-left y-coordinate
        "x1": top-left x-coordinate
        "y2": bottom-right y-coordinate
        "x2": bottom-right x-coordinate
        
    Notes:
        Make sure the aspect ratios strictly match the required format for the number of crops selected.
        Make sure the that the coordinates are within the image dimensions.
        The choice of number of crops (1, 2, or 3) depends on how many distinct important persons are visually identifiable.
    """

if __name__ == '__main__':
    images_dir = os.path.join(os.getcwd(), "dataset/photo")
    answers_dir = os.path.join(os.getcwd(), "dataset/bounding_boxes")
    output_file = os.path.join(os.getcwd(), "dataset/dataset.json")
    generate_dataset(images_dir, answers_dir, output_file)
