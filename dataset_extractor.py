import os
import zipfile
import re
from pathlib import Path
import shutil

def get_next_index(folder_path):

    max_num = 0

    for item in os.listdir(folder_path / "photo"):
        # We found a number. Convert it to int and update max_num.
        file_name = Path(item).stem
        current_num = int(file_name)
        if current_num > max_num:
            max_num = current_num

    # If the folder is empty (max_num is 0), start at 1. Otherwise, start at max_num + 1.
    return max_num + 1

def extract_zip(zip_path, dataset_path):

    if not os.path.exists(zip_path):
        print(f"File not found: {zip_path}")
        return
    
    with zipfile.ZipFile(zip_path, "r") as z:
        name =  zip_path.stem
        if not os.path.exists(dataset_path):
            
            z.extractall()
            os.rename(name, "dataset")
        else:
            cwd = Path(os.getcwd())
            temp_dir = cwd / "Jaspi"
            z.extractall()
            index = get_next_index(dataset_path)

            for p_path in os.listdir(temp_dir / "photo"):
                photo_path = temp_dir / "photo" / p_path
                new_photo_path = dataset_path / "photo" / f"{index}.jpg"
                bb_path = temp_dir / "bounding_boxes" / Path(p_path).with_suffix(".json")
                new_bb_path = dataset_path / "bounding_boxes" / f"{index}.json"
                os.rename(photo_path, new_photo_path)
                os.rename(bb_path, new_bb_path)
                index += 1
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    cwd = Path(os.getcwd())
    target_folder = cwd / "dataset"
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    zip1 = cwd / "dataset_orig.zip"
    zip2 = cwd / "nuovo_dataset.zip"
    extract_zip(zip1, target_folder)
    extract_zip(zip2, target_folder)
