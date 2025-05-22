import zipfile
import os

def extract_zip(zip_path, extract_to=None):
    if not os.path.exists(zip_path):
        print(f"File not found: {zip_path}")
        return
    if extract_to is None:
        # Use the zip file name (without extension) as the extraction folder.
        extract_to = os.path.splitext(zip_path)[0]
    with zipfile.ZipFile(zip_path, 'r') as z:
        cwd = os.getcwd()
        dataset_path = os.path.join(os.getcwd(), "dataset")
        print(dataset_path)
        os.makedirs(dataset_path, exist_ok=True)
        z.extractall(dataset_path)

if __name__ == "__main__":

    zip1 = os.path.join(os.getcwd(), "photo_2-20250522T154451Z-1-001.zip")
    zip2 = os.path.join(os.getcwd(), "bounding_boxes_2-20250522T154449Z-1-001.zip")
    extract_zip(zip1)
    extract_zip(zip2)