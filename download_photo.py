import firebase_admin
from firebase_admin import credentials, storage, firestore
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from PIL import Image
import random

cred = credentials.Certificate("secret.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'weldlabeling.firebasestorage.app'
})

bucket = storage.bucket()

folder = "dataset"
os.makedirs(train_folder, exist_ok=True)

blobs = list(bucket.list_blobs(prefix="yuri-endpoint/"))
random.shuffle(blobs)

db = firestore.client()

photos_ref = db.collection("yuri-endpoint")

photo_count = 0

for i, blob in enumerate(blobs):
    file_name = os.path.basename(blob.name)
    if not file_name:
        continue

    file_path = os.path.join(folder, file_name)

    photo_id, _ = os.path.splitext(file_name)
    photo_doc = photos_ref.document(photo_id).get()
    if not photo_doc.exists:
        continue

    photo_data = photo_doc.to_dict()
    if not photo_data.get("processed", False):
        continue

    pin_data = photo_data.get("annotations")
    if len(pin_data) == 0:
        continue

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = folder
    ET.SubElement(annotation, "filename").text = file_name
    ET.SubElement(annotation, "path").text = file_path
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "weldLabel"
    size = ET.SubElement(annotation, "size")
    blob.download_to_filename(file_path)
    with Image.open(file_path) as img:
        width, height = img.size
        depth = len(img.getbands())
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    for pin in pin_data:
        obj = ET.SubElement(annotation, "object")
        label = pin.get("label")
        if label in ["SI", "ACCETTABILE"]:
            label = "good_weld"
            yes_count += 1
        else:
            no_count += 1
            label = "bad_weld"

        ET.SubElement(obj, "name").text = label
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(pin.get("x_left"))
        ET.SubElement(bndbox, "xmax").text = str(pin.get("x_right"))
        ET.SubElement(bndbox, "ymin").text = str(pin.get("y_top"))
        ET.SubElement(bndbox, "ymax").text = str(pin.get("y_bottom"))

    tree = ET.ElementTree(annotation)
    xml_file = os.path.join(folder, f"{photo_id}.xml")
    tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    photo_count += 1

print("Download delle foto e creazione dei file XML completati! 🎉")
print(f"Yes count: {yes_count}")
print(f"No count: {no_count}")
print(f"Photos saved: {photo_count}")
