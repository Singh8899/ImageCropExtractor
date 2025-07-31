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
os.makedirs(folder, exist_ok=True)

blobs = list(bucket.list_blobs(prefix="yuri-endpoint/")) #dataset

random.shuffle(blobs)

db = firestore.client()

photos_ref = db.collection("yuri-endpoint") #collage

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


print("Download delle foto completato! 🎉")
print(f"Photos saved: {photo_count}")
