import json
import os
import xml.etree.ElementTree as ET

import firebase_admin
from firebase_admin import credentials, firestore, storage

cred = credentials.Certificate("secret.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'boomer-automation.firebasestorage.app'
})

bucket = storage.bucket()
db = firestore.client()

# Crea le cartelle per organizzare i file
photo_folder = "dataset/photo"
bb_folder = "dataset/bounding_boxes"
os.makedirs(photo_folder, exist_ok=True)
os.makedirs(bb_folder, exist_ok=True)

def download_from_collection(collection_name, bucket_prefix):
    """Scarica immagini da una collezione specifica"""
    print(f"\n🎯 Processando collezione: {collection_name} (bucket: {bucket_prefix})")
    
    photos_ref = db.collection(collection_name)
    docs = photos_ref.stream()
    photo_documents = list(docs)
    
    print(f"📚 Trovati {len(photo_documents)} documenti in {collection_name}")
    
    count = 0
    
    for doc in photo_documents:
        photo_data = doc.to_dict()
        
        # Controlla se il documento ha il campo image_name
        if 'image_name' not in photo_data:
            print(f"⚠️  Documento {doc.id} non ha il campo 'image_name', saltato")
            continue
        
        image_name = photo_data['image_name']
        
        # Controlla se l'immagine ha bounding boxes
        if 'bounding_boxes' not in photo_data or not photo_data['bounding_boxes']:
            print(f"⚠️  {image_name} non ha bounding boxes, saltato")
            continue
        
        blob_path = f"{bucket_prefix}/{image_name}"
        
        try:
            blob = bucket.blob(blob_path)
            
            if not blob.exists():
                print(f"❌ Immagine {image_name} non trovata nel bucket, saltata")
                continue
            
            # Path locali per salvare
            image_file_path = os.path.join(photo_folder, image_name)
            json_file_path = os.path.join(bb_folder, f"{os.path.splitext(image_name)[0]}.json")
            
            # Scarica l'immagine
            blob.download_to_filename(image_file_path)

            bounding_boxes = photo_data['bounding_boxes']
            
            # Salva i dati JSON (bounding boxes)
            with open(json_file_path, "w") as f:
                json.dump(bounding_boxes, f, indent=2)
            
            count += 1
            print(f"✅ Scaricato: {image_name} ({count}/{len(photo_documents)})")
            
        except Exception as e:
            print(f"❌ Errore scaricando {image_name}: {e}")
            continue
    
    return count

# Download da yuri-endpoint -> yuri-endpoint
total_count = 0
total_count += download_from_collection("yuri-endpoint", "yuri-endpoint")

# Download da collage -> dataset  
total_count += download_from_collection("collage", "dataset")


print("\n🎉 Download delle foto completato!")
print(f"📊 Totale foto scaricate: {total_count}")
print(f"📂 Immagini salvate in: {photo_folder}")
print(f"📋 Bounding boxes salvate in: {bb_folder}")
