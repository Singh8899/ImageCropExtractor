import requests
import time
from typing import List, Optional
import logging
from dataloader.intent_dataloader_HF    import get_train_val_datasets
import PIL 
from PIL import Image
import asyncio
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

def get_prompt(height, width):
    return f"""You are given an image of height {height} and width {width}. Your task is to extract 1, 2, or 3 rectangular/square crops based on the number and importance of the entities in the image. The goal is to focus on the most important aspects in the image. Follow these precise rules:
    Cropping Rules:
        If extracting 1 crop:
            The crop must have a 1:1 aspect ratio (square).
            It should centered on the most important person or group or entities.
        If extracting 2 crops:
            Each crop must have a 2:1 aspect ratio vertically (portrait style).
            Each crop should center on a different important entities.
        If extracting 3 crops:
            Two crops must be 1:1 aspect ratio.
            One crop must be 2:1 vertical aspect ratio.
            Each crop should focus on a different important entities. Assign the vertical crop to the most prominent one if possible.
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
        Remember Entities can be people, animals, or objects of interest.
        Make sure the aspect ratios strictly match the required format for the number of crops selected.
        Make sure the that the coordinates are within the image dimensions.
        The choice of number of crops (1, 2, or 3) depends on how many distinct important entities are visually identifiable.
    """

def prepare_prompt(image):
    # Convert image to base64 string
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """You are an advanced vision-language model specialized in annotating images.
                    You are a vision-language model. Analyze the provided image and respond **only in JSON** format. 
                    Do not include any explanation, description, or text outside of the JSON
                    Output Format:
                    Return the crops as a JSON array, where each object contains:
                        "y1": top-left y-coordinate
                        "x1": top-left x-coordinate
                        "y2": bottom-right y-coordinate
                        "x2": bottom-right x-coordinate"""
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_str  # base64 string
                },
                {
                    "type": "text",
                    "text": get_prompt(image.height, image.width)
                }
            ]
        }
    ]

class Cropper:
    def __init__(self):
        self.api_url = "https://api.runpod.ai/v2/0lvxt0phmbz41m/run"
        self.headers = {
            'Content-Type': "application/json",
            'Authorization': f"Bearer rpa_JWA1G2EMPVK7DM394YDNXRWS77M8YN75LNHY6I3L14my1q"
        }
        logger.info("Initialized Model")

    async def get_bb(self, image: Image) -> str:
        logger.info("Request next move")
        prompt = prepare_prompt(image)
        try:
            payload = {"input": {"prompt": prompt}}

            response = requests.post(
                self.api_url, json=payload, headers=self.headers)
            if response.status_code != 200:
                logger.error(
                    f"API request failed with status {response.status_code}")
                return []

            job_id = response.json()['id']

            while True:
                time.sleep(2)

                status_url = f"{self.api_url.rsplit('/', 1)[0]}/status/{job_id}"
                status_response = requests.get(
                    status_url, headers=self.headers)

                if status_response.status_code != 200:
                    logger.error("Error checking status")
                    break

                status_data = status_response.json()

                if status_data['status'] == 'COMPLETED':
                    bb_string = status_data['output'][0]['choices'][0]['tokens'][0].strip(
                    )
                    logger.info(f"Received move: {bb_string}")
                    return bb_string

                elif status_data['status'] == 'FAILED':
                    logger.error(f"Job failed: {status_data}")
                    break

        except Exception as e:
            logger.error(f"Error in attempt: {str(e)}")
        return []
    
cropper = Cropper()
path = "/root/ImageCropExtractor/dataset/photo/1.jpg"

async def main():
    with Image.open(path) as img:
        bb_string = await cropper.get_bb(img)
    print(bb_string)

asyncio.run(main())
