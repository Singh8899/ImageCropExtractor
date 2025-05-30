import base64
from io import BytesIO
import os
import json
from unsloth import FastVisionModel
from PIL import ImageDraw
from PIL import Image, ImageDraw

class CropInference:
    def __init__(self, model_path="/root/Workspace/Cropper/checkpoint-660", load_in_4bit=True):
        self.model, self.tokenizer =  FastVisionModel.from_pretrained(
            "/root/Workspace/ImageCropExtractor/outputs_checkpoint/checkpoint-300", # YOUR MODEL YOU USED FOR TRAINING
            load_in_4bit = True, # Set to False for 16bit LoRA
        )
        FastVisionModel.for_inference(self.model)

    def get_prompt(self, height, width):
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

    def prepare_prompt(self, image):
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
                    "content": [{"type": "image"}, {"type": "text", "text": self.get_prompt(image.height, image.width)}]
                }
            ]
    def infer(self, image: str):
        image_bytes = base64.b64decode(image)
        image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
        prompt = [self.prepare_prompt(image_pil)]
        
        input_text = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
        inputs = self.tokenizer(
            image_pil,
            input_text,
            max_length=128000,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        pred_coord = self.model.generate(
            **inputs, max_new_tokens=300, use_cache=True, do_sample=False
        )

        text = self.tokenizer.batch_decode(
            pred_coord[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )[0]
        print(f"Predicted Text: {text}")
        return json.loads(text)
