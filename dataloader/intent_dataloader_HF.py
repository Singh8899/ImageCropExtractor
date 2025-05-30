import os
import torch
import json
from torch.utils.data import Dataset
from PIL import Image


class DiegoDataset(Dataset):
    def __init__(self):
        # Assuming dataset/dataset.json is relative to the project root
        dataset_path = os.path.join(
            "/root/Workspace/ImageCropExtractor/dataset/dataset.json"
        )
        with open(dataset_path, "r") as json_file:
            self.dataset_list = json.load(json_file)

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        image_path = os.path.join(
            "/root/Workspace/ImageCropExtractor/dataset/photo",
            self.dataset_list[idx]["image"],
        )
        image_pil = Image.open(image_path)
        prompt = self.dataset_list[idx]["prompt"]
        answer = self.dataset_list[idx]["answer"]
        return {"messages": prepare_prompt(prompt, answer, image_pil)}


class TestDataset(Dataset):
    def __init__(self):
        # Assuming dataset/dataset.json is relative to the project root
        dataset_path = os.path.join(
            "/root/Workspace/ImageCropExtractor/dataset/dataset.json"
        )
        with open(dataset_path, "r") as json_file:
            self.dataset_list = json.load(json_file)

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        image_path = os.path.join(
            "/root/Workspace/ImageCropExtractor/dataset/photo",
            self.dataset_list[idx]["image"],
        )
        image_pil = Image.open(image_path)
        prompt = self.dataset_list[idx]["prompt"]
        answer = self.dataset_list[idx]["answer"]
        return image_pil, prompt, answer


def get_train_val_datasets(split_ratio=0.967):
    dataset = DiegoDataset()
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    torch.manual_seed(69)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    return train_dataset, val_dataset


def get_test_dataset(split_ratio=0.967):
    dataset = TestDataset()
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    torch.manual_seed(69)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    return train_dataset, val_dataset


def prepare_prompt(prompt, gt, image):
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
                            "x2": bottom-right x-coordinate""",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": gt}],
        },
    ]
