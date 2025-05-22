import os
import torch
import json

from torch.utils.data   import Dataset
from PIL                import Image



class DiegoDataset(Dataset):
  def __init__(self):
    # Assuming dataset/dataset.json is relative to the project root
    dataset_path = os.path.join("dataset", "dataset.json")
    with open(dataset_path, "r") as json_file:
      self.dataset_list = json.load(json_file)
    
  def __len__(self):
    return len(self.dataset_list)

  def __getitem__(self, idx):
    image_path = os.path.join("dataset", "photo", self.dataset_list[idx]['image'])
    image_pil = Image.open(image_path)
    prompt = self.dataset_list[idx]['prompt']
    answer = self.dataset_list[idx]['answer'] 
    return image_pil, prompt, answer

def get_train_val_datasets(split_ratio=0.8):
  dataset = DiegoDataset()
  train_size = int(len(dataset) * split_ratio)
  val_size = len(dataset) - train_size
  torch.manual_seed(42)
  train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
  return train_dataset, val_dataset



