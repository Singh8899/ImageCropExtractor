"""
VL_ModelHandler is an abstract base class that provides a framework for handling vision-language models. 
It includes methods for model retrieval, image processing, saving predictions, and computing loss functions.
Methods:
    __init__():
        Initializes the VL_ModelHandler with a default device set to "cuda".
    get_model():
        Returns the loaded model.
    get_processor():
        Returns the processor associated with the model.
    get_collator():
        Returns the collator used for data processing.
    crop_resize(image):
        Crops and resizes the given image to a predefined size.
        Args:
            image (PIL.Image): The input image to be cropped and resized.
        Returns:
            PIL.Image: The cropped and resized image.
    save(saving_folder, batch, gts, preds, resize):
        Saves images with ground truth and prediction bounding boxes drawn on them.
        Args:
            saving_folder (str): The folder where the images will be saved.
            batch (list): A batch of input data containing images and metadata.
            gts (list): Ground truth labels for the bounding boxes.
            preds (list): Predicted labels for the bounding boxes.
            resize (bool): Whether to resize the images before saving.
    save_yolo(image, saving_folder, preds, bboxes, frame_ids, no_pedestrian=False):
        Saves YOLO-style images with prediction bounding boxes drawn on them.
        Args:
            image (PIL.Image): The input image.
            saving_folder (str): The folder where the images will be saved.
            preds (list): Predicted labels for the bounding boxes.
            bboxes (list): Bounding box coordinates.
            frame_ids (list): Frame identifiers for the images.
            no_pedestrian (bool): Whether there are no pedestrians in the image.
    save_pie_yolo(image, saving_folder, preds, bboxes, frame_ids, no_pedestrian=False):
        Saves YOLO-style images with prediction bounding boxes drawn on them, 
        specifically for the PIE dataset.
        Args:
            image (PIL.Image): The input image.
            saving_folder (str): The folder where the images will be saved.
            preds (list): Predicted labels for the bounding boxes.
            bboxes (list): Bounding box coordinates.
            frame_ids (list): Frame identifiers for the images.
            no_pedestrian (bool): Whether there are no pedestrians in the image.
    locate_assistant_token(array, target=77091):
        Locates the positions of a specific token in a tensor.
        Args:
            array (torch.Tensor): The input tensor.
            target (int): The token to locate. Default is 77091.
        Returns:
            torch.Tensor: The positions of the target token in the tensor.
    compute_loss_func(outputs, labels, num_items_in_batch=None):
        Computes the loss function for the model outputs and labels.
        Args:
            outputs (dict): The model outputs containing logits.
            labels (torch.Tensor): The ground truth labels.
            num_items_in_batch (int, optional): The number of items in the batch.
        Returns:
            torch.Tensor: The computed loss.
    load_vlm():
        Abstract method to load the vision-language model. Must be implemented by subclasses.
    load_finetuned_vlm():
        Abstract method to load a fine-tuned vision-language model. Must be implemented by subclasses.
    inference():
        Abstract method to perform inference using the model. Must be implemented by subclasses.
"""

from abc import ABC, abstractmethod
from PIL import Image, ImageDraw
import os
import torch
from torch import nn

class VL_ModelHandler(ABC):
    def __init__(self):
        self.device = "cuda"

    def get_model(self):
        return self.model

    def get_processor(self):
        return self.processor
    
    def get_collator(self):
        return self.collator
    
    def crop_resize(self, image):
        crop_box = (0, 60, 1920, 1020)  
        cropped_img = image.crop(crop_box)
        new_size = (960, 480)  
        resized_img = cropped_img.resize(new_size)
        return resized_img 

    def save(self, saving_folder, batch, gts, preds, resize):
        for i, el in enumerate(batch):
            # img = el[0][-1].copy().copy()
            img_data = el[0][-1].copy()
            bboxes = el[1][-1]
            for (bbox, gt, pred) in zip(bboxes, gts, preds) :
                draw = ImageDraw.Draw(img_data, "RGBA")
                colour_gt = "green" if gt.upper() == "NO" else "red"
                colour_pred = "green" if pred.upper() == "NO" else "red"
                fill_color = (255, 0, 0, 50) if colour_pred == "red" else (0, 255, 0, 50)
                draw.rectangle(bbox, outline=colour_gt, fill=fill_color, width=2)
            if resize:
                img_data = self.crop_resize(img_data.copy())

            pred_img_folder  = os.path.join(saving_folder, 'pred')
            os.makedirs(pred_img_folder + "/" +  el[4][-1][0] + "_"+ el[4][-1][1], exist_ok=True)
            img_data.save(os.path.join(pred_img_folder, el[4][-1][0] + "_"+ el[4][-1][1] , str(el[4][-1][2]) + ".png"))

    def save_yolo(self, image, saving_folder, preds, bboxes, frame_ids, no_pedestrian=False):
        img_data = image.copy()
        if not no_pedestrian:
            for (bbox, pred) in zip(bboxes, preds) :
                draw = ImageDraw.Draw(img_data, "RGBA")
                colour_pred = "green" if pred.upper() == "NO" else "red"
                fill_color = (255, 0, 0, 50) if colour_pred == "red" else (0, 255, 0, 50)
                draw.rectangle(bbox, fill=fill_color, width=2)

        pred_img_folder  = os.path.join(saving_folder, 'pred')
        os.makedirs(pred_img_folder, exist_ok=True)
        img_data.save(os.path.join(pred_img_folder, frame_ids[-1] ))

    def save_pie_yolo(self, image, saving_folder, preds, bboxes, frame_ids, no_pedestrian=False):
        img_data = image.copy()
        if not no_pedestrian:
            for (bbox, pred) in zip(bboxes, preds) :
                draw = ImageDraw.Draw(img_data, "RGBA")
                colour_pred = "green" if pred.upper() == "NO" else "red"
                fill_color = (255, 0, 0, 50) if colour_pred == "red" else (0, 255, 0, 50)
                draw.rectangle(bbox, fill=fill_color, width=2)

        pred_img_folder  = os.path.join(saving_folder, 'pred')
        os.makedirs(pred_img_folder + "/" +  frame_ids[-1][0] + "_"+ frame_ids[-1][1], exist_ok=True)
        img_data.save(os.path.join(pred_img_folder, frame_ids[-1][0] + "_"+ frame_ids[-1][1] , str(frame_ids[-1][2]) + ".png"))

    def locate_assistant_token(self, array, target=77091):
        positions = torch.nonzero(array == target, as_tuple=False)
        return positions        

    @abstractmethod
    def load_vlm(self):
        pass

    @abstractmethod
    def load_finetuned_vlm(self):
        pass

    @abstractmethod
    def inference(self):
        pass
    
        
