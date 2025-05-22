import json
import os
from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from dataloader.intent_dataloader_HF    import get_train_val_datasets
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from torch import nn
import re

import torch
# import torch.nn.functional as F
# from scipy.optimize import linear_sum_assignment
# from torchmetrics.functional import generalized_box_iou

# def box_cxcywh_to_xyxy(boxes):
#     """Convert [cx, cy, w, h] -> [x0, y0, x1, y1]"""
#     x_c, y_c, w, h = boxes.unbind(-1)
#     b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
#          (x_c + 0.5 * w), (y_c + 0.5 * h)]
#     return boxes

# def compute_cost_matrix(pred_boxes, tgt_boxes, alpha=1.0, beta=1.0):
#     """
#     pred_boxes: [num_queries, 4]
#     tgt_boxes: [num_targets, 4]
#     """
#     # L1 cost
#     cost_l1 = torch.cdist(pred_boxes, tgt_boxes, p=1)

#     # GIoU cost
#     pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
#     tgt_boxes_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
#     giou = generalized_box_iou(pred_boxes_xyxy, tgt_boxes_xyxy)
#     cost_giou = -giou  # Maximize GIoU → Minimize -GIoU

#     # Final cost matrix
#     return alpha * cost_l1 + beta * cost_giou

# def hungarian_match(pred_boxes, tgt_boxes):
#     """
#     pred_boxes: Tensor [num_queries, 4]
#     tgt_boxes: Tensor [num_targets, 4]
#     Returns list of (pred_idx, tgt_idx)
#     """
#     cost_matrix = compute_cost_matrix(pred_boxes, tgt_boxes)
#     pred_ind, tgt_ind = linear_sum_assignment(cost_matrix.cpu().detach())
#     return torch.as_tensor(pred_ind, dtype=torch.int64), torch.as_tensor(tgt_ind, dtype=torch.int64)

# def loss_set_prediction(pred_boxes, tgt_boxes, alpha=1.0, beta=1.0):
#     """
#     pred_boxes: [batch_size, num_queries, 4]
#     tgt_boxes: list of [num_targets_i, 4] for each batch
#     """
#     batch_size = pred_boxes.shape[0]
#     total_l1_loss = 0.0
#     total_giou_loss = 0.0
#     total_boxes = 0

#     for b in range(batch_size):
#         pred = pred_boxes[b]       # [num_queries, 4]
#         tgt = tgt_boxes[b]         # [num_targets_i, 4]

#         pred_idx, tgt_idx = hungarian_match(pred, tgt)

#         matched_pred = pred[pred_idx]
#         matched_tgt = tgt[tgt_idx]

#         # L1 Loss
#         l1 = F.l1_loss(matched_pred, matched_tgt, reduction='sum')

#         # GIoU Loss
#         giou = generalized_box_iou(
#             box_cxcywh_to_xyxy(matched_pred),
#             box_cxcywh_to_xyxy(matched_tgt)
#         )
#         giou_loss = (1 - giou).sum()

#         total_l1_loss += l1
#         total_giou_loss += giou_loss
#         total_boxes += matched_pred.shape[0]

#     total_l1_loss /= total_boxes
#     total_giou_loss /= total_boxes
#     return alpha * total_l1_loss + beta * total_giou_loss

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
                            "x2": bottom-right x-coordinate"""
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image" : image
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text", 
                        "text": gt
                    }
                ],
            },
        ]


# def box_iou(pred_boxes, target_boxes, eps: float = 1e-6):
#     """
#     pred_boxes, target_boxes: tensors of shape (B, 4) in [x1, y1, x2, y2] format
#     returns: Tensor of shape (B,) with the IoU for each box
#     """
#     # Intersection coords
#     x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
#     y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
#     x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
#     y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

#     # Clamp to zero (no negative areas)
#     inter_w = (x2 - x1).clamp(min=0)
#     inter_h = (y2 - y1).clamp(min=0)
#     inter_area = inter_w * inter_h

#     # Areas
#     area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) * \
#                 (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)
#     area_tgt  = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=0) * \
#                 (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=0)

#     # Union
#     union = area_pred + area_tgt - inter_area + eps

#     return inter_area / union
# def locate_assistant_token(array, target=77091):
#     positions = torch.nonzero(array == target, as_tuple=False)
#     return positions

# class CustomSFTTrainer(SFTTrainer):
#     def __init__(self, *args, **kwargs):
#         super(CustomSFTTrainer, self).__init__(*args, **kwargs)
#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    
#         # get label and prediction tokens
#         # labels = inputs.get("labels")
#         outputs = model(**inputs)
#         predictions = outputs.get("logits")
#         assis_pos = locate_assistant_token(inputs["input_ids"])
#         batch_size = len(assis_pos)
#         gt = []
#         dim = []
#         for assis in assis_pos:
#             tokens = inputs["input_ids"].tolist()[assis[0]]
#             gt_temp = self.processing_class.decode(tokens[assis[1]:])
#             txt = self.processing_class.decode(tokens)
#             match = re.search(r'dimensions (\d+)x(\d+)', txt)
#             if match:
#                 width, height = match.groups()
#                 dim_temp = [int(width), int(height)]
#                 dim.append(dim_temp)
#             else:
#                 dim_temp = None
#             extracted_values = re.findall(r'\[(.*?)\]', gt_temp)
#             gt.append(extracted_values)

#         # decode predictions and labels
#         predicted_token_ids = torch.argmax(predictions, dim=-1)
#         assis_pos = locate_assistant_token(predicted_token_ids)
#         extracted_values = []
#         for assis in assis_pos:
#             gt_temp = self.processing_class.decode(inputs["input_ids"].tolist()[assis[0]][assis[1]:])
#             extracted_value = re.findall(r'\[(.*?)\]', gt_temp)
#             extracted_values.append(extracted_value)
        
#         masks_gt = [torch.zeros(d) for d in dim]
#         masks_pred = [torch.zeros(d) for d in dim]

#         for i in range(len(batch_size)):

#             for j in range(len(gt[i])):
#                 gt_temp = gt[i][j]
#                 pred_temp = extracted_values[i][j]
#                 gt_temp = [int(x) for x in gt_temp.split(",")]
#                 pred_temp = [int(x) for x in pred_temp.split(",")]
#                 masks_gt[i][gt_temp[0]:gt_temp[2], gt_temp[1]:gt_temp[3]] = 1
#                 masks_pred[i][pred_temp[0]:pred_temp[2], pred_temp[1]:pred_temp[3]] = 1

#             for j in range(len(extracted_values[i])):
#                 pred_temp = extracted_values[i][j]
#                 pred_temp = [int(x) for x in pred_temp.split(",")]
#                 masks_pred[i][pred_temp[0]:pred_temp[2], pred_temp[1]:pred_temp[3]] = 1


        
#         decoded_predictions = [self.processing_class.decode(p.tolist()) for p in predicted_token_ids]

#         # labels[labels == -100] = self.processing_class.pad_token_id
#         # decoded_labels = [self.processing_class.decode(l.tolist()) for l in labels]
        
#         # Compute MSE loss
#         loss = outputs["loss"] + 1
        
#         return (loss, outputs) if return_outputs else loss


# class CustomLossWithIoU(nn.Module):
#     def __init__(self, tokenizer, iou_weight: float = 1.0):
#         """
#         iou_weight: how much to weight the IoU loss relative to token CE loss
#         """
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.iou_weight = iou_weight
#         self.ce_loss = nn.CrossEntropyLoss()

#     def __call__(self, outputs, labels, num_items_in_batch=None):
#         # 1) pull out everything you need
#         labels = outputs.pop("labels")
#         gt_boxes = outputs.pop("gt_boxes")      # your dataset must provide this
#         outputs = model(**inputs, labels=labels)
        
#         # 2) text CE loss
#         logits = outputs.logits
#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         ce = nn.CrossEntropyLoss()( 
#             shift_logits.view(-1, shift_logits.size(-1)),
#             shift_labels.view(-1)
#         )
        
#         # 3) box iou loss
#         pred_boxes = outputs.pred_boxes     # assumes your model returns this
#         ious = box_iou(pred_boxes, gt_boxes)
#         iou_loss = (1.0 - ious).mean()
        
#         loss = ce + self.args.iou_weight * iou_loss
        
#         return (loss, outputs) if return_outputs else loss



# class custom_loss_func():
#     def __init__(self, tokenizer):
#         super().__init__()
#         self.tokenizer = tokenizer
#     # outputs, labels, num_items_in_batch=num_items_in_batch
#     def __call__(self, outputs, labels, num_items_in_batch=None):
#         logits = outputs.get("logits")
#         assistant_positions = self.locate_assistant_token(labels)
#         weights = torch.ones(logits.size(-1)).to(logits.device)
#         weights[self.yes_token_id] = 0.8
#         min_position = torch.min(assistant_positions[:,1])-5
        
#         # batch_size = labels.size(0)
#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         shift_logits = shift_logits[:, min_position:, :].contiguous()
#         shift_labels = shift_labels[:, min_position:].contiguous()
#         # Flatten the tokens
#         loss_fct = nn.CrossEntropyLoss(weight=weights)
#         loss = loss_fct(
#             shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
#         )
#         return loss

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-7B-Instruct",
    load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

dataset_train, dataset_test = get_train_val_datasets()
processed_dataset_train = [{ "messages" : prepare_prompt(prompt, answer, image_pil) }  for image_pil, prompt, answer in dataset_train]
processed_dataset_test = [{ "messages" : prepare_prompt(prompt, answer, image_pil) }  for image_pil, prompt, answer in dataset_test]

FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = processed_dataset_train,
    eval_dataset = processed_dataset_test,
    args = SFTConfig(
        per_device_train_batch_size = 8,
        per_device_eval_batch_size = 8,  # Batch size for evaluation
        do_eval=True,
        do_train=True,
        gradient_accumulation_steps = 1,
        warmup_steps = 5,
        max_steps = 500,
        eval_steps = 10,  # Steps interval for evaluation
        eval_strategy = "steps",  # Strategy for evaluation
        # num_train_epochs = 5, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_torch_fused",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs_dataset_900/2",
        report_to = "tensorboard",     # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 8,
        max_seq_length = 5000,
        save_strategy = "best",
        metric_for_best_model = "eval_loss",
        greater_is_better = False,
        save_total_limit = 1
    ),
)
trainer_stats = trainer.train()

# FastVisionModel.for_inference(model) # Enable for inference!

# image = dataset_train[0][0]

# messages = [
#     processed_dataset_train[0]["messages"][:2]
# ]
# print(processed_dataset_train[0]["messages"][1:])
# input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
# inputs = tokenizer(
#     image,
#     input_text,
#     add_special_tokens = False,
#     return_tensors = "pt",
# ).to("cuda")

# from transformers import TextStreamer
# text_streamer = TextStreamer(tokenizer, skip_prompt = True)
# _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
#                    use_cache = True, temperature = 0.1, min_p = 0.1)



