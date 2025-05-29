# import json
# import os
from unsloth import FastVisionModel # FastLanguageModel for LLMs
# import torch
# from dataloader.intent_dataloader_HF    import get_train_val_datasets
# from unsloth import is_bf16_supported
# from unsloth.trainer import UnslothVisionDataCollator
# from trl import SFTTrainer, SFTConfig
# from torch import nn
# import re

# import torch

# def locate_assistant_token( array, target=77091):
#     positions = torch.nonzero(array == target, as_tuple=False)
#     return positions

# def compute_loss_func(outputs, labels, num_items_in_batch=None):
#     logits = outputs.get("logits")
#     assistant_positions = locate_assistant_token(labels)
#     min_position = torch.min(assistant_positions[:,1])-5
    
#     # batch_size = labels.size(0)
#     shift_logits = logits[..., :-1, :].contiguous()
#     shift_labels = labels[..., 1:].contiguous()
#     shift_logits = shift_logits[:, min_position:, :].contiguous()
#     shift_labels = shift_labels[:, min_position:].contiguous()
#     # Flatten the tokens
#     loss_fct = nn.CrossEntropyLoss()
#     loss = loss_fct(
#         shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
#     )
#     return loss

# def prepare_prompt(prompt, gt, image):
#         return [
#             {
#                 "role": "system",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": """You are an advanced vision-language model specialized in annotating images.
#                         You are a vision-language model. Analyze the provided image and respond **only in JSON** format. 
#                         Do not include any explanation, description, or text outside of the JSON
#                         Output Format:
#                         Return the crops as a JSON array, where each object contains:
#                             "y1": top-left y-coordinate
#                             "x1": top-left x-coordinate
#                             "y2": bottom-right y-coordinate
#                             "x2": bottom-right x-coordinate"""
#                     }
#                 ]
#             },
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "image",
#                         "image" : image
#                     },
#                     {
#                         "type": "text",
#                         "text": prompt
#                     }
#                 ]
#             },
#             {
#                 "role": "assistant",
#                 "content": [
#                     {
#                         "type": "text", 
#                         "text": gt
#                     }
#                 ],
#             },
#         ]



# model, tokenizer = FastVisionModel.from_pretrained(
#      "/root/ImageCropExtractor/outputs_dataset5_on_comp/checkpoint-660",
#     load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
#     use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
#     max_seq_length = 50000, # Set to a large number for long context
# )

# # model = FastVisionModel.get_peft_model(
# #     model,
# #     finetune_vision_layers     = False, # False if not finetuning vision layers
# #     finetune_language_layers   = True, # False if not finetuning language layers
# #     finetune_attention_modules = True, # False if not finetuning attention layers
# #     finetune_mlp_modules       = True, # False if not finetuning MLP layers

# #     r = 32,           # The larger, the higher the accuracy, but might overfit
# #     lora_alpha = 32,  # Recommended alpha == r at least
# #     lora_dropout = 0,
# #     bias = "none",
# #     random_state = 69,
# #     use_rslora = False,  # We support rank stabilized LoRA
# #     loftq_config = None, # And LoftQ
# #     # target_modules = "all-linear", # Optional now! Can specify a list if needed
# # )
# dataset_train, dataset_test = get_train_val_datasets()
# processed_dataset_train = [{ "messages" : prepare_prompt(prompt, answer, image_pil) }  for image_pil, prompt, answer in dataset_train]
# processed_dataset_test = [{ "messages" : prepare_prompt(prompt, answer, image_pil) }  for image_pil, prompt, answer in dataset_test]

# FastVisionModel.for_training(model) # Enable for training!

# trainer = SFTTrainer(
#     model = model,
#     tokenizer = tokenizer,
#     data_collator = UnslothVisionDataCollator(model, 
#                                               tokenizer,
#                                               resize="max",
#                                               train_on_responses_only=True,
#                                               instruction_part="<|im_start|>system",
#                                               response_part="<|im_start|>assistant"),
#     train_dataset = processed_dataset_train,
#     eval_dataset = processed_dataset_test,
#     args = SFTConfig(
#         per_device_train_batch_size = 4,
#         per_device_eval_batch_size = 4,  # Batch size for evaluation
#         do_eval=True,
#         do_train=True,
#         gradient_accumulation_steps = 1,
#         # warmup_steps = 60,
#         max_steps = 1,
#         eval_steps = 60,  # Steps interval for evaluation
#         eval_strategy = "steps",  # Strategy for evaluation
#         # num_train_epochs = 5, # Set this instead of max_steps for full training runs
#         learning_rate = 2e-4,
#         fp16 = not is_bf16_supported(),
#         bf16 = is_bf16_supported(),
#         logging_steps = 1,
#         optim = "adamw_torch_fused",
#         weight_decay = 0.01,
#         lr_scheduler_type = "linear",
#         seed = 69,
#         output_dir = "outputs_hf",
#         report_to = "tensorboard",     # For Weights and Biases

#         # You MUST put the below items for vision finetuning:
#         remove_unused_columns = False,
#         dataset_text_field = "",
#         dataset_kwargs = {"skip_prepare_dataset": True},
#         dataset_num_proc = 8,
#         max_seq_length = 10000,
#         save_strategy = "best",
#         metric_for_best_model = "eval_loss",
#         greater_is_better = False,
#         save_total_limit = 1
#     ),
# )

# trainer_stats = trainer.train()
# model.save_pretrained("outputs_hf/lora_adapters")
# tokenizer.save_pretrained("outputs_hf/lora_adapters")

# FastVisionModel.for_inference(model) # Enable for inference!

# model.save_pretrained_merged("Cropper",
#                          tokenizer,
#                          save_method = "merged_16bit")

# # model.push_to_hub_merged("Singh8898/Cropper",
# #                          tokenizer,
# #                          token = "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",
# #                          save_method = "merged_16bit")

from peft import PeftModel

# # 1. Load your base + adapter from the adapter folder you already saved
# base,tokenizer = FastVisionModel.from_pretrained(
#     "unsloth/qwen2.5-vl-7b-instruct",
#     load_in_4bit=True,
#     use_gradient_checkpointing="unsloth",
#     max_seq_length=50000,
# )
# peft_model = PeftModel.from_pretrained(base, "outputs_hf/lora_adapters")

# # 2. Merge adapters into the base weights and unload them
# merged = peft_model.merge_and_unload()

# # 3. Save the resulting fused model
# # merged.save_pretrained("Cropper_merged")
# # tokenizer.save_pretrained("Cropper_merged")

# peft_model.push_to_hub_merged("Singh8898/Cropper",
#                          tokenizer,
#                          token = "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",
#                          save_method = "merged_16bit")
from huggingface_hub import HfApi, upload_folder

api = HfApi()
api.create_repo(repo_id="Singh8898/Cropper", private=False, exist_ok=True)

upload_folder(
    repo_id="Singh8898/Cropper",
    folder_path="Cropper_merged",
    token="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",
    repo_type="model"
)