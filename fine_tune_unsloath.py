import argparse
import json
import os
import re

import torch
from torch import nn
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel  # FastLanguageModel for LLMs
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator

from dataloader.intent_dataloader_HF import get_train_val_datasets


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


def train(output_model_path, token):

    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
        load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing=False,  # True or "unsloth" for long context
        max_seq_length=50000,  # Set to a large number for long context
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # False if not finetuning vision layers
        finetune_language_layers=True,  # False if not finetuning language layers
        finetune_attention_modules=True,  # False if not finetuning attention layers
        finetune_mlp_modules=True,  # False if not finetuning MLP layers
        r=32,  # The larger, the higher the accuracy, but might overfit
        lora_alpha=32,  # Recommended alpha == r at least
        lora_dropout=0,
        bias="none",
        random_state=69,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    processed_dataset_train, processed_dataset_test = get_train_val_datasets()

    FastVisionModel.for_training(model)  # Enable for training!

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(
            model,
            tokenizer,
            resize="max",
            train_on_responses_only=True,
            instruction_part="<|im_start|>system",
            response_part="<|im_start|>assistant",
        ),
        train_dataset=processed_dataset_train,
        eval_dataset=processed_dataset_test,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        args=SFTConfig(
            per_device_train_batch_size=8,
            per_device_eval_batch_size=4,  # Batch size for evaluation
            do_eval=True,
            do_train=True,
            gradient_accumulation_steps=1,
            warmup_steps=50,
            max_steps=1200,
            eval_steps=50,  # Steps interval for evaluation
            eval_strategy="steps",  # Strategy for evaluation
            # num_train_epochs = 5, # Set this instead of max_steps for full training runs
            learning_rate=2e-4,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1,
            optim="adamw_torch_fused",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=69,
            output_dir="outputs_checkpoint",
            report_to="wandb",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=8,
            max_seq_length=10000,
            save_strategy="best",
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=5,
            load_best_model_at_end=True,
        ),
    )
    trainer.train()
    model.push_to_hub_merged(
        output_model_path,
        tokenizer,
        token=token,
        save_method="merged_16bit",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", type=str, required=True, help="Path to the output model")
    parser.add_argument("-t", type=str, required=True, help="Hugging Face token for model upload")
    args = parser.parse_args()
    train(args.o, args.t)

if __name__ == "__main__":
    main()