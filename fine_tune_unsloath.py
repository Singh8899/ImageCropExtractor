from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from dataloader.intent_dataloader_HF    import get_train_val_datasets
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

def prepare_prompt(prompt, gt, image):
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": """You are an advanced vision-language model specialized in annotating images.
                        You are a vision-language model. Analyze the provided image and respond **only in JSON** format. 
                        Do not include any explanation, description, or text outside of the JSON"""
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


model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-7B-Instruct",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
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
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 1,  # Batch size for evaluation
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
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "tensorboard",     # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
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