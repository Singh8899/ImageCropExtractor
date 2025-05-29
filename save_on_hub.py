import os
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "/root/ImageCropExtractor/outputs_dataset5_on comp/checkpoint-120", # YOUR MODEL YOU USED FOR TRAINING
    load_in_4bit = True, # Set to False for 16bit LoRA

)
FastVisionModel.for_inference(model)

# Merge LoRA weights into the base model before pushing to hub
model = model.merge_and_unload()

model.push_to_hub("Singh8898/Cropper",
                  token = "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")
tokenizer.push_to_hub("Singh8898/Cropper",
                      token = "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")

