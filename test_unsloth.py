import os
from unsloth import FastVisionModel
from dataloader.intent_dataloader_HF    import get_train_val_datasets
import json
from PIL import ImageDraw

def prepare_prompt(prompt, gt):
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
                        "type": "image"
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



_, dataset_test = get_train_val_datasets()
processed_dataset_test = [(prepare_prompt(prompt, answer), image_pil) for image_pil, prompt, answer in dataset_test]



model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "/root/Workspace/ImageCropExtractor/outputs_dataset_900/2/checkpoint-360", # YOUR MODEL YOU USED FOR TRAINING
    load_in_4bit = False, # Set to False for 16bit LoRA
)

FastVisionModel.for_inference(model) # Enable for inference!
for i, pack in enumerate(processed_dataset_test[28:]):
    prompt, image_pil = pack
    messages = [
        prompt[:2]
    ]

    gt = prompt[2]["content"][0]["text"]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        image_pil,
        input_text,
        max_length=128000,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")


    # text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    pred_coord = model.generate(**inputs, max_new_tokens = 300,
                    use_cache = True, do_sample = False)

    text = tokenizer.batch_decode(pred_coord[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
    print(text)
    result_dict = json.loads(text)

    # Assuming result_dict contains bounding boxes in the format:
    # {"boxes": [{"x1": 10, "y1": 20, "x2": 110, "y2": 120}, ...]}
    draw = ImageDraw.Draw(image_pil)
    print(result_dict)
    for box in result_dict:
        x1 = box["x1"]
        y1 = box["y1"]
        x2 = box["x2"]
        y2 = box["y2"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    for gt_bbox in json.loads(gt):
        x1 = gt_bbox["x1"]
        y1 = gt_bbox["y1"]
        x2 = gt_bbox["x2"]
        y2 = gt_bbox["y2"]
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
    os.makedirs("test_output1", exist_ok=True)
    image_pil.save(f"test_output1/{i}.png")
