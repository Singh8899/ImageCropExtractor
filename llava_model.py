"""
This module contains classes and functions for training, testing, and inference of a vision-language model (LlavaOnevisionForConditionalGeneration) 
specialized in analyzing images and videos captured from a moving vehicle's camera. The model is designed to determine whether a pedestrian intends 
to cross the road based on their position, movement trend, and the car's speed.
Classes:
---------
- MyDataCollator:
    Abstract base class for data collators. It provides methods for preparing prompts and tokenizing inputs for training and testing.
- TrainDataCollator(MyDataCollator):
    A data collator for training. It processes batches of data, prepares prompts, and tokenizes inputs for training the model.
- TestDataCollator(MyDataCollator):
    A data collator for testing. It processes batches of data, prepares prompts, and tokenizes inputs for testing the model.
- TestDataCollatorPieYolo(MyDataCollator):
    A data collator for testing with PieYolo. It processes batches of data and returns video inputs and frame paths.
- YoloDataCollator(MyDataCollator):
    A data collator for testing with YOLO. It processes batches of data and returns video inputs and frame paths.
- Llava_model(VL_ModelHandler):
    A class for handling the LlavaOnevisionForConditionalGeneration model. It provides methods for loading, fine-tuning, inference, and training.
Functions:
----------
- locate_assistant_token(array, target=77091):
    Locates the positions of a specific token (default: assistant token) in a tensor.
- preprocess_logits_for_metrics(logits, labels_seq):
    Preprocesses logits and labels for computing evaluation metrics. Extracts predictions and labels for binary classification.
- compute_metrics(eval_pred):
    Computes evaluation metrics (accuracy, recall, precision, F1-score) for binary classification.
Usage:
------
1. Data Collators:
    - Use `TrainDataCollator` for preparing training data.
    - Use `TestDataCollator` for preparing test data.
    - Use `TestDataCollatorPieYolo` or `YoloDataCollator` for specific YOLO-based testing.
2. Llava_model:
    - Initialize the `Llava_model` class.
    - Use `load_vlm` to load the base model and processor for training.
    - Use `load_finetuned_vlm` to load a fine-tuned model for inference or testing.
    - Use `inference` to perform inference on input data.
    - Use `train` to fine-tune the model on a given dataset.
3. Metrics:
    - Use `preprocess_logits_for_metrics` to preprocess logits and labels for evaluation.
    - Use `compute_metrics` to compute evaluation metrics for binary classification.
Notes:
------
- The model uses PEFT (Parameter-Efficient Fine-Tuning) with LoRA (Low-Rank Adaptation) for fine-tuning.
- The `prepare_prompt` method in `MyDataCollator` and its subclasses generates prompts for the model based on bounding box coordinates, speed, and ground truth labels.
- The `inference` method in `Llava_model` supports dynamic caching for efficient generation.
Dependencies:
-------------
- torch
- transformers
- peft
- trl
- sklearn
- PIL
- abc
"""

import torch
from transformers               import BitsAndBytesConfig
from transformers               import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from models_handler             import VL_ModelHandler
from peft                       import get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel
from trl                        import SFTTrainer
from sklearn.metrics            import accuracy_score, recall_score, precision_score, f1_score
from abc                        import ABC, abstractmethod
from time                       import time   

class MyDataCollator(ABC):
    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id
        self.image_token_id = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)] #[151646]
        self.video_token_id = [processor.tokenizer.convert_tokens_to_ids(processor.video_token)] #[151647]

    @abstractmethod
    def __call__(self, batch):
        pass

    def prepare_prompt(self, prompt, gt):
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



class TrainDataCollator(MyDataCollator):
    def __init__(self, processor):
        super().__init__(processor)

    def __call__(self, batch):
        # batch comeposed by image_pil, prompt, answer
        texts = []
        image_inputs = []  # ex. torch batch x torch.Size([3, 1080, 1920])
        gts = []

        for el in batch:
            # bb_track = el[1]
            image_pil = el[0]
            prompt = el[1]
            answer = el[2]
            # gt = ["YES" if i == 1 else "NO" for i in el[2]]
            # gts.append(gt)
            # speed = el[3]
            # random_index = torch.randint(0, len(gt), (1,)).item()
            # bb_track[-1] = [bb_track[-1][random_index]]
            prepare_prompt = self.prepare_prompt(prompt, answer)
            texts.append(self.processor.apply_chat_template(prepare_prompt,
                                            tokenize=False,
                                            add_generation_prompt=False))
            # image_input = Image.open(el[0][-1])
            image_inputs.append(image_pil)
            
        model_inputs = self.processor(
            text=texts, # text dict keys ['input_ids', 'attention_mask', 'pixel_values', 'image_sizes']
            images=image_inputs,
            return_tensors="pt",
            padding=True
        )
        labels = model_inputs["input_ids"].clone()
        
        # mask padding tokens in labels
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # mask image token IDs in the labels
        for image_token_id in self.image_token_id:
            labels[labels == image_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs
        
class TestDataCollator(MyDataCollator):
    def __init__(self, processor):
        super().__init__(processor)

    def __call__(self, batch):
        # batch comeposed by image_tensor, bb_track, ped_int, obd_speed, heading
        # texts = []
        video_inputs = []  # torch batch x torch.Size([3, 1080, 1920])
        gts = []
        el = batch[0]
        bb_track = el[1]
        gt = ["YES" if i == 1 else "NO" for i in el[2]]
        gts.append(gt)
        speed = el[3]
        text_raw = self.prepare_prompt(bb_track, speed, gt)
        # text_i = [[text_raw[0],text_raw[1]]]+[[text_raw[i]] for i in range(3,len(bb_track[-1])*2+1,2)]
        text_i = [[text_raw[0],text_raw[1]]]+[[text_raw[i]] for i in range(2,len(bb_track[-1])*2+1,2)]
        video_inputs = el[0]
        return text_i, gts[0], video_inputs, batch
    
    
def locate_assistant_token(array, target=77091):
    positions = torch.nonzero(array == target, as_tuple=False)
    return positions

# Class version, in case something has to be passed to the metrics
# class Metrics_Preprocessor():
#     def __init__(self, processor):
#         self.processor = processor

#     def __call__(self, logits, labels_seq):
#         assistant_positions = locate_assistant_token(labels_seq)
#         pred_ids = torch.argmax(logits[0], dim=-1)

#         predictions_seq_trimmed = [ pred_ids[i][assistant_positions[i][1]+1:] 
#                                     for i in range(len(pred_ids))]
        
#         labels_seq_trimmed = [ labels_seq[i][assistant_positions[i][1]+2:] 
#                                     for i in range(len(pred_ids))]
        
#         for i in range(len(labels_seq_trimmed)):
#             labels_seq_trimmed[i][labels_seq_trimmed[i] == -100] = self.processor.tokenizer.pad_token_id

#         for i in range(len(predictions_seq_trimmed)):
#             predictions_seq_trimmed[i][predictions_seq_trimmed[i] == -100] = self.processor.tokenizer.pad_token_id

#         labels_decoded = self.processor.batch_decode(labels_seq_trimmed, 
#                                                 skip_special_tokens=True, 
#                                                 clean_up_tokenization_spaces=False)
        
#         predictions_decoded = self.processor.batch_decode(predictions_seq_trimmed, 
#                                                 skip_special_tokens=True, 
#                                                 clean_up_tokenization_spaces=False)

#         predictions = torch.tensor([1 if p == 14004 else 0 for p in predictions_seq_trimmed],dtype=torch.long, device=logits[0].device)
#         labels = torch.tensor([1 if l == 14004 else 0 for l in labels_seq_trimmed],dtype=torch.long, device=logits[0].device)
#         return predictions, labels
    
#     def prediction_parser(self, string):
#         raw = string.strip("'[] '").split(",")
#         return [i.strip("'[] '") for i in raw]
    
def preprocess_logits_for_metrics(logits, labels_seq):
    assistant_positions = locate_assistant_token(labels_seq)
    pred_ids = torch.argmax(logits[0], dim=-1)

    predictions_seq_trimmed = [ pred_ids[i][assistant_positions[i][1]+1] 
                                for i in range(len(pred_ids))]
    
    labels_seq_trimmed = [ labels_seq[i][assistant_positions[i][1]+2] 
                                for i in range(len(pred_ids))]
    predictions = torch.tensor([1 if p == 14004 else 0 for p in predictions_seq_trimmed],dtype=torch.long, device=logits[0].device)
    labels = torch.tensor([1 if l == 14004 else 0 for l in labels_seq_trimmed],dtype=torch.long, device=logits[0].device)
    return predictions, labels

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions[0]
    labels = eval_pred.predictions[1]

    accuracy = accuracy_score(predictions, labels)
    recall = recall_score(predictions, labels, average='binary', pos_label=0, zero_division=0)
    precision = precision_score(predictions, labels, average='binary',pos_label=0, zero_division=0)
    f1 = f1_score(predictions, labels, average='binary',pos_label=0, zero_division=0)
    return {
        "Accuracy": round(accuracy, 4),
        "Recall": round(recall, 4),
        "Precision": round(precision, 4),
        "F1-score": round(f1, 4),
    }

class Llava_model(VL_ModelHandler):
    def __init__(self):
        super().__init__()
        
    def _load_model(self, config ):
        MODEL_CARD  = config["model_card"]
        DO_QUANT   = config["fine_tune"]["do_quant"]
        print("Quanization it's enabled :", DO_QUANT)
        print("MODEL_CARD :", MODEL_CARD)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16
        )
        base_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            MODEL_CARD,
            quantization_config=bnb_config if DO_QUANT else None,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        return base_model

    def load_vlm(self, config):
        MODEL_CARD  = config["model_card"]
        self.model = self._load_model(config)

        self.lora_config = LoraConfig(
            **config["fine_tune"]["lora_config"]
            )
       
        if config["fine_tune"]["use_lora"]:
            print("Loading PEFT model")
            self.model = prepare_model_for_kbit_training(self.model, 
                                                        use_gradient_checkpointing=True,
                                                        gradient_checkpointing_kwargs=config["SFTConfig"]["gradient_checkpointing_kwargs"] 
                                                        )
            
            self.model = get_peft_model(self.model, self.lora_config)
            self.model.print_trainable_parameters()
        else:
            print("Loading FULL model")
            for name, param in self.model.named_parameters():

                if any(tune_layer in name for tune_layer in config["fine_tune"]["normal_tune"]):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self.processor = LlavaOnevisionProcessor.from_pretrained(MODEL_CARD, 
                                                            trust_remote_code=True)
        # for name, param in self.model.named_parameters():
        #     print(f"Layer: {name} | Size: {param.size()} | Requires Grad: {param.requires_grad}")
        self.collator = TrainDataCollator(self.processor)

    def load_finetuned_vlm(self, config, checkpoint_dir, is_peft_tuned,for_pie_yolo=False, for_yolo=False):
        MODEL_CARD  = config["model_card"]
        if is_peft_tuned:
            print("Loading PEFT model")
            base_model = self._load_model(config)
            self.model = PeftModel.from_pretrained(base_model, 
                                                    checkpoint_dir,
                                                    torch_dtype=torch.bfloat16,
                                                    attn_implementation="flash_attention_2")
        else:
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(checkpoint_dir,
                                                                                attn_implementation="flash_attention_2").to(self.device)
            # self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(MODEL_CARD).to(self.device)

        self.processor = LlavaOnevisionProcessor.from_pretrained(MODEL_CARD, 
                                                            trust_remote_code=True)
        # self.model = torch.compile(self.model)
        self.model.eval()
        # Critical Fixes for Generation
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        self.processor.tokenizer.padding_side = "left" # set to 'left' for generation and 'right' for training (default in 'right')
        self.collator = TestDataCollator(self.processor)

    def inference(self, inputs, image_inputs):
        with torch.inference_mode():

            texts = self.processor.apply_chat_template(inputs,
                                    tokenize=False,
                                    add_generation_prompt=True)
            model_inputs = self.processor(
                text=texts, # text dict keys ['input_ids', 'attention_mask', 'pixel_values', 'image_sizes']
                images=image_inputs,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            input_length = model_inputs["input_ids"].shape[1]

            outputs = self.model.generate(**model_inputs, do_sample=False, max_new_tokens=200)
            output_text = self.processor.decode(outputs[0, input_length: ], 
                                                skip_special_tokens=True, 
                                                clean_up_tokenization_spaces=False)
        return output_text

    def train(self, train_dataset, valid_dataset, training_args, use_lora):
        print("===Train_dataset length", len(train_dataset))
        print("===Valid_dataset length", len(valid_dataset))

        # preprocess_logits_for_metrics = Metrics_Preprocessor(self.processor)
        trainer = SFTTrainer(
                    model                           = self.model,
                    args                            = training_args,
                    train_dataset                   = train_dataset,
                    eval_dataset                    = valid_dataset,
                    data_collator                   = self.collator,
                    peft_config                     = self.lora_config if use_lora else None,
                    processing_class                = self.processor.tokenizer,
                    # compute_metrics                 = compute_metrics,
                    # preprocess_logits_for_metrics   = preprocess_logits_for_metrics,
                    # compute_loss_func               = self.compute_loss_func    
                )

        trainer.train()
        trainer.save_model(training_args.output_dir)