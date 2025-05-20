"""
This script performs inference without YOLO(BBoxes from PIe dataset) for a Vision-Language Model (VLM) using a fine-tuned model checkpoint. 
It evaluates the model on a test dataset and computes various metrics such as accuracy, precision, recall, and F1-score.
Functions:
----------
- get_best_checkpoint(checkpoint_dir, metric):
    Identifies the best model checkpoint based on a specified evaluation metric (e.g., F1-score, Recall, or loss).
    Args:
        checkpoint_dir (str): Path to the directory containing model checkpoints.
        metric (str): Metric to optimize ('F1-score', 'Recall', or 'loss').
    Returns:
        tuple: Path to the best checkpoint and a boolean indicating if the model is PEFT-tuned.
- parse_args():
    Parses command-line arguments for the script.
    Returns:
        argparse.Namespace: Parsed arguments including config file, test directory, checkpoint directory, metric, and save flag.
- main(config, test_dir, checkpoint_dir, save, is_peft_tuned):
    Performs inference using the specified model and test dataset.
    Args:
        config (dict): Configuration dictionary loaded from a YAML file.
        test_dir (str): Directory to save test results and metrics.
        checkpoint_dir (str): Path to the best model checkpoint.
        save (bool): Flag to save test images.
        is_peft_tuned (bool): Indicates if the model is PEFT-tuned.
    Returns:
        None
Usage:
------
Run the script from the command line with the following arguments:
    -c, --config_file: Path to the configuration YAML file (default: "./config/config.yaml").
    -o, --test_dir: Directory to save test results.
    -ch, --checkpoint_dir: Directory containing model checkpoints.
    -m, --metric: Metric to optimize (choices: "F1-score", "Recall", "loss"; default: "F1-score").
    -s, --save: Flag to save test images (optional).
Example:
--------
python inference_SFTT.py -c ./config/config.yaml -o ./test_results -ch ./checkpoints -m F1-score -s
Dependencies:
-------------
- argparse
- json
- yaml
- os
- time
- torch
- tqdm
- sklearn.metrics
- warnings
- tensorboard.backend.event_processing
Notes:
------
- The script suppresses FutureWarnings from the torch.utils.checkpoint module.
- The model class (`Llava_model`) and dataloader class (`IntentDataloader`) are imported from external modules.
- Metrics are saved in a JSON file in the specified test directory.
"""

import argparse
import json
import yaml
import os
from time import time
import torch
from dataloader.intent_dataloader_HF    import IntentDataloader
from tqdm                               import tqdm
from llava_model                        import Llava_model
from sklearn.metrics                    import accuracy_score, precision_score, recall_score, f1_score
import warnings
from tensorboard.backend.event_processing import event_accumulator
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")


def get_best_checkpoint(checkpoint_dir, metric):
    is_peft_tuned = False
    for root, dirs, files in os.walk(checkpoint_dir):
        if "adapter_model.safetensors" in files:
            is_peft_tuned = True
            break


    best_checkpoint = None
    best_metric_value = float('-inf') if metric != 'loss' else float('inf')
    metric = "eval/"+metric
    run_dir = os.path.join(checkpoint_dir, "runs")
    for root, dirs, files in os.walk(run_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                ea = event_accumulator.EventAccumulator(os.path.join(root, file))
                ea.Reload()
                for event in ea.Scalars(metric):
                    if (metric == 'eval/loss' and event.value < best_metric_value) or (metric != 'eval/loss' and event.value > best_metric_value):
                        best_metric_value = event.value
                        best_checkpoint = "checkpoint-" + str(event.step)
                
    print("Best checkpoint for metric: ",metric," -->",best_checkpoint )
    return os.path.join(checkpoint_dir, best_checkpoint), is_peft_tuned

# def prediction_parser(string):
#     raw = string.strip("'[] '").split(",")
#     return [i.strip("'[] '") for i in raw]

def parse_args():
    parser = argparse.ArgumentParser(description="VLM Training")
    parser.add_argument("-c",  "--config_file", help="Configuration file.",     default="./config/config.yaml")
    parser.add_argument("-o",  "--test_dir",   help="Test output directory",  metavar="") 
    parser.add_argument("-ch",  "--checkpoint_dir",   help="Checkpoint directory",  metavar="") 
    parser.add_argument("-m", "--metric", help="Metric to optimize (F1-score, Recall, loss)", choices=["F1-score", "Recall", "loss"], default="F1-score")
    parser.add_argument("-s", "--save", help="Save images", action='store_true')
    return parser.parse_args()

def main(config, test_dir, checkpoint_dir, save, is_peft_tuned):
    # MODEL IMPORT
    model_class = Llava_model()
    model_class.load_finetuned_vlm(config, checkpoint_dir, is_peft_tuned)
    # DATALOADER IMPORT
    collator = model_class.get_collator()
    dataloader_class = IntentDataloader(config, collator)
    test_dataloader = dataloader_class.create_test_dataloader()
    pbar = tqdm(test_dataloader, desc=f"Testing",leave=True)
    preds = []
    gts = []
    misclassified = 0
    with torch.inference_mode():
        for batch in pbar:
            # batch = {k: v for k, v in input[0].items() if k != "labels"}
            video_inputs = batch[2]
            old_time = time()
            pred = model_class.inference(batch[0], video_inputs)
            print("Time for inference: ", time()-old_time)
            gt = batch[1]
            print("pred",pred)
            print("gt",gt)
            # pred = [prediction_parser(p) for p in pred]
            # pred_list = [i for p in pred for i in p]
            preds.extend([1 if "YES" in i.upper()  else 0 for i in pred])
            gts.extend([1 if "YES" in i.upper() else 0 for i in gt])
            misclassified += sum([1 for i in pred if ("YES" not in i.upper() and "NO" not in i.upper())] )
            accuracy = accuracy_score(gts, preds)
            precision = precision_score(gts, preds, pos_label=0, zero_division=0)
            recall = recall_score(gts, preds, pos_label=0, zero_division=0)
            f1 = f1_score(gts, preds, pos_label=0, zero_division=0)
            
            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "miscalssified": misclassified
            }
            pbar.set_postfix(metrics)
            if save:
                model_class.save(test_dir, batch[3], gt, pred, resize=False)
            
        os.makedirs(test_dir, exist_ok=True)
        metrics_file = os.path.join(test_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            metrics["test_dir"] = test_dir
            json.dump(metrics, f, indent=4) 

if __name__ == '__main__':
    args = parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    best_checkpoint, is_peft_tuned = get_best_checkpoint(args.checkpoint_dir, args.metric)
    main(config, args.test_dir, best_checkpoint, args.save, is_peft_tuned)