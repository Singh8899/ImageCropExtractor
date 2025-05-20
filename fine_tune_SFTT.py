"""
This script fine-tunes a Vision-Language Model (VLM) using the SFT (Supervised Fine-Tuning) framework.

Modules:
    - argparse: For parsing command-line arguments.
    - json: For handling JSON data.
    - os: For interacting with the operating system.
    - yaml: For reading configuration files in YAML format.
    - warnings: For suppressing specific warnings.
    - torch: For PyTorch functionalities.
    - IntentDataset: Custom dataset class for loading intent data.
    - Llava_model: Custom model class for Vision-Language Model (VLM).
    - SFTConfig: Configuration class for supervised fine-tuning from the `trl` library.

Functions:
    - parse_args(): Parses command-line arguments for configuration file, training directory, and other options.
    - main(config, train_dir): Main function to configure training arguments, load the model, datasets, and start training.

Command-line Arguments:
    - -c, --config_file: Path to the YAML configuration file. Default is "./config/config.yaml".
    - -o, --train_dir: Directory to save the trained model. Default is None.
    - -d, --deepspeed_conf: Path to the DeepSpeed configuration file. Default is "./accelerate_deepseed_config/zero_stage3_off_config.json".
    - --local_rank: Local rank for distributed training. Default is -1.

Workflow:
    1. Parse command-line arguments.
    2. Load the configuration file in YAML format.
    3. Configure training arguments using `SFTConfig`.
    4. Initialize the model using `Llava_model` and load the VLM configuration.
    5. Load training and validation datasets using `IntentDataset`.
    6. Start the fine-tuning process using the `train` method of the model class.

Warnings:
    - Suppresses `FutureWarning` for specific modules (`torch.utils.checkpoint` and `trl.trainer.sft_trainer`).

Usage:
    Run the script from the command line with appropriate arguments:
    ```
    python fine_tune_SFTT.py -c <config_file> -o <train_dir> -d <deepspeed_conf>
    ```
"""



import argparse
import yaml
import warnings
# import deepspeed
from dataloader.intent_dataloader_HF    import get_train_val_datasets
from llava_model                        import Llava_model
from trl                                import SFTConfig
# warnings.filterwarnings("ignore", category=FutureWarning,   module="torch.utils.checkpoint")
# warnings.filterwarnings("ignore", category=FutureWarning,   module="trl.trainer.sft_trainer")
# warnings.filterwarnings("ignore", category=UserWarning,     module="torch.utils.checkpoint")
# warnings.filterwarnings("ignore", category=UserWarning,     module="torch._dynamo.eval_frame")


def parse_args():
    parser = argparse.ArgumentParser(description="VLM Training")
    parser.add_argument("-c",  "--config_file",         help="Configuration file.",     default="./config/config.yaml")
    parser.add_argument("-o",  "--train_dir",           help="Train output directory",  default="/root/Workspace/ImageCropExtractor/trains",   metavar="") 
    return parser.parse_args()

def main(config, train_dir):
    # Configure training arguments
    training_args = SFTConfig(
        output_dir=train_dir,  # Directory to save the model
        **config["SFTConfig"]
    )
    # MODEL IMPORT
    model_class = Llava_model()
    model_class.load_vlm(config)
    # DATASET IMPORT
    train_dataset, valid_dataset = get_train_val_datasets()
    model_class.train(train_dataset, valid_dataset, training_args, config["fine_tune"]["use_lora"])

if __name__ == '__main__':
    args = parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    main(config, args.train_dir)