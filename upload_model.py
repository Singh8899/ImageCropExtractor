#!/usr/bin/env python3
"""
Script per uploadare il modello fine-tuned su Hugging Face
"""
import argparse
import os

from unsloth import FastVisionModel


def upload_model(checkpoint_path, output_model_name, hf_token):
    """
    Carica e uploada il modello dal checkpoint specificato
    """
    print(f"Caricamento del modello dal checkpoint: {checkpoint_path}")

    # Carica il modello dal checkpoint
    model, tokenizer = FastVisionModel.from_pretrained(
        checkpoint_path,
        load_in_4bit=True,
        use_gradient_checkpointing=False,
        max_seq_length=50000,
    )

    print(f"Uploading del modello su Hugging Face come: {output_model_name}")

    # Upload su Hugging Face
    model.push_to_hub_merged(
        output_model_name,
        tokenizer,
        token=hf_token,
        save_method="merged_16bit",
    )

    print(
        f"✅ Modello uploadato con successo su: https://huggingface.co/{output_model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload del modello fine-tuned su Hugging Face")
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        default="outputs_checkpoint/checkpoint-950",
        help="Path al checkpoint da uploadare (default: outputs_checkpoint/checkpoint-950)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Nome del repository su Hugging Face (es: username/model-name)"
    )
    parser.add_argument(
        "-t", "--token",
        type=str,
        required=True,
        help="Token di Hugging Face per l'upload"
    )

    args = parser.parse_args()

    # Verifica che il checkpoint esista
    if not os.path.exists(args.checkpoint):
        print(f"❌ Errore: Il checkpoint {args.checkpoint} non esiste!")
        return

    # Esegui l'upload
    upload_model(args.checkpoint, args.output, args.token)


if __name__ == "__main__":
    main()
