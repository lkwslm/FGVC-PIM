import argparse
import yaml
import os
from train import train
from evaluate import evaluate

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate the ResNet-PIM model.')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate'], help='Mode to run: train or evaluate')
    parser.add_argument('--config', type=str, default='pipeline/config.yaml', help='Path to the config file')
    parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint for evaluation')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.mode == 'train':
        train(config)
        # Automatically evaluate after training
        last_checkpoint = f"pipeline/checkpoint_{config['epochs']}.pth"
        if os.path.exists(last_checkpoint):
            print("\nStarting evaluation after training...\n")
            evaluate(config, last_checkpoint)
    elif args.mode == 'evaluate':
        if not args.checkpoint:
            raise ValueError("Checkpoint path must be provided for evaluation.")
        evaluate(config, args.checkpoint)

if __name__ == '__main__':
    main()