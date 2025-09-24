import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import yaml
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.resnet_pim import ResNetPIM
from data.dataset import CustomDataset # Assuming you have a custom dataset class

def evaluate(config, checkpoint_path):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations
    data_transforms = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataloader
    # You need to replace this with your actual dataset
    test_dataset = CustomDataset(root_dir=config['data_root'], split='test', transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # Model
    model = ResNetPIM(num_classes=config['num_classes'], pretrain=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # Evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='pipeline/config.yaml', help='Path to the config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    evaluate(config, args.checkpoint)