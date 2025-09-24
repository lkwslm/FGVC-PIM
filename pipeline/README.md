# ResNet-PIM Training and Evaluation Pipeline

This pipeline provides scripts to train and evaluate the ResNet-PIM model.

## 1. Using the Main Script (Recommended)

The `main.py` script provides a convenient way to run both training and evaluation.

### Training

To train the model, run the following command:

```bash
python pipeline/main.py --mode train --config pipeline/config.yaml
```

This will first train the model and then automatically run evaluation on the last checkpoint.

### Evaluation

To evaluate a specific model checkpoint, run the following command, replacing `<checkpoint_path>` with the path to your saved model checkpoint:

```bash
python pipeline/main.py --mode evaluate --config pipeline/config.yaml --checkpoint <checkpoint_path>
```

## 2. Manual Execution

Alternatively, you can run the training and evaluation scripts separately.

### Configuration

The hyperparameters and data paths are configured in `pipeline/config.yaml`. You can modify this file to suit your needs.

### Training

To train the model, run the following command:

```bash
python pipeline/train.py --config pipeline/config.yaml
```

Checkpoints will be saved in the `pipeline` directory.

### Evaluation

To evaluate a trained model, run the following command, replacing `<checkpoint_path>` with the path to your saved model checkpoint:

```bash
python pipeline/evaluate.py --config pipeline/config.yaml --checkpoint <checkpoint_path>
```