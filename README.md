# SimCLR with TensorFlow 2

This repository implements **SimCLR**, a self-supervised learning framework for visual representations, using TensorFlow 2. SimCLR is designed to learn robust feature representations from unlabeled images through contrastive learning. This implementation is inspired by the paper:

> [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) by Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton

---

The paper introduces the SimCLR framework, which is targeted to improve the representations created by models with contrastive prediction tasks. The authors of the paper performed various experiments and used their findings to create an optimal contrastive learning framework for visual representations called SimCLR. The main goal I pursue in this project is to reproduce the two key findings in the original paper:
1. composition of data augmentations plays a critical role in defining ef ective predictive tasks
2. introducing a learnable nonlinear transformation between the representation and the contrastive loss substantially improves the quality of the learned representations.


The two key findings were then combined to create a final contrastive learning framework for visual representations called SimCLR.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- Self-supervised pretraining using contrastive loss.
- Support for data augmentation strategies (color jittering, cropping, flipping, etc.).
- Modular design for model definition, training, and evaluation.
- Transfer learning capabilities for downstream tasks.
- TensorFlow 2 compatible.

---

## Requirements

To run this project, ensure you have the following installed:

- Python >= 3.8
- TensorFlow >= 2.10
- NumPy
- Matplotlib
- tqdm
- OpenCV (optional, for advanced augmentations)

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/SamiraHajizadeh/SimCLR-with-Tensorflow2.git
cd SimCLR-with-Tensorflow2
```

---

## Usage

## Pretrained Models
Pretrained models can be found here:
https://drive.google.com/drive/folders/1Ya7itCLNu4UeWx-ln83tc1NOOqR29RoF?usp=sharing

### Training

To train the SimCLR model on a dataset, use the following command:

```bash
python train.py --data_dir /path/to/dataset --batch_size 128 --epochs 100
```

Available training arguments:

- `--data_dir`: Path to the dataset directory (default: `./data`).
- `--batch_size`: Batch size for training (default: `128`).
- `--epochs`: Number of training epochs (default: `100`).
- `--temperature`: Temperature parameter for contrastive loss (default: `0.5`).
- `--learning_rate`: Initial learning rate (default: `0.001`).

### Evaluation

To evaluate the quality of representations on a downstream task:

```bash
python evaluate.py --model_path /path/to/saved_model --data_dir /path/to/eval_dataset
```

---

## Configuration

The training and evaluation settings can be customized through `config.yaml`. Key configurations include:

- Augmentation settings (e.g., cropping, flipping, color jittering).
- Model architecture and hyperparameters.
- Paths for saving models and logs.

Example `config.yaml` snippet:

```yaml
batch_size: 128
epochs: 100
temperature: 0.5
learning_rate: 0.001
augmentations:
  crop: True
  flip: True
  color_jitter: True
```

---

## Project Structure

```plaintext
SimCLR-with-Tensorflow2/
├── data/                  # Dataset directory (not included in repo)
├── models/                # Pretrained and saved models
├── src/                   # Source code for training and evaluation
│   ├── data_loader.py     # Data loading and augmentation scripts
│   ├── model.py           # SimCLR model definition
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   ├── utils.py           # Utility functions
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve this project.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
