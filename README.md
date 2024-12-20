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

### Training and Evaluation

The SimCLR.ipynb notebook is the main interface of this project.

---

## Dataset
The dataset used for pretraining the models can be found here https://www.image-net.org/challenges/LSVRC/2012/ 
Specifically, I used a part of the validation set.

---

## Pretrained Models
Pretrained models can be found here:
https://drive.google.com/drive/folders/1Ya7itCLNu4UeWx-ln83tc1NOOqR29RoF?usp=sharing

---

## Project Structure

```plaintext
SimCLR-with-Tensorflow2/
├── figures/
├── training/
│   ├── learning_rate_schedule.py
│   ├── train.py
├── utils/
│   ├── NT_Xent.py
│   ├── data_augmentation.py
│   ├── evaluation_metrics.py
│   ├── test.py
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve this project.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
