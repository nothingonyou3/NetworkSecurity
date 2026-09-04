# NetworkSecurity
Project implementation

The project is composed of a main.py file that can be used to run the experiments. All the following code is contained into the main.py. Depeding on the test that you want to perform you can decomment the one you need to use and comment the other 2.
To run the basic version is necessary to use this model:
autoencoder = SKSAutoencoder(input_dim=A_input_dim, hidden_dim=A_hidden_dim, k=A_k, n_of_transitional_layers=2).

For the multi-head attention test:
autoencoder = SKSAutoencoderMultiheadSA(input_dim=A_input_dim,
                                        hidden_dim=A_hidden_dim,
                                        k=A_k,
                                        num_heads=8,
                                        n_of_transitional_layers=2)

while for the residual test:
autoencoder = SKSAutoencoderResidual(input_dim=A_input_dim,
                                     hidden_dim=A_hidden_dim,
                                     k=A_k,
                                     n_of_transitional_layers=2).

The file training.py contains the command to perform the training of the project while autoencoder.py contains the basic version, multi_head_attention.py contains the multi-head attention version while the residual one is performed by residual.py.
Results.txt presents a report of the results obtained.
Classifier.py presents the logic for a binary classification.
The Excel file explains the meaning of each field in the dataset.
In order to perform the test is necessary to put the files in this order and create a folder called: datasets and another called saved_weights as reported in the photo.

![image](https://github.com/user-attachments/assets/51e87838-8c52-4597-a960-5da2cef932ee)

A more detailed version of the redme is reported here:
# Network Intrusion Detection with K-Sparse Autoencoders

Deep learning project for **network intrusion detection** using a **K-Sparse Autoencoder** combined with a neural network classifier.

The project explores the use of sparse latent representations for network traffic analysis and classification, with an architecture incorporating **residual connections** and a **three-phase training strategy**.

## Overview

The system is designed to learn compact representations of network traffic and use them to classify traffic into different categories.

The pipeline is based on the **KDD intrusion detection dataset** and consists of two main components:

1. A **K-Sparse Autoencoder** that learns a compressed representation of network traffic features.
2. A **Traffic Classifier** that uses the learned representation for intrusion classification.

The autoencoder can be configured with different architectures, including standard, multi-head self-attention, and residual variants. The current experiment uses the **residual K-Sparse Autoencoder**.

## Architecture

The overall pipeline is:

```text
Network Traffic Features
          │
          ▼
┌─────────────────────────┐
│  K-Sparse Autoencoder   │
│                         │
│  Input → Encoder        │
│          ↓              │
│     K-Sparse Layer      │
│          ↓              │
│       Bottleneck        │
│          ↓              │
│       Decoder           │
└─────────────────────────┘
          │
          ▼
   Learned Representation
          │
          ▼
┌─────────────────────────┐
│   Traffic Classifier    │
└─────────────────────────┘
          │
          ▼
    Predicted Class
```

The current configuration uses a **24-dimensional bottleneck**, with the K-sparse parameter set to:

```python
A_hidden_dim = 24
A_k = A_hidden_dim // 6
```

## Dataset

The project uses the **KDD intrusion detection dataset**, with:

* `KDDTrain+.txt` for training
* `KDDTest+.txt` for validation/evaluation

The `Difficulty Level` feature is removed from the dataset.

The data is configured for classification using binary label encoding where applicable.

The training dataset is further split into:

* **80% training**
* **20% testing**

The official `KDDTest+` dataset is used as an additional validation set.

## Models

Several autoencoder implementations are available in the project:

```python
from autoencoder import SKSAutoencoder
from multi_head_attention import SKSAutoencoderMultiheadSA
from residual import SKSAutoencoderResidual
```

The current experiment uses:

```python
autoencoder = SKSAutoencoderResidual(
    input_dim=A_input_dim,
    hidden_dim=A_hidden_dim,
    k=A_k,
    n_of_transitional_layers=2
)
```

The classifier is implemented as:

```python
classifier = TrafficClassifier(
    input_dim=C_input_dim,
    output_dim=C_output_dim
)
```

## Training Strategy

Training is divided into three phases.

### Phase 1 — Autoencoder Pre-training

The autoencoder is trained independently to reconstruct the input data.

```text
Input
  │
  ▼
Encoder
  │
  ▼
K-Sparse Bottleneck
  │
  ▼
Decoder
  │
  ▼
Reconstructed Input
```

The reconstruction objective uses Mean Squared Error:

```python
A_criterion = torch.nn.MSELoss()
```

The autoencoder is trained using Adam with:

* Learning rate: `0.001`
* Weight decay: `1e-7`
* Maximum epochs: `70`
* Early stopping enabled

The best model checkpoint is saved and subsequently loaded for the next phase.

### Phase 2 — Classifier Training

The pretrained autoencoder is combined with the traffic classifier:

```python
combined_model = torch.nn.Sequential(
    autoencoder,
    classifier
)
```

During this phase, the autoencoder parameters are **frozen**:

```python
for param in autoencoder.parameters():
    param.requires_grad = False
```

Only the classifier is trained.

The classifier uses SGD with:

* Learning rate: `0.001`
* Maximum epochs: `100`
* Early stopping enabled

The loss function is selected according to the number of labels:

```python
C_criterion = C_CEL if dataset.n_labels > 1 else C_BCE
```

### Phase 3 — Fine-Tuning

In the final stage, the autoencoder is unfrozen and the entire model is fine-tuned jointly.

```text
Autoencoder
     +
Classifier
     │
     ▼
End-to-end fine-tuning
```

The model uses **AdamW** together with **Layer-wise Learning Rate Decay (LLRD)**.

The base learning rate is:

```python
c_base_lr = 1e-4
```

with:

```python
c_lr_decay = 0.85
```

This allows different layers of the network to be updated with progressively smaller learning rates.

The fine-tuning stage runs for up to **50 epochs** with early stopping.

## Training Pipeline

The complete training workflow can be summarized as:

```text
             KDD Dataset
                  │
                  ▼
          Data preprocessing
                  │
                  ▼
        Train / Test split
                  │
                  ▼
      ┌─────────────────────┐
      │ Phase 1             │
      │ Autoencoder         │
      │ pre-training        │
      └──────────┬──────────┘
                 │
                 ▼
        Load best weights
                 │
                 ▼
      ┌─────────────────────┐
      │ Phase 2             │
      │ Freeze Autoencoder  │
      │ Train Classifier    │
      └──────────┬──────────┘
                 │
                 ▼
        Load best weights
                 │
                 ▼
      ┌─────────────────────┐
      │ Phase 3             │
      │ Unfreeze Model      │
      │ Fine-tuning + LLRD  │
      └──────────┬──────────┘
                 │
                 ▼
             Evaluation
```

## Evaluation

After each training stage, the model is evaluated on:

* Training set
* Test set
* `KDDTest+` validation set

Evaluation is performed through the `CoolUniversalModelTrainer` class.

The script reports the resulting scores as percentages:

```text
Train set score:       XX.XX %
Test set score:        XX.XX %
Validation set score:  XX.XX %
```

## Project Structure

A possible repository structure is:

```text
.
├── datasets/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
│
├── saved_weights/
│   ├── Phase_01_initial_autoencoder_training_RESIDUALS.pth
│   ├── Phase_02_combined_model_initial_training_RESIDUALS.pth
│   └── Phase_03_combined_model_initial_training_RESIDUALS.pth
│
├── autoencoder.py
├── multi_head_attention.py
├── residual.py
├── classifier.py
├── datasets.py
├── training.py
├── main.py
└── README.md
```

## Requirements

The project requires Python and the following main libraries:

```text
numpy
torch
```

Install the dependencies with:

```bash
pip install numpy torch
```

Additional dependencies may be required depending on the implementation of the supporting modules.

## Running the Project

Place the KDD dataset files inside the `datasets/` directory:

```text
datasets/
├── KDDTrain+.txt
└── KDDTest+.txt
```

Before running the training script, configure the training flags:

```python
P1_train = False
P2_train = False
P3_train = False
```

Set the corresponding flag to `True` to enable a training phase.

For example:

```python
P1_train = True
```

will train the autoencoder from scratch.

After training, the best checkpoint is saved to the `saved_weights/` directory.




