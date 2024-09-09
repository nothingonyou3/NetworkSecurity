import numpy as np
import torch
from torch.utils.data import DataLoader
from autoencoder import SKSAutoencoder
from multi_head_attention import SKSAutoencoderMultiheadSA
from residual import SKSAutoencoderResidual
from classifier import TrafficClassifier
from datasets import KDDDataset, split_dataset
from training import CoolUniversalModelTrainer

# - IMPORTING THE DATASET - #
dataset = KDDDataset("./datasets/KDDTrain+.txt",
                     cols_to_drop=["Difficulty Level"],  # IDK what this is for, therefore I delete it
                     overwrite_cache=False,
                     downcast_binary=True)  # We'll perform only a binary classification
train_dataset, test_dataset = split_dataset(dataset, 0.8)

validation_dataset = KDDDataset("./datasets/KDDTest+.txt",
                                cols_to_drop=["Difficulty Level"],
                                overwrite_cache=False,
                                downcast_binary=True)
validation_dataset.like(dataset)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

validation_dataloader = DataLoader(validation_dataset, batch_size=128, shuffle=True)

# - MODEL SETTINGS - #

# Autoencoder:
A_input_dim = dataset.n_features
A_hidden_dim = 24  # Autoencoder hidden dimension (bottleneck size)
A_k = A_hidden_dim // 6  # Parameter for the K sparse layer

# Classifier:
C_input_dim = dataset.n_features
C_output_dim = dataset.n_labels

# Then we create the models
# autoencoder = SKSAutoencoder(input_dim=A_input_dim, hidden_dim=A_hidden_dim, k=A_k, n_of_transitional_layers=2)
autoencoder = SKSAutoencoderResidual(input_dim=A_input_dim,
                                     hidden_dim=A_hidden_dim,
                                     k=A_k,
                                     n_of_transitional_layers=2)
classifier = TrafficClassifier(input_dim=C_input_dim, output_dim=C_output_dim)

# - TRAINING PARAMETERS - #

# Autoencoder:
A_learning_rate = 0.001
A_n_epochs = 70

A_criterion = torch.nn.MSELoss()
A_adam = torch.optim.Adam(lr=A_learning_rate, weight_decay=1e-7, params=autoencoder.parameters())

# Classifier:
C_learning_rate = 0.001
C_n_epochs = 100

C_BCE = torch.nn.BCELoss()  # For binary classification
C_CEL = torch.nn.CrossEntropyLoss()  # For n of class > 2 (more than -one- binary label)
C_criterion = C_CEL if dataset.n_labels > 1 else C_BCE
C_SGD = torch.optim.SGD(params=classifier.parameters(), lr=C_learning_rate)  # Stochastic gradient descend


# Function to evaluate the models
def model_evaluation(model_trainer: CoolUniversalModelTrainer):
    print(">>>: Model evaluation...")

    train_set_score = model_trainer.evaluate_model(train_dataloader)
    test_set_score = model_trainer.evaluate_model(test_dataloader)
    validation_set_score = model_trainer.evaluate_model(validation_dataloader)

    print(f"Train set score:\t\t {(1 - train_set_score) * 100:4.2f} %\n"
          f"Test set score: \t\t {(1 - test_set_score) * 100:4.2f} %\n"
          f"Validation set score:\t {(1 - validation_set_score) * 100:4.2f} %")


# - TRAINING THE MODELS - #
# Enabling or disabling training for
P1_train = False
P1_save_as = "./saved_weights/Phase_01_initial_autoencoder_training_RESIDUALS.pth"
P2_train = False
P2_save_as = "./saved_weights/Phase_02_combined_model_initial_training_RESIDUALS.pth"
P3_train = False
P3_save_as = "./saved_weights/Phase_03_combined_model_initial_training_RESIDUALS.pth"

# -- PHASE 1: we train the autoencoder -- #
print(">>>: Phase 1 initialized: training the autoencoder standalone")
# Autoencoder:
if P1_train:
    autoencoder_trainer = CoolUniversalModelTrainer(model=autoencoder,
                                                    early_stopping=True,
                                                    mode="Autoencoder",
                                                    optimizer=A_adam,
                                                    criterion=A_criterion)

    autoencoder_trainer.train_model(train_loader=train_dataloader,
                                    test_loader=test_dataloader,
                                    epochs=A_n_epochs,
                                    save_as=P1_save_as)  # we keep saving the best version of the model till we go

print(">>>: Loading stored weights...")
autoencoder.load_state_dict(torch.load(P1_save_as, weights_only=True))  # loading best model version

# -- PHASE 2: we freeze the weights in the autoencoder, then we train the classifier with the data coming out of it -- #
print(">>>: Phase 2 initialized: training the combined model with autoencoder weights freezed")
for param in autoencoder.parameters():
    param.requires_grad = False  # freezing the weights

combined_model = torch.nn.Sequential(  # We combine the two models
    autoencoder,
    classifier
)
# autoencoder = combined_model[0]
# classifier  = combined_model[1]

combined_model_trainer = CoolUniversalModelTrainer(model=combined_model,
                                                   early_stopping=True,
                                                   mode="Default",
                                                   optimizer=C_SGD,
                                                   criterion=C_criterion)

if P2_train:

    combined_model_trainer.train_model(train_loader=train_dataloader,
                                       test_loader=test_dataloader,
                                       epochs=C_n_epochs,
                                       save_as=P2_save_as)

print(">>>: Loading stored weights...")
combined_model.load_state_dict(torch.load(P2_save_as, weights_only=True))  # loading best model version

model_evaluation(combined_model_trainer)

# -- PHASE 3: we fine-tune the entire model -- #
print(">>>: Phase 3 initialized: fine tuning of the combined model")
for param in combined_model[0].parameters():
    param.requires_grad = True  # unfreeze the weights


# Layer-wise Learning Rate Decay (LLRD)
def set_model_lr_decay(model, lr_decay, base_lr, param_list, index=0):
    for layer in model.children():
        lr = base_lr * (lr_decay ** index)
        optimizer_grouped_parameters.append({'params': layer.parameters(), 'lr': lr})
        if isinstance(layer, torch.nn.Sequential):
            index = set_model_lr_decay(layer, lr_decay, base_lr, param_list, index=index + 1)
        else:
            index += 1
    return index


c_base_lr = 1e-4
c_lr_decay = 0.85
optimizer_grouped_parameters = []
set_model_lr_decay(model=combined_model,
                   lr_decay=c_lr_decay,
                   base_lr=c_base_lr,
                   param_list=optimizer_grouped_parameters)

fine_tuning_optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
P3_epochs = 50

combined_model_trainer = CoolUniversalModelTrainer(model=combined_model,
                                                   early_stopping=True,
                                                   mode="Default",
                                                   optimizer=fine_tuning_optimizer,
                                                   criterion=C_criterion)

if P3_train:

    combined_model_trainer.train_model(train_loader=train_dataloader,
                                       test_loader=test_dataloader,
                                       epochs=P3_epochs,
                                       save_as=P3_save_as)

print(">>>: Loading stored weights...")
combined_model.load_state_dict(torch.load(P3_save_as, weights_only=True))  # loading best model version

model_evaluation(combined_model_trainer)
