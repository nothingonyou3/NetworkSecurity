import torch


class CoolUniversalModelTrainer:
    default_early_stopping_patience = 3

    def __init__(self, model, early_stopping=False, mode: str = "Default", criterion=None, optimizer=None):
        self.model = model
        self.mode = mode
        self.early_stopping = early_stopping
        self.criterion = criterion
        self.optimizer = optimizer
        self.patience = self.__class__.default_early_stopping_patience
        # These two only serve informational purposes
        self.__initial_loss = None
        self.__previous_loss = None

    def __compute_loss(self, features, labels=None):
        if self.criterion is None:
            raise ValueError("criterion is not set")
        if self.mode == "Autoencoder":
            reconstructed = self.model(features)
            return self.criterion(features, reconstructed)
        elif self.mode == "Default":
            outputs = self.model(features)
            return self.criterion(outputs, labels)

    def __evaluate_model(self, test_dataloader):
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i, (features, labels) in enumerate(test_dataloader):
                features = features.float()
                batch_loss = self.__compute_loss(features=features, labels=labels)
                test_loss += batch_loss.item()

        test_loss /= len(test_dataloader)

        return test_loss

    def __compute_training_test_metrics(self, test_loss):
        if self.__initial_loss is None:
            self.__initial_loss = test_loss  # We set the initial loss only during the first epoch
            improvement_from_beginning = 0
        else:
            improvement_from_beginning = (self.__initial_loss - test_loss) / self.__initial_loss * 100

        if self.__previous_loss is None:
            improvement_from_previous = 0
        else:
            improvement_from_previous = (self.__previous_loss - test_loss) / self.__previous_loss * 100
        self.__previous_loss = test_loss

        return improvement_from_beginning, improvement_from_previous

    def __train_epoch(self, train_dataloader):
        if self.optimizer is None:
            raise ValueError("optimizer is not set")
        self.model.train()
        train_loss = 0.0
        # Actual training
        for i, (features, labels) in enumerate(train_dataloader):
            features = features.float()
            self.optimizer.zero_grad()
            loss = self.__compute_loss(features=features, labels=labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        return train_loss

    def train_model(self, epochs, train_loader, test_loader, save_as=None):
        """
       Trains the given model with the provided datasets.

       :param train_loader: DataLoader for the training set
       :param test_loader: DataLoader for the test set
       :param optimizer: Optimizer to use for training
       :param criterion: Loss function to minimize
       :param epochs: Number of training epochs
       :param save_as: Path to save the best model
       :return: The trained model
        """
        print(f"├──>>: Training the selected model ({epochs} epochs)")
        if self.early_stopping:
            print(f"├──>>: Early stopping mechanism enabled, with patience set to {self.patience} "
                  f"(default: {self.__class__.default_early_stopping_patience})")
        patience_counter = 0
        min_loss_test = float('inf')
        for epoch in range(epochs):
            # Training the model
            train_loss = self.__train_epoch(train_dataloader=train_loader)

            print(f"│   ├──Epoch {epoch + 1}/{epochs},\t\t\tLoss (average):\t{train_loss:.5f}")

            # Validation on test set (we do this for each epoch)
            test_loss = self.__evaluate_model(test_dataloader=test_loader)

            # Compute metrics (on the test_set)
            improvement_from_beginning, improvement_from_previous = self.__compute_training_test_metrics(
                test_loss=test_loss)

            # Printing information, it seems complex, but it's just printing of stuff, it's not that important either
            if test_loss < min_loss_test:
                if save_as is not None:
                    torch.save(self.model.state_dict(), save_as)  # We save the weights of the best model till now
                min_loss_test = test_loss
                hint = f" <-- New best value ({improvement_from_previous:4.2f} % improvement, " \
                       f"{improvement_from_beginning:4.2f} % from beginning)"
                patience_counter = 0
            else:
                hint = f"     No improvement ({improvement_from_beginning:4.2f} % improvement from beginning)"
                if self.early_stopping:
                    patience_counter += 1
                    if patience_counter == self.patience:
                        print(f"└──>>: ! --->>> Patience {patience_counter}/{self.patience}, stopping now <<<--- !  ")
                        return
                    hint += f" patience: {patience_counter}/{self.patience}"
            if epoch == 0:
                hint = ""
            print(f"│   │   └──Average loss on the [test set]:\t{test_loss:.10f} {hint}")

        print("└──>>: ! --->>> Training completed <<<--- !")
        return self.model

    def evaluate_model(self, test_loader):
        return self.__evaluate_model(test_dataloader=test_loader)
