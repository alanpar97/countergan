import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Classifier:
    """Binary neural network classifier (PyTorch).

    Parameters
    ----------
    input_shape : tuple[int, ...]
        Shape of a single input sample.
    learning_rate : float
        Adam optimizer learning rate.
    beta_1 : float
        Adam optimizer beta_1 parameter.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        learning_rate: float = 0.0002,
        beta_1: float = 0.5,
    ) -> None:
        self._input_shape = input_shape
        self._model = self._build()
        self._lr = learning_rate
        self._beta_1 = beta_1

    @property
    def model(self) -> nn.Module:
        """The underlying PyTorch model."""
        return self._model

    def _build(self) -> nn.Module:
        """Build the classifier network."""
        return nn.Sequential(
            nn.Linear(self._input_shape[0], 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
            nn.Softmax(dim=1),
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 32,
        epochs: int = 200,
    ) -> "Classifier":
        """Train the classifier.

        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training labels (one-hot encoded).
        X_test : np.ndarray
            Validation features.
        y_test : np.ndarray
            Validation labels (one-hot encoded).
        batch_size : int
            Number of samples per gradient update.
        epochs : int
            Number of training epochs.

        Returns
        -------
        Classifier
            The fitted instance (for method chaining).
        """
        self._model.train()
        optimizer = optim.Adam(
            self._model.parameters(), lr=self._lr, betas=(self._beta_1, 0.999)
        )
        loss_fn = nn.BCELoss()

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self._model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()

        self._log_metrics(X_train, y_train, X_test, y_test)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference on the given samples.

        Parameters
        ----------
        X : np.ndarray
            Input samples.

        Returns
        -------
        np.ndarray
            Predicted class probabilities.
        """
        self._model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32)
            return self._model(x_t).numpy()

    def save(self, save_dir: str = "./") -> None:
        """Save the model to disk.

        Parameters
        ----------
        save_dir : str
            Directory where ``classifier.pt`` will be written.
        """
        import os

        torch.save(self._model.state_dict(), os.path.join(save_dir, "classifier.pt"))

    def _log_metrics(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Print training and validation accuracy."""
        train_pred = self.predict(X_train)
        train_acc = np.mean(np.argmax(train_pred, axis=1) == np.argmax(y_train, axis=1))
        val_pred = self.predict(X_test)
        val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_test, axis=1))
        print(f"Classifier — train: acc={train_acc:.4f}")
        print(f"Classifier — val:   acc={val_acc:.4f}")
