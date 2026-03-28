import os
import numpy as np
from keras.models import Sequential, Model
from keras.layers import (
    Dense,
    Input,
)
from keras import optimizers


class Classifier:
    """Binary neural network classifier.

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
        self._model = self._build(learning_rate, beta_1)

    @property
    def model(self) -> Model:
        """The underlying Keras model."""
        return self._model

    def _build(self, learning_rate: float, beta_1: float) -> Sequential:
        """Build and compile the classifier network."""
        model = Sequential(
            [
                Input(shape=self._input_shape),
                Dense(20, activation="relu"),
                Dense(20, activation="relu"),
                Dense(2, activation="softmax"),
            ],
            name="classifier",
        )
        optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1)
        model.compile(optimizer, "binary_crossentropy")
        return model

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
        history = self._model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_data=(X_test, y_test),
        )
        self._log_metrics(history, X_train, y_train, X_test, y_test)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
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
        return self._model.predict(X, verbose=0)

    def save(self, save_dir: str = "./") -> None:
        """Save the model to disk.

        Parameters
        ----------
        save_dir : str
            Directory where ``classifier.keras`` will be written.
        """
        self._model.save(os.path.join(save_dir, "classifier.keras"))

    def _log_metrics(
        self,
        history,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Print training and validation loss and accuracy."""
        train_acc = np.mean(
            np.argmax(self._model.predict(X_train, verbose=0), axis=1)
            == np.argmax(y_train, axis=1)
        )
        val_acc = np.mean(
            np.argmax(self._model.predict(X_test, verbose=0), axis=1)
            == np.argmax(y_test, axis=1)
        )
        print(
            f"Classifier — train: loss={history.history['loss'][-1]:.4f}, acc={train_acc:.4f}"
        )
        print(
            f"Classifier — val:   loss={history.history['val_loss'][-1]:.4f}, acc={val_acc:.4f}"
        )
