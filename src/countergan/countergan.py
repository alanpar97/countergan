"""GAN-based counterfactual explainer.

Provides the :class:`CounterGAN` class which trains a GAN to generate
counterfactual explanations for a pre-trained classifier.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, ClassVar

import numpy as np

from countergan.backends import detect_backend, get_backend

logger = logging.getLogger(__name__)


class CounterGAN:
    """GAN-based counterfactual explainer.

    Supports three strategies:

    - ``"regular_gan"``: plain GAN without residual connections.
    - ``"countergan"``: first formulation — backpropagates through a
      differentiable classifier during generator training.
    - ``"countergan_wt"``: second formulation — model-agnostic, uses
      weighted discriminator loss based on classifier scores.

    Parameters
    ----------
    strategy : str
        One of ``"regular_gan"``, ``"countergan"``, or ``"countergan_wt"``.
    classifier
        Pre-trained classifier used for counterfactual guidance.  Must
        expose a ``.predict(X) -> np.ndarray`` method returning class
        probabilities.  For ``strategy="countergan"``, must also expose a
        ``.model`` attribute returning the underlying framework-native model
        (``torch.nn.Module`` or ``keras.Model``).
    n_mutable_features : int
        Number of mutable features. Features are assumed to be ordered
        mutable-first, immutable-last. The immutable projection sets
        counterfactual values back to the original for features beyond
        this index.
    n_discriminator_steps : int
        Number of discriminator update steps per training iteration.
    n_generator_steps : int
        Number of generator update steps per training iteration.
    n_iterations : int
        Total number of training iterations.
    desired_class : int
        Target class index for the counterfactuals.
    number_of_classes : int
        Total number of classes in the classification problem.
    backend : str or None
        Backend to use: ``"torch"`` or ``"tensorflow"``.  When ``None``
        (the default), the installed framework is auto-detected (preferring
        torch).
    """

    STRATEGIES: ClassVar[dict[str, dict[str, bool]]] = {
        "regular_gan": {"residuals": False, "weighted_version": False},
        "countergan": {"residuals": True, "weighted_version": False},
        "countergan_wt": {"residuals": True, "weighted_version": True},
    }

    def __init__(
        self,
        strategy: str,
        classifier: Any,
        n_mutable_features: int,
        n_discriminator_steps: int = 2,
        n_generator_steps: int = 4,
        n_iterations: int = 2000,
        desired_class: int = 1,
        number_of_classes: int = 2,
        backend: str | None = None,
    ) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}. Choose from {list(self.STRATEGIES)}"
            )

        config = self.STRATEGIES[strategy]
        self._strategy = strategy
        self._residuals: bool = config["residuals"]
        self._weighted_version: bool = config["weighted_version"]

        backend_name = detect_backend(backend)
        self._backend = get_backend(backend_name)

        self._backend.validate_classifier(
            classifier,
            needs_gradient_flow=not self._weighted_version,
        )

        self._classifier = classifier
        self._n_mutable_features = n_mutable_features

        self._n_discriminator_steps = n_discriminator_steps
        self._n_generator_steps = n_generator_steps
        self._n_iterations = n_iterations
        self._desired_class = desired_class
        self._number_of_classes = number_of_classes

        self._generator: Any | None = None
        self._discriminator: Any | None = None
        self._is_fitted = False

    @property
    def generator(self) -> Any | None:
        """The trained generator model. ``None`` before :meth:`fit` is called."""
        return self._generator

    @property
    def _fitted_generator(self) -> Any:
        if self._generator is None:
            raise RuntimeError(
                "Generator is not initialized. Call fit() first."
            )
        return self._generator

    @property
    def _fitted_discriminator(self) -> Any:
        if self._discriminator is None:
            raise RuntimeError(
                "Discriminator is not initialized. Call fit() first."
            )
        return self._discriminator

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> CounterGAN:
        """Train the GAN. This will initialize the generator and discriminator models and run the
        training loop. After calling this method, the explainer is ready to generate counterfactuals, specifically
        for the desired class specified at initialization.

        Parameters
        ----------
        X_train : np.ndarray
            Training features (scaled).
        y_train : np.ndarray
            Training labels (one-hot encoded).
        X_test : np.ndarray
            Test features used for progress logging during training.

        Returns
        -------
        CounterGAN
            The fitted instance (for method chaining).
        """
        input_dim = X_train.shape[1]

        self._discriminator = self._backend.create_discriminator(input_dim)
        self._generator = self._backend.create_generator(input_dim, self._residuals)
        batches = self._infinite_data_stream(X_train, y_train, batch_size=256)

        self._train(batches, X_test)
        self._is_fitted = True
        return self

    def generate_counterfactuals(self, X: np.ndarray) -> np.ndarray:
        """Generate counterfactuals for the given samples.

        Parameters
        ----------
        X : np.ndarray
            Input samples (scaled), shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Counterfactual samples with immutable features projected back
            to original values.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before "
                "generate_counterfactuals()."
            )

        counterfactuals = self._backend.predict(self._fitted_generator, X)

        # Immutable feature projection
        counterfactuals[:, self._n_mutable_features :] = X[
            :, self._n_mutable_features :
        ]

        return counterfactuals

    @staticmethod
    def _data_stream(x: np.ndarray, y: np.ndarray | None = None, batch_size: int = 500):
        """Yield batches until the dataset is exhausted."""
        n = x.shape[0]
        n_batches = (n + batch_size - 1) // batch_size
        perm = np.random.permutation(n)
        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            yield (x[idx], y[idx]) if y is not None else x[idx]

    def _infinite_data_stream(
        self, x: np.ndarray, y: np.ndarray | None = None, batch_size: int = 500
    ):
        """Infinitely yield batches, reshuffling on each pass."""
        batches = self._data_stream(x, y, batch_size=batch_size)
        while True:
            try:
                yield next(batches)
            except StopIteration:
                batches = self._data_stream(x, y, batch_size=batch_size)
                yield next(batches)

    def _train(self, batches, X_test: np.ndarray) -> None:
        """Run the GAN training loop.

        Parameters
        ----------
        batches
            Infinite data stream yielding ``(x_batch, y_batch)`` tuples.
        X_test : np.ndarray
            Test set used for periodic progress logging.
        """
        gen_optimizer = self._backend.create_generator_optimizer(
            self._fitted_generator, self._weighted_version
        )
        bce, cce = self._backend.create_loss_functions()

        def _has_diverged(x: np.ndarray) -> bool:
            return bool(np.all(np.isnan(x)))

        def _log_progress(iteration: int) -> None:
            X_gen = self._backend.predict(self._fitted_generator, X_test)
            clf_pred_test = self._classifier.predict(X_test)
            clf_pred = self._classifier.predict(X_gen)
            delta = (clf_pred - clf_pred_test)[:, self._desired_class]
            logger.info("=" * 88)
            logger.info("Training iteration %d at %s", iteration, datetime.now())
            logger.info(
                "Counterfactual prediction gain   (↑ is better): %.3f",
                delta.mean(),
            )
            logger.info(
                "Sparsity (L1, ↓ is better):                     %.3f",
                np.mean(np.abs(X_gen - X_test)),
            )

        last_batch = None

        for iteration in range(self._n_iterations):
            if iteration > 0 and last_batch is not None:
                gen_pred = self._backend.predict(
                    self._fitted_generator, last_batch
                )
                if _has_diverged(gen_pred):
                    logger.warning("Training diverged — stopping early.")
                    break

            if (iteration % 1000 == 0) or (iteration == self._n_iterations - 1):
                _log_progress(iteration)

            for _ in range(self._n_discriminator_steps):
                last_batch, _ = next(batches)
                self._backend.discriminator_step(
                    self._fitted_generator,
                    self._fitted_discriminator,
                    self._classifier,
                    last_batch,
                    self._weighted_version,
                    self._desired_class,
                )

            for _ in range(self._n_generator_steps):
                last_batch, _ = next(batches)
                self._backend.generator_step(
                    self._fitted_generator,
                    self._fitted_discriminator,
                    gen_optimizer,
                    self._classifier,
                    last_batch,
                    self._weighted_version,
                    self._desired_class,
                    self._number_of_classes,
                )
