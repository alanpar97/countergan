"""Abstract backend interface for framework-specific GAN operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Backend(ABC):
    """Abstract interface that each framework backend must implement.

    Every method that touches framework-specific tensors, models, optimizers,
    or gradient computation lives here.  The :class:`CounterGAN` orchestrator
    calls these methods without knowing which framework is underneath.
    """

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------

    @abstractmethod
    def create_generator(self, input_dim: int, residuals: bool) -> Any:
        """Build the generator network.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        residuals : bool
            If ``True``, add a residual (skip) connection from input to output.

        Returns
        -------
        Any
            A framework-native model object.
        """

    @abstractmethod
    def create_discriminator(self, input_dim: int) -> Any:
        """Build and configure the discriminator network.

        Parameters
        ----------
        input_dim : int
            Number of input features.

        Returns
        -------
        Any
            A framework-native model object (compiled, where applicable).
        """

    # ------------------------------------------------------------------
    # Optimizers & losses
    # ------------------------------------------------------------------

    @abstractmethod
    def create_generator_optimizer(self, generator: Any, weighted_version: bool) -> Any:
        """Create the optimizer used for the generator update step.

        Parameters
        ----------
        generator
            The generator model whose parameters will be optimized.
        weighted_version : bool
            ``True`` for the weighted countergan variant (uses higher LR).

        Returns
        -------
        Any
            A framework-native optimizer.
        """

    @abstractmethod
    def create_loss_functions(self) -> tuple[Any, Any]:
        """Return ``(bce, cce)`` loss callables.

        * ``bce`` — binary cross-entropy
        * ``cce`` — categorical cross-entropy (operates on probability vectors)

        Returns
        -------
        tuple[Any, Any]
        """

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------

    @abstractmethod
    def discriminator_step(
        self,
        generator: Any,
        discriminator: Any,
        classifier: Any,
        x_real: np.ndarray,
        weighted: bool,
        desired_class: int,
    ) -> None:
        """Perform one discriminator gradient update.

        Parameters
        ----------
        generator
            The current generator model (used to produce fake samples).
        discriminator
            The discriminator model to update.
        classifier
            User-provided classifier wrapper (needs ``.predict_proba()`` for the
            weighted variant).
        x_real : np.ndarray
            Batch of real samples.
        weighted : bool
            Whether to use sample-weighted loss (``countergan_wt``).
        desired_class : int
            Target class index.
        """

    @abstractmethod
    def generator_step(
        self,
        generator: Any,
        discriminator: Any,
        gen_optimizer: Any,
        classifier: Any,
        x_input: np.ndarray,
        weighted: bool,
        desired_class: int,
        n_classes: int,
    ) -> None:
        """Perform one generator gradient update.

        Parameters
        ----------
        generator
            The generator model to update.
        discriminator
            The discriminator model (frozen during this step).
        gen_optimizer
            Optimizer for the generator weights.
        classifier
            User-provided classifier wrapper (needs ``.model`` for the
            non-weighted variant).
        x_input : np.ndarray
            Batch of input samples.
        weighted : bool
            Whether to use the weighted (model-agnostic) formulation.
        desired_class : int
            Target class index.
        n_classes : int
            Total number of classes.
        """

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @abstractmethod
    def predict(self, model: Any, x: np.ndarray) -> np.ndarray:
        """Run inference and return a NumPy array.

        Parameters
        ----------
        model
            A framework-native model.
        x : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
        """

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @abstractmethod
    def validate_classifier(self, classifier: Any, needs_gradient_flow: bool) -> None:
        """Check that the user-provided classifier is compatible.

        Parameters
        ----------
        classifier
            The classifier wrapper passed to :class:`CounterGAN`.
        needs_gradient_flow : bool
            ``True`` when ``strategy="countergan"`` — the classifier's
            ``.model`` must be a differentiable framework-native model.

        Raises
        ------
        TypeError
            If the classifier does not satisfy the backend's requirements.
        """
