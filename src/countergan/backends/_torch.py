"""PyTorch backend for CounterGAN.

All PyTorch imports are contained in this module so that the rest of the
package remains framework-agnostic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from countergan._backend import Backend


# ---------------------------------------------------------------------------
# nn.Module definitions
# ---------------------------------------------------------------------------


class Generator(nn.Module):
    """Feed-forward generator with optional residual connection."""

    def __init__(self, input_dim: int, residuals: bool = True) -> None:
        super().__init__()
        self.residuals = residuals
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        if self.residuals:
            return x + raw
        return raw

    def forward_with_raw(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(output, raw_pre_residual)`` for activity regularization."""
        raw = self.net(x)
        out = x + raw if self.residuals else raw
        return out, raw


class Discriminator(nn.Module):
    """Feed-forward discriminator."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Backend implementation
# ---------------------------------------------------------------------------


class TorchBackend(Backend):
    """Backend implementation using PyTorch."""

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------

    def create_generator(self, input_dim: int, residuals: bool) -> Generator:
        return Generator(input_dim, residuals)

    def create_discriminator(self, input_dim: int) -> Discriminator:
        return Discriminator(input_dim)

    # ------------------------------------------------------------------
    # Optimizers & losses
    # ------------------------------------------------------------------

    def create_generator_optimizer(
        self, generator: Any, weighted_version: bool
    ) -> optim.Optimizer:
        lr = 5e-4 if weighted_version else 2e-4
        return optim.RMSprop(generator.parameters(), lr=lr)

    def create_loss_functions(self) -> tuple[nn.BCELoss, nn.BCELoss]:
        # Both bce and cce operate on probability vectors.  For the
        # categorical cross-entropy we implement it manually in the
        # generator step to match the Keras behavior (takes probabilities,
        # not logits).  We return bce twice as a placeholder; the actual
        # cce computation is inlined.
        return nn.BCELoss(), nn.BCELoss()

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------

    def discriminator_step(
        self,
        generator: Any,
        discriminator: Any,
        classifier: Any,
        x_real: np.ndarray,
        weighted: bool,
        desired_class: int,
    ) -> None:
        generator.eval()
        discriminator.train()

        with torch.no_grad():
            x_real_t = torch.tensor(x_real, dtype=torch.float32)
            x_fake_t = generator(x_real_t)
        x_fake = x_fake_t.numpy()

        x_batch = np.concatenate([x_real, x_fake])
        y_batch = np.concatenate([np.ones(len(x_real)), np.zeros(len(x_fake))])
        p = np.random.permutation(len(y_batch))
        x_batch, y_batch = x_batch[p], y_batch[p]

        x_t = torch.tensor(x_batch, dtype=torch.float32)
        y_t = torch.tensor(y_batch.reshape(-1, 1), dtype=torch.float32)

        # Discriminator optimizer (Adam, same hyperparams as Keras backend)
        if not hasattr(discriminator, "_optim"):
            discriminator._optim = optim.Adam(
                discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999)
            )

        discriminator._optim.zero_grad()
        y_pred = discriminator(x_t)

        if weighted:
            scores = classifier.predict_proba(x_batch)[:, desired_class]
            real_idx = np.where(y_batch == 1.0)
            scores[real_idx] /= np.mean(scores[real_idx])
            scores[np.where(y_batch == 0.0)] = 1.0
            sw = torch.tensor(scores, dtype=torch.float32)

            per_sample = F.binary_cross_entropy(y_pred, y_t, reduction="none").squeeze()
            d_loss = torch.mean(per_sample * sw)
        else:
            d_loss = F.binary_cross_entropy(y_pred, y_t)

        d_loss.backward()
        discriminator._optim.step()

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
        generator.train()
        discriminator.eval()

        x_t = torch.tensor(x_input, dtype=torch.float32)
        y_fake_t = torch.ones(len(x_input), 1)

        gen_optimizer.zero_grad()

        x_gen, raw = generator.forward_with_raw(x_t)
        disc_pred = discriminator(x_gen)

        g_loss = F.binary_cross_entropy(disc_pred, y_fake_t)

        if not weighted:
            clf_pred = classifier.model(x_gen)
            # Build one-hot target
            y_target = torch.zeros(len(x_input), n_classes)
            y_target[:, desired_class] = 1.0
            # Categorical cross-entropy on probabilities (matches Keras behavior)
            g_loss = g_loss - torch.mean(
                torch.sum(y_target * torch.log(clf_pred + 1e-7), dim=1)
            )

        # Activity regularization: L2 penalty on raw (pre-residual) output
        g_loss = g_loss + 1e-6 * torch.mean(raw ** 2)

        g_loss.backward()
        gen_optimizer.step()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, model: Any, x: np.ndarray) -> np.ndarray:
        model.eval()
        x_t = torch.tensor(x, dtype=torch.float32)
        return model(x_t).numpy()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_classifier(self, classifier: Any, needs_gradient_flow: bool) -> None:
        if not hasattr(classifier, "predict_proba"):
            raise TypeError(
                "Classifier must have a predict_proba(X) method that returns "
                "class probabilities as a NumPy array."
            )
        if needs_gradient_flow:
            if not hasattr(classifier, "model"):
                raise TypeError(
                    "Strategy 'countergan' requires the classifier to expose a "
                    "'.model' attribute (a torch.nn.Module) for gradient "
                    "computation."
                )
            model = classifier.model
            if not isinstance(model, nn.Module):
                raise TypeError(
                    f"Expected classifier.model to be a torch.nn.Module, "
                    f"got {type(model).__name__}. Make sure you are using a "
                    f"PyTorch classifier with the 'torch' backend."
                )
