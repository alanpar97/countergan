import argparse
import logging
import os

import numpy as np
import pandas as pd

from example_utils import load_data
from countergan import CounterGAN

BASE_PATH = os.environ.get("BASE_PATH", os.getcwd())

INITIAL_CLASS = 0
DESIRED_CLASS = 1
N_CLASSES = 2

RANDOM_SEED = 2020
np.random.seed(RANDOM_SEED)


logger = logging.getLogger(__name__)


def _get_classifier(backend: str, input_shape: tuple[int, ...]):
    """Build a classifier for the chosen backend."""
    if backend == "torch":
        import torch

        torch.manual_seed(RANDOM_SEED)
        from classifier_torch import Classifier
    else:
        import tensorflow as tf

        tf.random.set_seed(RANDOM_SEED)
        tf.keras.utils.disable_interactive_logging()
        from classifier_tf import Classifier

    return Classifier(input_shape)


def main():
    parser = argparse.ArgumentParser(description="CounterGAN example")
    parser.add_argument(
        "--backend",
        choices=["torch", "tensorflow"],
        default=None,
        help="Backend to use (auto-detected if omitted)",
    )
    args = parser.parse_args()

    # --- Data ---
    logger.info("[1/5] Loading and preprocessing data...")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        features,
        mutable_features,
        immutable_features,
        scaler,
    ) = load_data(BASE_PATH, RANDOM_SEED)
    input_shape = (X_train.shape[1],)
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Resolve backend (auto-detect if not specified)
    from countergan.backends import detect_backend

    backend = detect_backend(args.backend)
    logger.info(f"Using backend: {backend}")

    logger.info("[2/5] Training classifier...")
    model = _get_classifier(backend, input_shape)
    model.fit(X_train, y_train, X_test, y_test)

    logger.info("[3/5] Training autoencoder (density estimator)...")

    logger.info("[4/5] Training CounterGAN...")
    gan = CounterGAN(
        strategy="countergan",
        classifier=model,
        n_mutable_features=len(mutable_features),
        n_discriminator_steps=2,
        n_iterations=2000,
        desired_class=DESIRED_CLASS,
        number_of_classes=N_CLASSES,
        backend=backend,
    )
    gan.fit(X_train, y_train, X_test)

    # --- Individual example ---
    logger.info(
        "\n[5/5] Showing individual example with perturbations relative to original..."
    )
    sample_idx = 20
    sample = np.expand_dims(X_test[sample_idx], axis=0)

    orig_values = scaler.inverse_transform(X_test)[sample_idx]
    counterfactual: np.ndarray = gan.generate_counterfactuals(sample)

    cf_values = scaler.inverse_transform(counterfactual)[0]
    deltas = cf_values - orig_values
    deltas[-len(immutable_features) :] = 0.0

    original = dict(zip(features, orig_values))
    original["Classifier Prediction"] = float(model.predict_proba(sample)[0][1])

    cf = dict(zip(features, cf_values))
    cf["Classifier Prediction"] = float(model.predict_proba(counterfactual)[0][1])

    delta = dict(zip(features, deltas))
    delta["Classifier Prediction"] = (
        cf["Classifier Prediction"] - original["Classifier Prediction"]
    )

    combined = pd.DataFrame(
        {"Original": original, "Counterfactual": cf, "Delta": delta}
    )
    logger.info("\nIndividual example:")
    print(combined.to_string())


if __name__ == "__main__":
    main()
