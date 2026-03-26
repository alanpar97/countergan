"""TensorFlow / Keras backend for CounterGAN.

All TensorFlow and Keras imports are contained in this module so that the
rest of the package remains framework-agnostic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf
import keras
from keras import Model, optimizers
from keras.layers import (
    ActivityRegularization,
    Add,
    Dense,
    Dropout,
    Input,
)
from keras.models import Sequential
from keras.utils import to_categorical

from countergan._backend import Backend


class TensorFlowBackend(Backend):
    """Backend implementation using TensorFlow / Keras."""

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------

    def create_generator(self, input_dim: int, residuals: bool) -> Model:
        input_shape = (input_dim,)
        generator_input = Input(shape=input_shape, name="generator_input")
        h = Dense(64, activation="relu")(generator_input)
        h = Dense(32, activation="relu")(h)
        h = Dense(64, activation="relu")(h)
        h = Dense(input_dim, activation="tanh")(h)
        generator_output = ActivityRegularization(l1=0.0, l2=1e-6)(h)

        if residuals:
            generator_output = Add(name="output")(
                [generator_input, generator_output]
            )

        return Model(inputs=generator_input, outputs=generator_output)

    def create_discriminator(self, input_dim: int) -> Sequential:
        input_shape = (input_dim,)
        model = Sequential(
            [
                Input(shape=input_shape),
                Dense(128, activation="relu"),
                Dropout(0.2),
                Dense(1, activation="sigmoid"),
            ],
            name="discriminator",
        )
        optimizer = optimizers.Adam(learning_rate=0.0005, beta_1=0.5)
        model.compile(optimizer, "binary_crossentropy")
        return model

    # ------------------------------------------------------------------
    # Optimizers & losses
    # ------------------------------------------------------------------

    def create_generator_optimizer(
        self, generator: Any, weighted_version: bool
    ) -> Any:
        lr = 5e-4 if weighted_version else 2e-4
        return optimizers.RMSprop(learning_rate=lr)

    def create_loss_functions(self) -> tuple[Any, Any]:
        return (
            keras.losses.BinaryCrossentropy(),
            keras.losses.CategoricalCrossentropy(),
        )

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
        x_fake = generator.predict(x_real)

        x_batch = np.concatenate([x_real, x_fake])
        y_batch = np.concatenate([np.ones(len(x_real)), np.zeros(len(x_fake))])
        p = np.random.permutation(len(y_batch))
        x_batch, y_batch = x_batch[p], y_batch[p]

        x_t = tf.constant(x_batch, dtype=tf.float32)
        y_t = tf.constant(y_batch.reshape(-1, 1), dtype=tf.float32)

        sample_weights = None
        if weighted:
            scores = classifier.predict(x_batch)[:, desired_class]
            real_idx = np.where(y_batch == 1.0)
            scores[real_idx] /= np.mean(scores[real_idx])
            scores[np.where(y_batch == 0.0)] = 1.0
            sample_weights = tf.constant(scores, dtype=tf.float32)

        with tf.GradientTape() as tape:
            y_pred = discriminator(x_t, training=True)
            if weighted:
                per_sample = keras.losses.binary_crossentropy(y_t, y_pred)
                d_loss = tf.reduce_mean(per_sample * sample_weights)
            else:
                bce = keras.losses.BinaryCrossentropy()
                d_loss = bce(y_t, y_pred)

        grads = tape.gradient(d_loss, discriminator.trainable_weights)
        discriminator.optimizer.apply_gradients(
            zip(grads, discriminator.trainable_weights)
        )

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
        y_fake = np.ones(len(x_input))

        x_t = tf.constant(x_input, dtype=tf.float32)
        y_fake_t = tf.constant(y_fake.reshape(-1, 1), dtype=tf.float32)

        bce = keras.losses.BinaryCrossentropy()
        cce = keras.losses.CategoricalCrossentropy()

        with tf.GradientTape() as tape:
            x_gen = generator(x_t, training=True)
            disc_pred = discriminator(x_gen, training=False)

            if weighted:
                g_loss = bce(y_fake_t, disc_pred)
            else:
                clf_pred = classifier.model(x_gen, training=False)
                y_target = to_categorical(
                    [desired_class] * len(x_input),
                    num_classes=n_classes,
                )
                y_target_t = tf.constant(y_target, dtype=tf.float32)
                g_loss = bce(y_fake_t, disc_pred) + cce(y_target_t, clf_pred)

        grads = tape.gradient(g_loss, generator.trainable_weights)
        gen_optimizer.apply_gradients(zip(grads, generator.trainable_weights))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, model: Any, x: np.ndarray) -> np.ndarray:
        return model.predict(x, verbose=0)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_classifier(self, classifier: Any, needs_gradient_flow: bool) -> None:
        if not hasattr(classifier, "predict"):
            raise TypeError(
                "Classifier must have a predict(X) method that returns "
                "class probabilities as a NumPy array."
            )
        if needs_gradient_flow:
            if not hasattr(classifier, "model"):
                raise TypeError(
                    "Strategy 'countergan' requires the classifier to expose a "
                    "'.model' attribute (the underlying Keras Model) for "
                    "gradient computation."
                )
