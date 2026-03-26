# CounteRGAN

A standalone Python package for **CounteRGAN** — a GAN-based method for generating counterfactual explanations.

Based on the paper:

> Nemirovsky, D., Thiebaut, N., Xu, Y., & Gupta, A. (2022, August). **CounteRGAN: Generating counterfactuals for real-time recourse and interpretability using residual GANs** In *Uncertainty in Artificial Intelligence* (pp. 1488-1497). PMLR.
>
> [arXiv:2009.05199](https://arxiv.org/abs/2009.05199)

This repository is a packaged version of the [original CounteRGAN code](https://github.com/nkthiebaut/countergan), restructured as an installable Python library.

## Installation

The base package depends only on NumPy. Install with your preferred backend:

```bash
# PyTorch (default / recommended)
pip install "countergan[torch]"

# TensorFlow / Keras
pip install "countergan[tensorflow]"
```

To also install the dependencies needed to run the example:

```bash
pip install "countergan[torch]" pandas scikit-learn
```

> Requires Python 3.12+.

## Backends

CounterGAN supports two backends — **PyTorch** and **TensorFlow/Keras**. Only one needs to be installed. When both are available, PyTorch is preferred by default.

You can let the backend be auto-detected or select it explicitly:

```python
# Auto-detect (prefers torch if both are installed)
gan = CounterGAN(strategy="countergan", classifier=clf, ...)

# Explicit selection
gan = CounterGAN(strategy="countergan", classifier=clf, ..., backend="torch")
gan = CounterGAN(strategy="countergan", classifier=clf, ..., backend="tensorflow")
```

### Classifier requirements

The classifier you pass to `CounterGAN` must satisfy a simple interface:

| Method / attribute | Required by | Description |
|---|---|---|
| `.predict(X) -> np.ndarray` | All strategies | Return class probabilities, shape `(n_samples, n_classes)` |
| `.model` | `"countergan"` strategy only | The underlying framework model (`torch.nn.Module` or `keras.Model`) for gradient flow |

The `"countergan_wt"` and `"regular_gan"` strategies only call `.predict()`, so any classifier works. The `"countergan"` strategy backpropagates through the classifier, so `.model` must match the backend (e.g. an `nn.Module` for torch).

## Quick start

```python
from countergan import CounterGAN

# Given a pre-trained classifier and your data:
gan = CounterGAN(
    strategy="countergan",
    classifier=classifier,
    n_mutable_features=5,
    desired_class=1,
    number_of_classes=2,
)
gan.fit(X_train, y_train, X_test)

counterfactuals = gan.generate_counterfactuals(X_test)
```

## Strategies

CounterGAN supports three training strategies:

| Strategy | Description |
|---|---|
| `"regular_gan"` | Plain GAN without residual connections. |
| `"countergan"` | Backpropagates through the classifier during generator training. Produces counterfactuals that are both realistic and classified as the desired class. |
| `"countergan_wt"` | Model-agnostic variant. Uses classifier scores as sample weights for the discriminator loss instead of backpropagating through the classifier. |

## API

### `CounterGAN(strategy, classifier, n_mutable_features, ...)`

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `strategy` | *(required)* | `"regular_gan"`, `"countergan"`, or `"countergan_wt"` |
| `classifier` | *(required)* | Pre-trained classifier (see [Classifier requirements](#classifier-requirements)) |
| `n_mutable_features` | *(required)* | Number of mutable features. Features must be ordered mutable-first, immutable-last |
| `n_discriminator_steps` | `2` | Discriminator updates per training iteration |
| `n_generator_steps` | `4` | Generator updates per training iteration |
| `n_iterations` | `2000` | Total training iterations |
| `desired_class` | `1` | Target class for counterfactuals |
| `number_of_classes` | `2` | Total number of classes |
| `backend` | `None` | `"torch"`, `"tensorflow"`, or `None` (auto-detect) |

**Methods:**

- **`fit(X_train, y_train, X_test)`** — Train the GAN. Returns `self` for method chaining.
- **`generate_counterfactuals(X)`** — Generate counterfactual samples. Immutable features are projected back to their original values.

**Properties:**

- **`generator`** — The trained generator model (`None` before `fit()` is called).

## Example

A full working example using the Pima Indians Diabetes dataset is in the [`example/`](example/) directory.

```bash
cd example
python example.py                     # auto-detects backend
python example.py --backend torch      # force PyTorch
python example.py --backend tensorflow # force TensorFlow
```

The example:
1. Loads and preprocesses the diabetes dataset
2. Trains a binary classifier (PyTorch or Keras, depending on the backend)
3. Trains a CounterGAN with the `"countergan"` strategy
4. Generates a counterfactual for a test sample and displays the feature perturbations

## Citation

```bibtex
@inproceedings{nemirovsky2022countergan,
  title={CounteRGAN: Generating counterfactuals for real-time recourse and interpretability using residual GANs},
  author={Nemirovsky, Daniel and Thiebaut, Nicolas and Xu, Ye and Gupta, Abhishek},
  booktitle={Uncertainty in Artificial Intelligence},
  pages={1488--1497},
  year={2022},
  organization={PMLR}
}
```
