# ML Lightning Template

A template for machine learning projects built using PyTorch Lightning and Hydra.

## Tech Stack

- **ML Framework**: [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning)
- **Configuration**: [Hydra](https://github.com/facebookresearch/hydra)
- **Package/Project Manager**: [uv](https://github.com/astral-sh/uv)
- **Task Runner**: [taskipy](https://github.com/taskipy/taskipy)
- **Testing**: [pytest](https://github.com/pytest-dev/pytest)
- **Code Quality**: [Ruff](https://github.com/astral-sh/ruff), [Mypy](https://github.com/python/mypy), [pre-commit](https://github.com/pre-commit/pre-commit)

## Requirements

```bash
$ uv --version
uv 0.5.18 (27d1bad55 2025-01-11)
```

## Installation

To use this template, you can either click the [<kbd>Use this template</kbd>](https://github.com/k-kuroguro/ml-lightning-template/generate) button to generate a GitHub repository or run the following command to generate the local project:

```bash
$ uvx copier copy gh:k-kuroguro/ml-lightning-template my_project
```

Next, set up the project:

```bash
$ cd my_project
$ uv run task setup
```

## Usage

### Training the Model

To train the model:

```bash
uv run src/train.py
```

To switch configurations for different experiments, add your configuration file under the `configs/experiment` directory and run:

```bash
uv run src/train.py experiment=my_exp
```

For detailed information on the configuration files, please refer to the [Hydra framework documentation](https://hydra.cc/docs/intro/).

### Evaluating the Model

To evaluate the model, specify the path to the checkpoint:

```bash
uv run src/eval.py ckpt_path=path/to/checkpoint
```

## References

This template is based on the following projects:

- [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)
