# Autograd

A scalar autograd engine built from scratch. Currently supports addition,
subtraction, multiplication, division, powers with a number exponent, tanh, ReLU,
exp and log, along with backpropagation and numerical gradient checking. It also
has a small neural network library (neurons, layers, a multilayer perceptron and
mean squared error loss), plus SGD and Momentum optimizers. An Adam optimizer is
planned.

## Running the tests

```bash
pip install pytest
python -m pytest
```
