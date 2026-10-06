import random

from engine import Value
from gradcheck import gradcheck
from nn import MLP, Layer, Neuron, mse_loss


def test_neuron_output_is_in_tanh_range():
    random.seed(0)
    out = Neuron(3)([1.0, -2.0, 0.5])
    assert isinstance(out, Value)
    assert -1 < out.data < 1


def test_layer_output_size():
    random.seed(0)
    assert len(Layer(3, 4)([1.0, 2.0, 3.0])) == 4
    assert isinstance(Layer(3, 1)([1.0, 2.0, 3.0]), Value)


def test_mlp_output_is_a_single_value():
    random.seed(0)
    out = MLP(2, [4, 4, 1])([1.0, 0.0])
    assert isinstance(out, Value)


def test_parameter_counts():
    # each neuron has one weight per input plus a bias
    assert len(Neuron(3).parameters()) == 4
    assert len(Layer(3, 4).parameters()) == 16
    # (2*4 + 4) + (4*4 + 4) + (4*1 + 1) = 12 + 20 + 5
    assert len(MLP(2, [4, 4, 1]).parameters()) == 37


def test_zero_grad():
    random.seed(0)
    model = MLP(2, [3, 1])
    model([1.0, -1.0]).backward()
    assert any(p.grad != 0 for p in model.parameters())
    model.zero_grad()
    assert all(p.grad == 0 for p in model.parameters())


def test_mse_loss():
    loss = mse_loss([Value(1.0), Value(-1.0)], [0.0, 1.0])
    # ((1 - 0)**2 + (-1 - 1)**2) / 2 = (1 + 4) / 2
    assert loss.data == 2.5


def test_gradcheck_on_mlp_loss():
    random.seed(1)
    model = MLP(2, [3, 3, 1])
    xs = [[0.5, -1.0], [1.5, 0.2], [-0.3, 0.8]]
    ys = [1.0, -1.0, 1.0]

    # The parameters are the same Value objects the model uses, so nudging
    # them inside gradcheck changes the model's output.
    def loss(params):
        return mse_loss([model(x) for x in xs], ys)

    assert gradcheck(loss, model.parameters()) < 1e-6
