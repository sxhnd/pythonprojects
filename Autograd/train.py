import random

import matplotlib

matplotlib.use("Agg")  # write the PNG without needing a display
import matplotlib.pyplot as plt

from nn import MLP, mse_loss
from optim import SGD, Adam, Momentum

XS = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
YS = [-1.0, 1.0, 1.0, -1.0]
SEED = 0
STEPS = 300


def train(make_optimizer):
    # Same seed before building each model, so every optimizer starts
    # from the same weights.
    random.seed(SEED)
    model = MLP(2, [4, 4, 1])
    optimizer = make_optimizer(model.parameters())

    losses = []
    for _ in range(STEPS):
        # Full batch: all 4 XOR points every step.
        loss = mse_loss([model(x) for x in XS], YS)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data)
    return model, losses


def main():
    runs = {
        "SGD (lr 0.1)": lambda params: SGD(params, lr=0.1),
        "Momentum (lr 0.1, beta 0.9)": lambda params: Momentum(params, lr=0.1, beta=0.9),
        "Adam (lr 0.05)": lambda params: Adam(params, lr=0.05),
    }
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for (name, make_optimizer), color in zip(runs.items(), colors):
        model, losses = train(make_optimizer)
        predictions = [model(x).data for x in XS]
        print(f"{name}: final loss {losses[-1]:.6f}")
        for x, y, p in zip(XS, YS, predictions):
            print(f"  {x} -> {p:+.3f} (target {y:+.0f})")
        ax.plot(losses, color=color, linewidth=2, label=name)
        ax.annotate(name.split(" ")[0], (STEPS - 1, losses[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color="#52514e")

    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE loss (log scale)")
    ax.set_title("XOR training loss, same starting weights")
    ax.grid(True, which="major", color="#e6e5e0", linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("loss_curves.png", dpi=150)
    print("Saved loss_curves.png")


if __name__ == "__main__":
    main()
